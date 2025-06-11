import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
import json
from utils.prompter import Prompter
import warnings

class RoadInspectionDataset(Dataset):
    def __init__(self, data_path, image_base_path, processor, prompter, instruction_type="full", cutoff_len=2048, add_eos_token=False):
        self.data_path = data_path
        self.image_base_path = image_base_path
        self.processor = processor
        self.prompter = prompter
        self.instruction_type = instruction_type
        self.cutoff_len = cutoff_len
        self.add_eos_token = add_eos_token
        # プロンプトの応答部分を特定するための文字列
        self.response_key = "### Response:"

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]

        # 1. 画像のロードと前処理
        relative_image_path = data_point["input"]
        
        # パスのクリーンアップ
        # data/プレフィックスの削除
        if relative_image_path.startswith("data/"):
            relative_image_path = relative_image_path[len("data/"):]
        
        # パスの正規化
        # Road_inspection/Road_inspection/train/... の形式を維持
        parts = relative_image_path.split('/')
        if len(parts) >= 1 and parts[0] == "Road_inspection":
            if len(parts) >= 2 and parts[1] == "Road_inspection":
                # 正しい形式なのでそのまま使用
                pass
            else:
                # Road_inspectionを追加
                parts.insert(1, "Road_inspection")
                relative_image_path = '/'.join(parts)
        
        image_path = os.path.join(self.image_base_path, relative_image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Returning dummy data for index {idx}.")
            raise FileNotFoundError(f"Image not found: {image_path} for data point index {idx}")

        # 2. テキストプロンプトの生成
        instruction = data_point.get("instruction", "")
        text_input_for_prompter = data_point.get("input", "")
        output_text = data_point.get("output", "")

        if self.instruction_type == "full":
            full_text_prompt = self.prompter.generate_prompt(instruction, text_input_for_prompter, output_text)
        elif self.instruction_type == "no_instruction":
            if not instruction:
                instruction_for_llava = "Is there a crack in this road image? Output 'Abnormal' if there is a crack, otherwise 'No abnormality'."
            else:
                instruction_for_llava = instruction

            full_text_prompt = self.prompter.generate_prompt(instruction_for_llava, text_input_for_prompter, output_text)
        else:
            raise ValueError("instruction_type must be 'full' or 'no_instruction'")

        # 3. LLaVAプロセッサによる画像とテキストの処理
        # LLaVAの要件に合わせてプロンプトを修正
        # <image>タグを追加してLLaVAに画像の位置を認識させる
        if not full_text_prompt.startswith("<image>"):
            full_text_prompt = "<image>\n" + full_text_prompt

        # LLaVAプロセッサで画像とテキストを同時に処理
        encoding = self.processor(
            text=full_text_prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=False,  # 切り捨てを無効化
            max_length=self.cutoff_len
        )

        # バッチ次元を削除
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)

        # 4. labelsの準備
        labels = encoding["input_ids"].clone()

        # プロンプトの応答部分を特定
        response_start_idx = full_text_prompt.find(self.response_key)
        if response_start_idx != -1:
            # response_keyまでのテキストをトークン化して長さを取得
            # response_keyの後の改行とスペースも含める
            response_key_end = response_start_idx + len(self.response_key)
            while response_key_end < len(full_text_prompt) and full_text_prompt[response_key_end] in ['\n', ' ', '\r']:
                response_key_end += 1
            
            prefix_tokens = self.processor.tokenizer(
                full_text_prompt[:response_key_end],
                add_special_tokens=True,  # 特殊トークンを含める
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            
            # response_key以前のトークンをマスク
            labels[:len(prefix_tokens)] = -100

            if self.add_eos_token and len(labels) > 0 and labels[-1] == self.processor.tokenizer.eos_token_id:
                labels[-1] = -100
        else:
            warnings.warn(f"Response split '{self.response_key}' not found in prompt for index {idx}. Labels might be incorrectly masked. Full prompt: {full_text_prompt}")
            # 応答部分が見つからない場合は、出力テキストの部分だけを学習対象にする
            output_tokens = self.processor.tokenizer(
                output_text,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
            
            labels[:] = -100  # 一旦すべてマスク
            if len(output_tokens) > 0:
                # 出力テキストの位置を特定してマスクを解除
                output_str = self.processor.tokenizer.decode(output_tokens)
                full_tokens = self.processor.tokenizer.decode(encoding["input_ids"])
                output_start = full_tokens.find(output_str)
                if output_start != -1:
                    # 出力テキストの位置を特定してマスクを解除
                    output_token_ids = self.processor.tokenizer(
                        output_str,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"].squeeze(0)
                    labels[-len(output_token_ids):] = encoding["input_ids"][-len(output_token_ids):]

        encoding["labels"] = labels

        return encoding

def collate_fn(batch, processor=None):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(len(x) for x in input_ids)

    # パディングトークンIDをprocessorから取得（利用可能な場合）
    if processor is not None and hasattr(processor.tokenizer, 'pad_token_id'):
        actual_pad_token_id = processor.tokenizer.pad_token_id
    else:
        actual_pad_token_id = 0

    padded_input_ids = torch.full((len(input_ids), max_len), 
                                  fill_value=actual_pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.full((len(attention_mask), max_len), 
                                       fill_value=0, dtype=torch.long)
    padded_labels = torch.full((len(labels), max_len), 
                               fill_value=-100, dtype=torch.long)

    for i, (ids, attn, lbl) in enumerate(zip(input_ids, attention_mask, labels)):
        padded_input_ids[i, :len(ids)] = ids
        padded_attention_mask[i, :len(attn)] = attn
        padded_labels[i, :len(lbl)] = lbl

    pixel_values = torch.stack([item['pixel_values'] for item in batch])

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'pixel_values': pixel_values,
        'labels': padded_labels
    }

# データセットのパス設定例
if __name__ == '__main__':
    print("--- Testing RoadInspectionDataset and collate_fn ---")

    # 正しいパスの設定
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # dataディレクトリをベースとしてパスを設定
    data_dir = os.path.join(base_dir, "data")
    image_data_base_path = data_dir
    train_json_path = os.path.join(data_dir, "Road_inspection", "Road_inspection_full_train.json")
    test_json_path = os.path.join(data_dir, "Road_inspection", "Road_inspection_full_test.json")
    LLaVA_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        processor_for_test = AutoProcessor.from_pretrained(LLaVA_MODEL_PATH)
        prompter_for_test = Prompter(template_name="alpaca")

        train_dataset = RoadInspectionDataset(
            data_path=train_json_path,
            image_base_path=image_data_base_path,
            processor=processor_for_test,
            prompter=prompter_for_test,
            instruction_type="full",
            cutoff_len=2048
        )
        print(f"Loaded {len(train_dataset)} samples from {train_json_path}")

        processor = processor_for_test

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )
        print("DataLoader created.")

        for i, batch in enumerate(train_dataloader):
            print(f"\n--- Batch {i+1} ---")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Attention Mask shape: {batch['attention_mask'].shape}")
            print(f"Pixel Values shape: {batch['pixel_values'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")

            print("\n--- Decoded Sample 1 ---")
            print(f"Original Input IDs (first 50): {batch['input_ids'][0][:50]}")
            print(f"Original Labels (first 50): {batch['labels'][0][:50]}")

            decoded_input = processor_for_test.tokenizer.decode(
                batch['input_ids'][0], 
                skip_special_tokens=False
            )
            decoded_labels = processor_for_test.tokenizer.decode(
                batch['labels'][0].masked_fill(batch['labels'][0] == -100, processor_for_test.tokenizer.pad_token_id), 
                skip_special_tokens=False
            )
            print(f"Decoded Input (Sample 1):\n{decoded_input[:500]}...")
            print(f"Decoded Labels (Sample 1, masked -100):\n{decoded_labels[:500]}...")

            actual_response_tokens = batch['labels'][0][batch['labels'][0] != -100]
            if actual_response_tokens.numel() > 0:
                decoded_actual_response = processor_for_test.tokenizer.decode(
                    actual_response_tokens, 
                    skip_special_tokens=False
                )
                print(f"Decoded Actual Response (Sample 1):\n{decoded_actual_response}")
            else:
                print("No actual response tokens found (all masked or empty).")

            if i == 0:
                break

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your FDLI RoadInspection dataset is correctly placed.")
        print(f"Expected image_base_path: {image_data_base_path}")
        print(f"Expected train_json_path: {train_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred during data loading test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test finished ---")