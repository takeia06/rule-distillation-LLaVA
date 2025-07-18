# finetune_llava_v1_5_with_evaluation.py
# =================================================================
# 生徒モデル (LLaVA-v1.5) のファインチューニングスクリプト
# トレーニング後の自動評価機能付き (最終版)
# =================================================================

import torch
from PIL import Image
import os
import csv
from torch.utils.data import Dataset
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# --- 1. 設定 ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATA_DIR = "data/VisA/chewinggum/no"


# --- 2. データセットの準備 ---
class LlavaFinetuneDataset(Dataset):
    def __init__(self, data_dir, processor, split="train"):
        self.data_dir = data_dir
        self.processor = processor
        self.data = []
        csv_file_path = os.path.join(data_dir, f"{split}.csv")
        try:
            with open(csv_file_path, "r", newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    image_path, instruction, output = row[0], row[1], row[2]
                    if image_path == 'image_path': continue
                    if not instruction:
                        instruction = "Describe the image."
                    self.data.append({
                        "image_file": image_path,
                        "instruction": instruction,
                        "output": output
                    })
        except FileNotFoundError:
            print(f"!!! エラー: {split}.csv が見つかりません。パスを確認してください: {csv_file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file, instruction, output = item["image_file"], item["instruction"], item["output"]
        try:
            image = Image.open(image_file)
        except FileNotFoundError:
            return None

        prompt_for_processor = f"USER: <image>\n{instruction}\nASSISTANT: {output}"
        prompt_for_loss_calculation = f"USER: <image>\n{instruction}\nASSISTANT:"
        processed_inputs = self.processor(text=prompt_for_processor, images=image, return_tensors="pt")

        final_data = {
            "input_ids": processed_inputs["input_ids"].squeeze(0),
            "attention_mask": processed_inputs["attention_mask"].squeeze(0),
            "pixel_values": processed_inputs["pixel_values"].squeeze(0),
            "prompt_text": prompt_for_loss_calculation
        }
        return final_data

# --- 3. データコレータの準備 ---
def get_data_collator(processor):
    class DataCollatorForLLaVAFinetuning:
        def __init__(self, processor):
            self.processor = processor
        def __call__(self, features):
            features = [f for f in features if f is not None]
            if not features: return None

            prompt_texts = [f.pop("prompt_text") for f in features]
            input_ids_list = [f["input_ids"] for f in features]
            pixel_values_list = [f["pixel_values"] for f in features]

            padded_inputs = self.processor.tokenizer.pad(
                {"input_ids": input_ids_list}, padding=True, return_tensors="pt"
            )
            labels = padded_inputs["input_ids"].clone()
            for i in range(len(prompt_texts)):
                prompt_len = len(self.processor.tokenizer.encode(prompt_texts[i]))
                labels[i, :prompt_len] = -100
            return {
                "input_ids": padded_inputs["input_ids"],
                "attention_mask": padded_inputs["attention_mask"],
                "pixel_values": torch.stack(pixel_values_list),
                "labels": labels,
            }
    return DataCollatorForLLaVAFinetuning(processor)

# --- 4. モデルの読み込みとLoRAの設定 ---
print("--- 4-bit量子化モデルの読み込みを開始します ---")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=quantization_config, torch_dtype=torch.float16, device_map="auto"
)
processor = LlavaProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.pad_token = processor.tokenizer.eos_token
print("--- モデルの読み込みが完了しました ---")

print("\n--- LoRAアダプターの設定を開始します ---")
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad: trainable_params += param.numel()
    print(f"学習可能なパラメータ数: {trainable_params} || 全パラメータ数: {all_param} || 学習可能率: {100 * trainable_params / all_param:.2f}%")
print_trainable_parameters(model)
print("--- LoRAアダプターの設定が完了しました ---")

# --- 5. トレーニングと評価の実行 ---
print("\n--- トレーニングの準備を開始します ---")
train_dataset = LlavaFinetuneDataset(DATA_DIR, processor, split="train")
val_dataset = LlavaFinetuneDataset(DATA_DIR, processor, split="val")

print(f"訓練データ数: {len(train_dataset)} 件, 検証データ数: {len(val_dataset)} 件")

data_collator = get_data_collator(processor)

training_args = TrainingArguments(
    output_dir="./llava-1.5-7b-baseline-lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=1e-4,
    logging_steps=10,
    fp16=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # ▼▼▼ この一行を追加して、外部ツールへの報告を無効化します ▼▼▼
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./llava-1.5-7b-baseline-lora/final_model")
print("\n--- ファインチューニングと評価が完了しました！ ---")