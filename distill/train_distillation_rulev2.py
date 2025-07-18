import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from functools import partial

from transformers import (
    AutoProcessor,
    AutoConfig,
    LlavaNextForConditionalGeneration,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from PIL import Image
import os
import csv
from tqdm import tqdm
import time
import gc

# --- 1. 設定項目 ---
TEACHER_MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
STUDENT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TEACHER_ADAPTER_PATH = "checkpoints/llava-v1.6-vicuna-7b-lora-okvqa/checkpoint-2252"
STUDENT_ADAPTER_PATH = None
TRAIN_CSV_PATH = "data/VisA/chewinggum/full7/train.csv"
EVAL_CSV_PATH = "data/VisA/chewinggum/no/val.csv"
OUTPUT_DIR = "output/distilled_student_model_v15_final/"

# --- 2. ハイパーパラメータ (推奨設定) ---
NUM_EPOCHS = 20
BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4
TEMPERATURE = 1.0
LOSS_WEIGHT_LOGITS = 1.0
LOSS_WEIGHT_HIDDEN = 10.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
WARMUP_STEPS = 20
# -------------------------------------------------------------

def load_models_and_processors():
    # (変更なし)
    print("1. 教師・生徒のプロセッサをそれぞれロード中...")
    teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID)
    student_processor = AutoProcessor.from_pretrained(STUDENT_MODEL_ID)
    print("   >>> プロセッサのロード完了！")
    print("2. 8-bit量子化設定を準備中...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    print("   >>> 量子化設定の準備完了！")
    print(f"3. 教師モデルのベース '{TEACHER_MODEL_ID}' (8-bit) をロード中...")
    teacher_config = AutoConfig.from_pretrained(TEACHER_MODEL_ID)
    teacher_config.output_hidden_states = True
    try:
        image_grid_pinpoints = teacher_processor.image_processor.image_grid_pinpoints
        teacher_config.image_grid_pinpoints = image_grid_pinpoints
    except AttributeError: pass
    teacher_model = LlavaNextForConditionalGeneration.from_pretrained(TEACHER_MODEL_ID, config=teacher_config, quantization_config=quantization_config, torch_dtype=torch.float16)
    if TEACHER_ADAPTER_PATH:
        teacher_model = PeftModel.from_pretrained(teacher_model, TEACHER_ADAPTER_PATH)
        teacher_model = teacher_model.merge_and_unload()
    teacher_model.eval()
    print("   >>> 教師モデルのロード完了！")
    print(f"4. 生徒モデルのベース '{STUDENT_MODEL_ID}' (8-bit LoRA) をロード中...")
    student_config = AutoConfig.from_pretrained(STUDENT_MODEL_ID)
    student_config.output_hidden_states = True
    student_model = LlavaForConditionalGeneration.from_pretrained(STUDENT_MODEL_ID, config=student_config, quantization_config=quantization_config, torch_dtype=torch.float16)
    if STUDENT_ADAPTER_PATH:
        student_model = PeftModel.from_pretrained(student_model, STUDENT_ADAPTER_PATH)
        student_model = student_model.merge_and_unload()
    print("5. 生徒モデルに蒸留学習用のLoRAを適用中...")
    student_model = prepare_model_for_kbit_training(student_model)
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()
    print("   >>> 生徒モデルのロードとLoRAの適用完了！\n")
    return teacher_model, student_model, teacher_processor, student_processor

### ★ 変更点 ★ ###
# Datasetの役割を、プロンプト長を計算して返すように変更
class StaticDistillationDataset(Dataset):
    def __init__(self, csv_path, teacher_processor, student_processor, is_train=True):
        self.teacher_processor = teacher_processor
        self.student_processor = student_processor
        self.is_train = is_train
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                self.tasks = list(csv.DictReader(f))
            print(f"データセット: {csv_path} から {len(self.tasks)} 件のタスクをロードしました。")
        except (FileNotFoundError, KeyError) as e:
            print(f"エラー: データファイル '{csv_path}' の読み込みに失敗しました。'{e}'")
            self.tasks = []

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if not self.tasks: return None
        task = self.tasks[idx]
        image_path, instruction, output = task['image_path'], task['instruction'], task['output']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"警告: 画像 '{image_path}' が見つかりません。タスク {idx} をスキップします。")
            return None

        # 教師用の完全なプロンプトと、プロンプトのみの部分を作成
        teacher_full_prompt = f"USER: <image>\n{instruction}\nASSISTANT: {output}"
        teacher_prompt_only = f"USER: <image>\n{instruction}\nASSISTANT:"
        
        # 生徒用の完全なプロンプトと、プロンプトのみの部分を作成
        student_instruction = " " if self.is_train else instruction
        student_full_prompt = f"USER: <image>\n{student_instruction}\nASSISTANT: {output}"
        student_prompt_only = f"USER: <image>\n{student_instruction}\nASSISTANT:"

        # トークン化して、モデルへの入力とプロンプト長を取得
        teacher_inputs = self.teacher_processor(text=teacher_full_prompt, images=image, return_tensors="pt")
        teacher_prompt_len = self.teacher_processor(text=teacher_prompt_only, images=image, return_tensors="pt").input_ids.shape[1]

        student_inputs = self.student_processor(text=student_full_prompt, images=image, return_tensors="pt")
        student_prompt_len = self.student_processor(text=student_prompt_only, images=image, return_tensors="pt").input_ids.shape[1]

        image_sizes = teacher_inputs.get("image_sizes", torch.tensor([image.size]).to(torch.int64))

        return {
            "teacher_pixel_values": teacher_inputs.pixel_values.squeeze(0),
            "teacher_input_ids": teacher_inputs.input_ids.squeeze(0),
            "student_pixel_values": student_inputs.pixel_values.squeeze(0),
            "student_input_ids": student_inputs.input_ids.squeeze(0),
            "image_sizes": image_sizes.squeeze(0),
            "teacher_prompt_len": teacher_prompt_len,
            "student_prompt_len": student_prompt_len,
        }


def custom_collate_fn(batch, processor):
    """異なる長さのシーケンスをパディングしてバッチ化する"""
    batch = [item for item in batch if item is not None]
    if not batch: return {}
    
    from torch.nn.utils.rnn import pad_sequence
    pad_token_id = processor.tokenizer.pad_token_id
    
    padded_batch = {}
    # 全てのキーをループ
    keys = batch[0].keys()
    for key in keys:
        items = [d[key] for d in batch]
        if "input_ids" in key:
            # 左側にパディング
            padded_batch[key] = pad_sequence(items, batch_first=True, padding_value=pad_token_id)
        elif "pixel_values" in key or "image_sizes" in key:
             padded_batch[key] = torch.stack(items)
        else: # prompt_len
             padded_batch[key] = torch.tensor(items)

    # attention_maskを生成
    padded_batch["teacher_attention_mask"] = (padded_batch["teacher_input_ids"] != pad_token_id).long()
    padded_batch["student_attention_mask"] = (padded_batch["student_input_ids"] != pad_token_id).long()
    
    return padded_batch

### ★ 変更点 ★ ###
# 損失計算関数をスライシングベースに全面的に書き換え
def calculate_distillation_loss(student_outputs, teacher_outputs, student_prompt_len, teacher_prompt_len):
    
    # 応答部分のlogitsをスライス
    # バッチ内の各サンプルでプロンプト長が違う可能性があるため、ループで処理（ただしバッチサイズ1なら不要）
    batch_size = student_outputs.logits.shape[0]
    loss_logits_list, loss_hidden_list = [], []

    for i in range(batch_size):
        s_prompt_len = student_prompt_len[i]
        t_prompt_len = teacher_prompt_len[i]

        s_logits_answer = student_outputs.logits[i, s_prompt_len:, :]
        t_logits_answer = teacher_outputs.logits[i, t_prompt_len:, :]
        
        answer_len = min(s_logits_answer.shape[0], t_logits_answer.shape[0])
        if answer_len == 0: continue

        s_logits_answer = s_logits_answer[:answer_len]
        t_logits_answer = t_logits_answer[:answer_len]

        # Logits Loss
        teacher_probs = F.softmax(t_logits_answer.float() / TEMPERATURE, dim=-1)
        student_log_probs = F.log_softmax(s_logits_answer.float() / TEMPERATURE, dim=-1)
        loss_logits_list.append(F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (TEMPERATURE ** 2))
        
        # Hidden Loss
        if student_outputs.hidden_states is not None and teacher_outputs.hidden_states is not None:
            num_layers = min(len(student_outputs.hidden_states[1:-1]), len(teacher_outputs.hidden_states[1:-1]))
            hidden_loss_per_item = 0
            for layer_idx in range(num_layers):
                s_hidden_answer = student_outputs.hidden_states[layer_idx+1][i, s_prompt_len:, :]
                t_hidden_answer = teacher_outputs.hidden_states[layer_idx+1][i, t_prompt_len:, :]
                s_hidden_answer = s_hidden_answer[:answer_len]
                t_hidden_answer = t_hidden_answer[:answer_len]

                s_hidden_norm = F.normalize(s_hidden_answer.float(), p=2, dim=-1)
                t_hidden_norm = F.normalize(t_hidden_answer.float(), p=2, dim=-1)
                hidden_loss_per_item += F.mse_loss(s_hidden_norm, t_hidden_norm)
            
            if num_layers > 0:
                loss_hidden_list.append(hidden_loss_per_item / num_layers)

    # バッチ全体の平均損失を計算
    final_loss_logits = torch.stack(loss_logits_list).mean() if loss_logits_list else torch.tensor(0.0, device=student_outputs.logits.device)
    final_loss_hidden = torch.stack(loss_hidden_list).mean() if loss_hidden_list else torch.tensor(0.0, device=student_outputs.logits.device)

    return final_loss_logits, final_loss_hidden


def evaluate(student_model, teacher_model, dataloader, device):
    """
    モデルの評価を行う関数。
    メモリ不足（OOM）エラーを回避するため、評価時の隠れ状態損失の計算はスキップする。
    """
    student_model.eval()
    teacher_model.eval()
    total_eval_loss = 0
    total_logits_loss = 0
    
    # 評価時は勾配計算が不要なので、torch.no_grad()で全体を囲む
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if not batch: continue
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                # --- 教師モデルの処理 (隠れ状態は不要) ---
                teacher_inputs = {
                    "pixel_values": batch["teacher_pixel_values"], 
                    "input_ids": batch["teacher_input_ids"], 
                    "attention_mask": batch["teacher_attention_mask"], 
                    "image_sizes": batch["image_sizes"],
                    "output_hidden_states": False 
                }
                teacher_outputs = teacher_model(**teacher_inputs)
                
                # --- 生徒モデルの処理 (隠れ状態は不要) ---
                student_inputs = {
                    "pixel_values": batch["student_pixel_values"], 
                    "input_ids": batch["student_input_ids"], 
                    "attention_mask": batch["student_attention_mask"],
                    "output_hidden_states": False
                }
                student_outputs = student_model(**student_inputs)

            # 損失計算はautocastの外で
            with autocast(device_type="cuda", enabled=False):
                # ★ 変更点: 損失計算を呼び出すのではなく、ここで直接計算
                student_prompt_len = batch["student_prompt_len"]
                teacher_prompt_len = batch["teacher_prompt_len"]
                batch_size = student_outputs.logits.shape[0]
                
                for i in range(batch_size):
                    s_prompt_len = student_prompt_len[i]
                    t_prompt_len = teacher_prompt_len[i]
                    s_logits_answer = student_outputs.logits[i, s_prompt_len:, :]
                    t_logits_answer = teacher_outputs.logits[i, t_prompt_len:, :]
                    answer_len = min(s_logits_answer.shape[0], t_logits_answer.shape[0])
                    if answer_len == 0: continue
                    s_logits_answer = s_logits_answer[:answer_len]
                    t_logits_answer = t_logits_answer[:answer_len]
                    teacher_probs = F.softmax(t_logits_answer.float() / TEMPERATURE, dim=-1)
                    student_log_probs = F.log_softmax(s_logits_answer.float() / TEMPERATURE, dim=-1)
                    
                    loss_logits = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (TEMPERATURE ** 2)
                    total_logits_loss += loss_logits.item()

    avg_logits_loss = total_logits_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    print(f"✅ Eval Logits Loss: {avg_logits_loss:.4f}")
    return avg_logits_loss

def main():
    print("--- マルチモーダル・ルール蒸留 (v15 評価ループ修正) ---")
    teacher_model, student_model, teacher_processor, student_processor = load_models_and_processors()
    device = student_model.device
    
    print("\n--- 💾 データセットをロード中... ---")
    train_dataset = StaticDistillationDataset(csv_path=TRAIN_CSV_PATH, teacher_processor=teacher_processor, student_processor=student_processor, is_train=True)
    eval_dataset = StaticDistillationDataset(csv_path=EVAL_CSV_PATH, teacher_processor=teacher_processor, student_processor=student_processor, is_train=False)
    
    collate_fn_wrapped = partial(custom_collate_fn, processor=student_processor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_wrapped)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_wrapped)

    if not train_dataset.tasks:
        print("エラー: 学習データがロードできませんでした。処理を終了します。")
        return

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    total_training_steps = (len(train_dataloader) // GRAD_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_training_steps)
    scaler = GradScaler(device='cuda')

    print("\n--- 🚀 学習を開始します！ ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        student_model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            if not batch: continue
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            with autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    teacher_inputs = {"pixel_values": batch["teacher_pixel_values"], "input_ids": batch["teacher_input_ids"], "attention_mask": batch["teacher_attention_mask"], "image_sizes": batch["image_sizes"]}
                    teacher_outputs = teacher_model(**teacher_inputs)
                
                student_inputs = {"pixel_values": batch["student_pixel_values"], "input_ids": batch["student_input_ids"], "attention_mask": batch["student_attention_mask"]}
                student_outputs = student_model(**student_inputs)
            
            with autocast(device_type="cuda", enabled=False):
                loss_logits, loss_hidden = calculate_distillation_loss(
                    student_outputs, teacher_outputs, batch["student_prompt_len"], batch["teacher_prompt_len"]
                )
                total_loss = (LOSS_WEIGHT_LOGITS * loss_logits + LOSS_WEIGHT_HIDDEN * loss_hidden).float()
            
            total_train_loss += total_loss.item()
            scaled_loss = scaler.scale(total_loss) / GRAD_ACCUMULATION_STEPS
            scaled_loss.backward()

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix({"Loss": total_loss.item(), "Logits": loss_logits.item(), "Hidden": loss_hidden.item() if isinstance(loss_hidden, torch.Tensor) else loss_hidden})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"--- Epoch {epoch + 1} 訓練完了 ---\nAverage Training Loss: {avg_train_loss:.4f}")
        
        # ★★★★★★★★★★★★★★★★★★ ここからが復活した部分 ★★★★★★★★★★★★★★★★★★
        print("🧹 メモリをクリーンアップしています...")
        if 'batch' in locals(): del batch
        if 'total_loss' in locals(): del total_loss
        if 'scaled_loss' in locals(): del scaled_loss
        if 'student_outputs' in locals(): del student_outputs
        if 'teacher_outputs' in locals(): del teacher_outputs
        if 'loss_logits' in locals(): del loss_logits
        if 'loss_hidden' in locals(): del loss_hidden
        gc.collect()
        torch.cuda.empty_cache()
        print("✨ メモリのクリーンアップ完了！")

        if eval_dataloader and eval_dataset.tasks:
            print(f"--- Epoch {epoch + 1} 評価開始 ---")
            avg_eval_loss = evaluate(student_model, teacher_model, eval_dataloader, device)
            print(f"--- Epoch {epoch + 1} 評価完了 ---\n✅ Average Evaluation Loss: {avg_eval_loss:.4f}")

            epoch_output_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            print(f"モデルを {epoch_output_dir} に保存中...")
            student_model.save_pretrained(epoch_output_dir)
        else:
            print("評価データが見つからないため、評価をスキップします。")
        # ★★★★★★★★★★★★★★★★★★★★ ここまでが復活した部分 ★★★★★★★★★★★★★★★★★★

    print("\n🎉 全ての学習が完了しました！")
    
if __name__ == '__main__':
    main()