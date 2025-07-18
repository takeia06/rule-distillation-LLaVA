import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from transformers import (
    AutoProcessor,
    AutoConfig,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from PIL import Image
import os
import csv
from tqdm import tqdm
import time
import gc ### 変更点: ガベージコレクションをインポート ###

# --- 1. 設定項目 ---
# (変更なし)
# --- モデル設定 ---
TEACHER_MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
STUDENT_MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"

# --- データ設定 ---
TRAIN_CSV_PATH = "data/VisA/chewinggum/full/train.csv"
EVAL_CSV_PATH = "data/VisA/chewinggum/no/val.csv"

# --- 学習パラメータ ---
OUTPUT_DIR = "student_model_v1.6_output/"
NUM_EPOCHS = 20
BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
MAX_NEW_TOKENS_TEACHER = 512

# --- 損失の重み設定 ---
LOSS_WEIGHT_RESPONSE = 1.0
LOSS_WEIGHT_LOGITS = 0.5
LOSS_WEIGHT_HIDDEN = 0.2

# --- LoRA設定 ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
# -------------------------------------------------------------

# (load_models_and_processors, OnlineDistillationDataset, custom_collate_fn, evaluate 関数は変更なし)
def load_models_and_processors():
    """教師・生徒それぞれのモデルとプロセッサをロードします。"""
    print("1. 教師・生徒のプロセッサをそれぞれロード中...")
    teacher_processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID)
    student_processor = AutoProcessor.from_pretrained(STUDENT_MODEL_ID)
    print("   >>> プロセッサのロード完了！")

    print("2. モデルの設計図(config)を準備中...")
    model_config = AutoConfig.from_pretrained(TEACHER_MODEL_ID)
    try:
        image_grid_pinpoints = teacher_processor.image_processor.image_grid_pinpoints
        model_config.image_grid_pinpoints = image_grid_pinpoints
        print(f"   >>> 取得したグリッド設定: {image_grid_pinpoints}")
    except AttributeError:
        print("   >>> 警告: プロセッサから image_grid_pinpoints を取得できませんでした。デフォルト設定を使用します。")
    print("   >>> 設計図(config)の準備完了！")
    
    print("3. 最新の8-bit量子化設定を準備中...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    print("   >>> 量子化設定の準備完了！")

    print(f"4. 教師モデル '{TEACHER_MODEL_ID}' (8-bit) をロード中...")
    teacher_model = LlavaNextForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_ID,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    ).eval()
    print("   >>> 教師モデルのロード完了！")

    print(f"5. 生徒モデル '{STUDENT_MODEL_ID}' (8-bit LoRA) をロード中...")
    student_model = LlavaNextForConditionalGeneration.from_pretrained(
        STUDENT_MODEL_ID,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    student_model = prepare_model_for_kbit_training(student_model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()
    print("   >>> 生徒モデルのロードとLoRAの適用完了！\n")

    return teacher_model, student_model, teacher_processor, student_processor

class OnlineDistillationDataset(Dataset):
    """オンライン蒸留用のデータセット"""
    def __init__(self, csv_path, teacher_processor, student_processor):
        self.teacher_processor = teacher_processor
        self.student_processor = student_processor
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                self.tasks = list(csv.DictReader(f))
            print(f"データセット: {csv_path} から {len(self.tasks)} 件のタスクをロードしました。")
        except FileNotFoundError:
            print(f"エラー: データファイル '{csv_path}' が見つかりません。")
            self.tasks = []

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        if not self.tasks: return None
        task = self.tasks[idx]
        image_path = task['image_path']
        instruction = task['instruction']

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"警告: 画像 '{image_path}' が見つかりません。タスク {idx} をスキップします。")
            return None

        vicuna_system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        
        teacher_prompt = (
            f"{vicuna_system_prompt} "
            f"USER: <image>\n{instruction}\n"
            "ASSISTANT:"
        )

        student_prompt = (
            f"{vicuna_system_prompt} "
            f"USER: <image>\n"
            "ASSISTANT:"
        )
        
        teacher_inputs = self.teacher_processor(
            text=teacher_prompt, images=image, return_tensors="pt"
        )
        student_inputs = self.student_processor(
            text=student_prompt, images=image, return_tensors="pt"
        )

        return {
            "teacher_pixel_values": teacher_inputs.pixel_values.squeeze(0),
            "student_pixel_values": student_inputs.pixel_values.squeeze(0),
            "image_sizes": teacher_inputs.get("image_sizes", [image.size]).squeeze(0),
            "teacher_input_ids": teacher_inputs.input_ids.squeeze(0),
            "teacher_attention_mask": teacher_inputs.attention_mask.squeeze(0),
            "student_input_ids": student_inputs.input_ids.squeeze(0),
            "student_attention_mask": student_inputs.attention_mask.squeeze(0),
        }

def custom_collate_fn(batch):
    """Noneをフィルタリングし、データを適切にバッチ化するカスタムcollate関数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def evaluate(student_model, teacher_model, dataloader, device):
    """モデルの評価を行う関数"""
    student_model.eval()
    teacher_model.eval()
    total_eval_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if not batch: continue

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                # --- ステップA: 教師モデルでお手本を生成 ---
                teacher_inputs = {
                    "input_ids": batch["teacher_input_ids"],
                    "pixel_values": batch["teacher_pixel_values"],
                    "attention_mask": batch["teacher_attention_mask"],
                }
                if "image_sizes" in batch:
                    teacher_inputs["image_sizes"] = batch["image_sizes"]
                
                generated_ids = teacher_model.generate(
                    **teacher_inputs, max_new_tokens=MAX_NEW_TOKENS_TEACHER, do_sample=False
                )
                prompt_len = teacher_inputs["input_ids"].shape[1]
                teacher_answer_ids = generated_ids[:, prompt_len:]

                full_teacher_input_ids = torch.cat([teacher_inputs["input_ids"], teacher_answer_ids], dim=1)
                full_teacher_attention_mask = torch.ones_like(full_teacher_input_ids).to(device)

                forward_teacher_inputs = {
                    "input_ids": full_teacher_input_ids,
                    "attention_mask": full_teacher_attention_mask,
                    "pixel_values": batch["teacher_pixel_values"],
                    "output_hidden_states": True,
                }
                if "image_sizes" in batch:
                    forward_teacher_inputs["image_sizes"] = batch["image_sizes"]
                
                teacher_forward_outputs = teacher_model(**forward_teacher_inputs)
                
                # --- ステップB: 生徒モデルのフォワードパス ---
                student_full_input_ids = torch.cat([batch["student_input_ids"], teacher_answer_ids], dim=1)
                student_full_attention_mask = torch.cat([batch["student_attention_mask"], torch.ones_like(teacher_answer_ids)], dim=1)
                labels = student_full_input_ids.clone()
                student_prompt_len = batch["student_input_ids"].shape[1]
                labels[:, :student_prompt_len] = -100

                student_inputs = {
                    "pixel_values": batch["student_pixel_values"],
                    "input_ids": student_full_input_ids,
                    "attention_mask": student_full_attention_mask,
                    "labels": labels,
                    "output_hidden_states": True,
                }
                if "image_sizes" in batch:
                    student_inputs["image_sizes"] = batch["image_sizes"]
                
                student_outputs = student_model(**student_inputs)

                # --- ステップC: 損失計算 ---
                loss_response = student_outputs.loss
                
                with autocast(device_type="cuda", enabled=False):
                    teacher_answer_start = prompt_len
                    student_answer_start = student_prompt_len
                    
                    teacher_logits_answer = teacher_forward_outputs.logits[:, teacher_answer_start:, :].float()
                    teacher_hidden_answer = teacher_forward_outputs.hidden_states[-1][:, teacher_answer_start:, :].float()
                    
                    student_logits_answer = student_outputs.logits[:, student_answer_start:, :].float()
                    student_hidden_answer = student_outputs.hidden_states[-1][:, student_answer_start:, :].float()
                    
                    seq_len = min(student_logits_answer.shape[1], teacher_logits_answer.shape[1])
                    
                    loss_logits = F.kl_div(
                        F.log_softmax(student_logits_answer[:, :seq_len, :], dim=-1),
                        F.softmax(teacher_logits_answer[:, :seq_len, :], dim=-1),
                        reduction='batchmean', log_target=False
                    )
                    loss_hidden = F.mse_loss(
                        student_hidden_answer[:, :seq_len, :], teacher_hidden_answer[:, :seq_len, :]
                    )
                    
                total_loss = (
                    LOSS_WEIGHT_RESPONSE * loss_response.float() +
                    LOSS_WEIGHT_LOGITS * loss_logits +
                    LOSS_WEIGHT_HIDDEN * loss_hidden
                )
                total_eval_loss += total_loss.item()

    avg_eval_loss = total_eval_loss / len(dataloader)
    return avg_eval_loss

def main():
    print("--- マルチモーダル蒸留 v2.3 (メモリ解放対応) ---")

    teacher_model, student_model, teacher_processor, student_processor = load_models_and_processors()
    device = student_model.device

    print("\n--- 💾 データセットをロード中... ---")
    train_dataset = OnlineDistillationDataset(
        csv_path=TRAIN_CSV_PATH,
        teacher_processor=teacher_processor,
        student_processor=student_processor
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    eval_dataset = OnlineDistillationDataset(
        csv_path=EVAL_CSV_PATH,
        teacher_processor=teacher_processor,
        student_processor=student_processor
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    if not train_dataset.tasks:
        print("エラー: 学習データがロードできませんでした。処理を終了します。")
        return

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
    total_training_steps = (len(train_dataloader) // GRAD_ACCUMULATION_STEPS) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_training_steps
    )
    scaler = GradScaler()

    print("\n--- 🚀 学習を開始します！ ---")
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        student_model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            if not batch: continue

            # (訓練ループの内部は変更なし)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    teacher_inputs = {
                        "input_ids": batch["teacher_input_ids"],
                        "pixel_values": batch["teacher_pixel_values"],
                        "attention_mask": batch["teacher_attention_mask"],
                    }
                    if "image_sizes" in batch:
                        teacher_inputs["image_sizes"] = batch["image_sizes"]
                    
                    generated_ids = teacher_model.generate(
                        **teacher_inputs, max_new_tokens=MAX_NEW_TOKENS_TEACHER, do_sample=False
                    )
                    prompt_len = teacher_inputs["input_ids"].shape[1]
                    teacher_answer_ids = generated_ids[:, prompt_len:]

                    full_teacher_input_ids = torch.cat([teacher_inputs["input_ids"], teacher_answer_ids], dim=1)
                    full_teacher_attention_mask = torch.ones_like(full_teacher_input_ids).to(device)

                    forward_teacher_inputs = {
                        "input_ids": full_teacher_input_ids,
                        "attention_mask": full_teacher_attention_mask,
                        "pixel_values": batch["teacher_pixel_values"],
                        "output_hidden_states": True,
                    }
                    if "image_sizes" in batch:
                        forward_teacher_inputs["image_sizes"] = batch["image_sizes"]
                    
                    teacher_forward_outputs = teacher_model(**forward_teacher_inputs)
                
                student_full_input_ids = torch.cat([batch["student_input_ids"], teacher_answer_ids], dim=1)
                student_full_attention_mask = torch.cat([batch["student_attention_mask"], torch.ones_like(teacher_answer_ids)], dim=1)
                labels = student_full_input_ids.clone()
                student_prompt_len = batch["student_input_ids"].shape[1]
                labels[:, :student_prompt_len] = -100

                student_inputs = {
                    "pixel_values": batch["student_pixel_values"],
                    "input_ids": student_full_input_ids,
                    "attention_mask": student_full_attention_mask,
                    "labels": labels,
                    "output_hidden_states": True,
                }
                if "image_sizes" in batch:
                    student_inputs["image_sizes"] = batch["image_sizes"]

                student_outputs = student_model(**student_inputs)

                loss_response = student_outputs.loss

                with autocast(device_type="cuda", enabled=False):
                    teacher_answer_start = prompt_len
                    student_answer_start = student_prompt_len
                    teacher_logits_answer = teacher_forward_outputs.logits[:, teacher_answer_start:, :].float()
                    teacher_hidden_answer = teacher_forward_outputs.hidden_states[-1][:, teacher_answer_start:, :].float()
                    student_logits_answer = student_outputs.logits[:, student_answer_start:, :].float()
                    student_hidden_answer = student_outputs.hidden_states[-1][:, student_answer_start:, :].float()
                    seq_len = min(student_logits_answer.shape[1], teacher_logits_answer.shape[1])
                    
                    loss_logits = F.kl_div(
                        F.log_softmax(student_logits_answer[:, :seq_len, :], dim=-1),
                        F.softmax(teacher_logits_answer[:, :seq_len, :], dim=-1),
                        reduction='batchmean', log_target=False
                    )
                    loss_hidden = F.mse_loss(
                        student_hidden_answer[:, :seq_len, :], teacher_hidden_answer[:, :seq_len, :]
                    )
                
                total_loss = (
                    LOSS_WEIGHT_RESPONSE * loss_response.float() +
                    LOSS_WEIGHT_LOGITS * loss_logits +
                    LOSS_WEIGHT_HIDDEN * loss_hidden
                )
            
            total_train_loss += total_loss.item()
            scaled_loss = scaler.scale(total_loss) / GRAD_ACCUMULATION_STEPS
            scaled_loss.backward()

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % 10 == 0:
                    print(f"Epoch: {epoch + 1}, Step: {global_step}, Train Loss: {total_loss.item():.4f}")
                global_step += 1
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"--- Epoch {epoch + 1} 訓練完了 ---")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        ### 変更点: ここから ###
        # 評価フェーズの前に、不要になった変数を削除し、CUDAキャッシュをクリアする
        print("🧹 メモリをクリーンアップしています...")
        del batch, total_loss, scaled_loss, student_outputs, teacher_forward_outputs
        gc.collect()
        torch.cuda.empty_cache()
        print("✨ メモリのクリーンアップ完了！")
        ### 変更点: ここまで ###

        if eval_dataloader and eval_dataset.tasks:
            print(f"--- Epoch {epoch + 1} 評価開始 ---")
            avg_eval_loss = evaluate(student_model, teacher_model, eval_dataloader, device)
            print(f"--- Epoch {epoch + 1} 評価完了 ---")
            print(f"✅ Average Evaluation Loss: {avg_eval_loss:.4f}")
        else:
            print("評価データが見つからないため、評価をスキップします。")

        epoch_output_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        print(f"モデルを {epoch_output_dir} に保存中...")
        student_model.save_pretrained(epoch_output_dir)

    print("\n🎉 全ての学習が完了しました！")


if __name__ == '__main__':
    main()