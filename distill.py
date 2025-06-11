import os
import sys
from typing import List
import math
import torch.nn as nn
import torch.nn.functional as F
import fire
import torch
import transformers
from datasets import load_dataset # これはテキストデータセットロード用なのでLLaVAでは置き換える
import warnings
# DataLoaderとDistributedSamplerはそのまま使用
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import (
    get_constant_schedule_with_warmup, 
    get_polynomial_decay_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_inverse_sqrt_schedule
)
import numpy as np
import random

# LLaVAモデルのインポートを追加
from transformers import LlavaForConditionalGeneration, AutoProcessor 
from transformers import BitsAndBytesConfig, AutoConfig

# PEFT関連のインポートはLoRAを使わないのでコメントアウト、または必要に応じて削除
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
# from transformers import LlamaForCausalLM, LlamaTokenizer # これはLLaMA用なので削除またはコメントアウト

# 自分で作成したデータローダーとプロンプターをインポート
from utils.data_utils import RoadInspectionDataset, collate_fn # collate_fnもインポート
from utils.prompter import Prompter


def get_learning_rate_scheduler(lr_scheduler_name, optimizer, total_iters, warmup_steps=0):
    """Get learning rate scheduler."""
    if lr_scheduler_name == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif lr_scheduler_name == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_iters)
    elif lr_scheduler_name == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_iters)
    elif lr_scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_iters)
    elif lr_scheduler_name == "inverse_sqrt":
        return get_inverse_sqrt_schedule(optimizer, num_warmup_steps=warmup_steps)
    else:
        raise ValueError(f"lr_scheduler of type {lr_scheduler_name} is not supported.")


def train(
    # model/data params
    base_model: str = "",  # 生徒モデルのベースとなるLLaVAモデルのパス
    teacher_model: str = "",  # 教師モデルのLLaVAモデルのパス
    # データパスはJSONファイルのパスになります
    full_inst_desp_data_path: str = "", # instructionありのデータJSONパス
    no_inst_desp_data_path: str = "", # instructionなしのデータJSONパス
    valid_data_path: str = "", # validationデータJSONパス
    image_base_path: str = "", # 画像ファイルのルートパスを追加
    output_dir: str = "output_path",
    padding: str = None, # 使用しないが引数として残す
    # training hyperparams
    seed: int = 1234,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = None,  # 追加：勾配蓄積ステップ数（Noneの場合は batch_size // micro_batch_size を使用）
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 20,
    temperature: int = 1, # ロジット蒸留の温度
    distill_loss_type: str = 'KL', # can be chosen from [entropy, KL]
    distill_from_hidden_states: bool = True,  # 隠れ層蒸留をデフォルトでTrueに
    hidden_beta: float = 10.0, # 隠れ層損失の重み
    # lora hyperparams (LoRAは使用しないのでデフォルト値を設定し、コード内では適用しない)
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [], # LoRAモジュールを空リストに
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False, # データローダーで処理
    group_by_length: bool = False,  # faster, but produces an odd training loss curve (LLaVAでは非推奨)
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    teacher_resume_from_checkpoint: str = None, # 教師モデルのチェックポイント
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Distilling LLaVA model with params:\n" # ここをLLaVAに変更
            f"base_model: {base_model}\n"
            f"teacher_model: {teacher_model}\n"
            f"full_inst_desp_data_path: {full_inst_desp_data_path}\n"
            f"no_inst_desp_data_path: {no_inst_desp_data_path}\n"
            f"valid_data_path: {valid_data_path}\n"
            f"image_base_path: {image_base_path}\n" # 追加
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='llava-hf/llava-1.5-7b-hf'" # LLaVAモデルの例に修正
    assert (
        teacher_model
    ), "Please specify a --teacher_model, e.g. --teacher_model='llava-hf/llava-1.5-7b-hf'" # LLaVAモデルの例に修正
    # 勾配蓄積ステップ数の設定
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = batch_size // micro_batch_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    device = torch.cuda.current_device()

    # モデルとプロセッサ（トークナイザーと画像プロセッサ）のロード関数
    def load_llava_model_and_processor(model_path, is_teacher=False, local_resume_from_checkpoint=None):
        # LLaVAモデルとプロセッサをロード
        processor = AutoProcessor.from_pretrained(model_path)
        
        # 8bit量子化の設定
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        # モデルの設定を取得して更新
        config = AutoConfig.from_pretrained(model_path)
        config.use_cache = False  # キャッシュを無効化（勾配チェックポイントと互換性のため）
        config.output_hidden_states = True if distill_from_hidden_states else False  # 隠れ層が必要な場合のみ出力

        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            config=config,  # 更新した設定を使用
            quantization_config=quantization_config,  # 8bit量子化の設定を使用
            torch_dtype=torch.float16,  # 混合精度学習のためfloat16
            device_map=device_map,
            attn_implementation="flash_attention_2",  # Flash Attention 2を使用（新しい方法）
        )

        # メモリ効率化の設定
        model.gradient_checkpointing_enable()  # 勾配チェックポイントを有効化
        if torch.cuda.is_available():
            # PyTorchのメモリアロケータの設定
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)  # GPU メモリの95%まで使用
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if is_teacher:
            # 教師モデルは学習させないので勾配計算を無効化
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()  # 評価モードに設定
        else:
            model.train()  # 学習モードに設定

        return model, processor

    # 教師モデルと生徒モデルのロード
    # 注意: 生徒モデルは `base_model` から、教師モデルは `teacher_model` からロード
    # 今回はどちらも同じモデルを使うので、パスは同じになるはずです。
    student_model, student_processor = load_llava_model_and_processor(base_model, is_teacher=False, local_resume_from_checkpoint=resume_from_checkpoint)
    teacher_model, teacher_processor = load_llava_model_and_processor(teacher_model, is_teacher=True, local_resume_from_checkpoint=teacher_resume_from_checkpoint)

    # データローダーのインスタンス化 (collate_fnは外部で定義されているのでそのまま使用)
    # data_path, image_base_path, processor, prompter, instruction_type, cutoff_len, add_eos_token

    # 重要な変更点: collate_fnがprocessorを引数として受け取るように変更
    # または、collate_fnをクロージャとして定義し、processorをキャプチャする
    # ここでは、collate_fnを呼び出す際にprocessorを引数として渡せるようにする

    # collate_fnをラムダでラップして、processorを渡すように変更
    _student_collate_fn = lambda batch_data: collate_fn(batch_data, student_processor)
    _teacher_collate_fn = lambda batch_data: collate_fn(batch_data, teacher_processor)


    full_inst_desp_train_dataset = RoadInspectionDataset(
        data_path=full_inst_desp_data_path,
        image_base_path=image_base_path,
        processor=teacher_processor,
        prompter=prompter,
        instruction_type="full",
        cutoff_len=cutoff_len,
        add_eos_token=add_eos_token
    )
    no_inst_desp_train_dataset = RoadInspectionDataset(
        data_path=no_inst_desp_data_path,
        image_base_path=image_base_path,
        processor=student_processor, # 生徒モデルのプロセッサを使用
        prompter=prompter,
        instruction_type="no_instruction",
        cutoff_len=cutoff_len,
        add_eos_token=add_eos_token
    )

    # バリデーションデータセットも同様に作成
    no_inst_desp_val_dataset = None
    if valid_data_path:
        no_inst_desp_val_dataset = RoadInspectionDataset(
            data_path=valid_data_path,
            image_base_path=image_base_path,
            processor=student_processor, # 生徒モデルのプロセッサを使用
            prompter=prompter,
            instruction_type="no_instruction",
            cutoff_len=cutoff_len,
            add_eos_token=add_eos_token
        )

    # DataLoaderの準備
    # DistributedSamplerはDDP (Distributed Data Parallel) の場合のみ
    full_inst_desp_train_dataloader = DataLoader(
        full_inst_desp_train_dataset,
        batch_size=micro_batch_size, # micro_batch_sizeをDataLoaderのbatch_sizeとして使用
        shuffle=True if not ddp else False,
        sampler=DistributedSampler(full_inst_desp_train_dataset) if ddp else None,
        collate_fn=_teacher_collate_fn, # ここで教師モデル用のcollate_fnを渡す
        num_workers=os.cpu_count() // 2, # 適宜調整
        pin_memory=True,
    )
    no_inst_desp_train_dataloader = DataLoader(
        no_inst_desp_train_dataset,
        batch_size=micro_batch_size,
        shuffle=True if not ddp else False,
        sampler=DistributedSampler(no_inst_desp_train_dataset) if ddp else None,
        collate_fn=_student_collate_fn, # ここで生徒モデル用のcollate_fnを渡す
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )

    no_inst_desp_val_dataloader = None
    if no_inst_desp_val_dataset:
        no_inst_desp_val_dataloader = DataLoader(
            no_inst_desp_val_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            sampler=DistributedSampler(no_inst_desp_val_dataset) if ddp else None,
            collate_fn=_student_collate_fn, # ここで生徒モデル用のcollate_fnを渡す
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
        )

    # 最適化とスケジューラーの設定
    # LoRAを使わないので、モデル全体のパラメータを対象にする
    optimizer = AdamW(student_model.parameters(), lr=learning_rate, weight_decay=0)

    # DataLoaderのイテレータを準備
    full_inst_desp_train_iter = iter(full_inst_desp_train_dataloader)
    no_inst_desp_train_iter = iter(no_inst_desp_train_dataloader)

    # 学習ステップ数の計算
    num_steps_per_epoch = len(full_inst_desp_train_dataloader) # または no_inst_desp_train_dataloader
    total_training_steps = num_epochs * num_steps_per_epoch

    lr_scheduler = get_learning_rate_scheduler(lr_scheduler, optimizer, total_training_steps, warmup_steps)

    print("################ Start Distilling ##############")

    student_model.train() # 生徒モデルを学習モードに
    teacher_model.eval() # 教師モデルを評価モードに

    student_model.zero_grad() # 勾配をゼロに初期化

    total_loss_sum = 0 # 全体損失の合計
    update_loss_sum = 0 # 現在の勾配蓄積ステップでの損失合計
    hidden_mse_loss_sum = 0 # 隠れ層損失の合計

    best_val_loss = float('inf') # 最良のバリデーション損失を無限大で初期化

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
        for step_idx in range(num_steps_per_epoch): # 各エポック内のステップを反復
            try:
                # 教師モデルと生徒モデルのバッチを同時に取得
                full_desp_batch = next(full_inst_desp_train_iter)
                no_desp_batch = next(no_inst_desp_train_iter)
            except StopIteration:
                # エポックの終わりに達したらイテレータをリセット
                full_inst_desp_train_iter = iter(full_inst_desp_train_dataloader)
                no_inst_desp_train_iter = iter(no_inst_desp_train_dataloader)
                full_desp_batch = next(full_inst_desp_train_iter)
                no_desp_batch = next(no_inst_desp_train_dataloader)


            # データをGPUに移動
            full_desp_batch = {k: v.to(device) for k, v in full_desp_batch.items()}
            no_desp_batch = {k: v.to(device) for k, v in no_desp_batch.items()}

            # シーケンス長を確認し、短い方に合わせる
            teacher_seq_len = full_desp_batch["input_ids"].size(1)
            student_seq_len = no_desp_batch["input_ids"].size(1)
            min_seq_len = min(teacher_seq_len, student_seq_len)

            # 入力をトリミング
            full_desp_batch["input_ids"] = full_desp_batch["input_ids"][:, :min_seq_len]
            full_desp_batch["attention_mask"] = full_desp_batch["attention_mask"][:, :min_seq_len]
            full_desp_batch["labels"] = full_desp_batch["labels"][:, :min_seq_len]
            no_desp_batch["input_ids"] = no_desp_batch["input_ids"][:, :min_seq_len]
            no_desp_batch["attention_mask"] = no_desp_batch["attention_mask"][:, :min_seq_len]
            no_desp_batch["labels"] = no_desp_batch["labels"][:, :min_seq_len]

            # 生徒モデルのフォワードパス
            # output_hidden_states=True で隠れ層の出力を取得
            outputs_student = student_model(
                input_ids=no_desp_batch["input_ids"],
                attention_mask=no_desp_batch["attention_mask"],
                pixel_values=no_desp_batch["pixel_values"], # 画像入力を追加
                labels=no_desp_batch["labels"], # ロジット損失の計算にも使用される
                output_hidden_states=True if distill_from_hidden_states else False,
            )
            student_logits = outputs_student.logits # 生徒モデルのロジット

            # 教師モデルのフォワードパス (no_gradで勾配計算をしない)
            with torch.no_grad():
                outputs_teacher = teacher_model(
                    input_ids=full_desp_batch["input_ids"],
                    attention_mask=full_desp_batch["attention_mask"],
                    pixel_values=full_desp_batch["pixel_values"], # 画像入力を追加
                    labels=full_desp_batch["labels"], # ロジット損失の計算にも使用される
                    output_hidden_states=True if distill_from_hidden_states else False,
                )
                teacher_logits = outputs_teacher.logits # 教師モデルのロジット

            # ロジット蒸留損失の計算
            # 数値安定性のためにロジットをクリップ
            max_logit_value = 100.0  # 最大ロジット値
            student_logits = torch.clamp(student_logits, min=-max_logit_value, max=max_logit_value)
            teacher_logits = torch.clamp(teacher_logits, min=-max_logit_value, max=max_logit_value)

            # 温度でスケーリング
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

            # マスク処理
            logits_mask = (no_desp_batch["labels"] != -100)
            masked_student_log_probs = student_log_probs[logits_mask]
            masked_teacher_probs = teacher_probs[logits_mask]

            if masked_student_log_probs.numel() == 0:
                distill_loss = torch.tensor(1e-8, device=device, dtype=torch.float16, requires_grad=True)
            else:
                if distill_loss_type == 'KL':
                    # KL-divergenceの計算で数値安定性を確保
                    distill_loss = F.kl_div(
                        masked_student_log_probs,
                        masked_teacher_probs,
                        reduction='batchmean',
                        log_target=False
                    ) * (temperature ** 2)
                elif distill_loss_type == 'Entropy':
                    prod_probs = masked_teacher_probs * masked_student_log_probs
                    distill_loss = - torch.sum(prod_probs) / masked_student_log_probs.shape[0]
                    distill_loss = distill_loss * (temperature ** 2)
                else:
                    raise ValueError(f"distill_loss_type of type {distill_loss_type} is not supported.")

            # 損失値のチェックと処理
            if torch.isnan(distill_loss) or torch.isinf(distill_loss):
                print(f"Warning: Invalid loss detected. Using small loss for this step.")
                distill_loss = torch.tensor(1e-8, device=device, dtype=torch.float16, requires_grad=True)

            # 隠れ層蒸留損失の計算 (MSE Loss)
            # 論文の式(5)に対応
            hidden_state_loss = torch.tensor(0.0, device=device, dtype=torch.float16)
            if distill_from_hidden_states:
                # LLaVAモデルの隠れ層は、CLIPの隠れ層とLLaMAの隠れ層が異なる構造を持つ可能性がある。
                # 論文の隠れ層蒸留はLLM部分を対象としていると解釈。
                # LLaVAモデルの出力にある`hidden_states`はLLM部分の層のリストであると仮定。
                # `outputs_student.hidden_states` はタプルのリストで、各要素が層の出力テンソル。
                # 最初の要素は埋め込み層の出力、以降はTransformer層の出力。

                # 最終層を除く全ての隠れ層 (論文の `[: -1]`)
                # LLaVAの出力構造によるが、通常は`outputs_student.hidden_states`はLLMの各層の出力を含む
                student_hidden_states = outputs_student.hidden_states[1:] # 埋め込み層を除く
                teacher_hidden_states = outputs_teacher.hidden_states[1:] # 埋め込み層を除く

                # 各層の隠れ層の平均MSEを計算
                layer_losses = []
                for s_h, t_h in zip(student_hidden_states, teacher_hidden_states):
                    # 隠れ層の正規化 (論文の F.normalize)
                    s_h_norm = F.normalize(s_h, p=2, dim=-1)
                    t_h_norm = F.normalize(t_h, p=2, dim=-1)

                    # マスクを適用 (ロジット損失と同じマスクを隠れ層にも適用)
                    # `(batch_size, seq_len, hidden_size)` -> `(num_masked_tokens, hidden_size)`
                    masked_s_h_norm = s_h_norm[logits_mask]
                    masked_t_h_norm = t_h_norm[logits_mask]

                    if masked_s_h_norm.numel() > 0:
                        # MSE loss
                        layer_mse = F.mse_loss(masked_s_h_norm, masked_t_h_norm, reduction='mean')
                        layer_losses.append(layer_mse)
                    else:
                        layer_losses.append(torch.tensor(0.0, device=device, dtype=torch.float16))

                if layer_losses:
                    hidden_state_loss = torch.mean(torch.stack(layer_losses)) # 全ての層の平均MSE
                    hidden_state_loss *= hidden_beta # ベータで重み付け
                else:
                    hidden_state_loss = torch.tensor(0.0, device=device, dtype=torch.float16)

                hidden_mse_loss_sum += hidden_state_loss.item()


            # 総合損失
            total_distill_loss = distill_loss + hidden_state_loss

            # 損失のスケーリング（勾配蓄積のため）
            total_distill_loss = total_distill_loss / gradient_accumulation_steps

            total_distill_loss.backward()

            update_loss_sum += total_distill_loss.item() # このマイクロバッチでの損失を加算
            total_loss_sum += total_distill_loss.item() # 全体の損失に加算

            # 勾配の更新
            if (step_idx + 1) % gradient_accumulation_steps == 0 or (step_idx + 1) == num_steps_per_epoch:
                optimizer.step() # オプティマイザを更新
                lr_scheduler.step() # 学習率スケジューラーを更新
                student_model.zero_grad() # 勾配をゼロにリセット

                # ログの出力
                global_step = epoch * num_steps_per_epoch + (step_idx + 1)
                print(
                    f"Train | Epoch: {epoch + 1}/{num_epochs} | Step: {step_idx + 1}/{num_steps_per_epoch} | "
                    f"Global Step: {global_step}/{total_training_steps} | "
                    f"Loss: {update_loss_sum / gradient_accumulation_steps:.4f} | " # 勾配蓄積で割る
                    f"Hidden MSE Loss: {hidden_mse_loss_sum / gradient_accumulation_steps:.4f} | " # 勾配蓄積で割る
                    f"LR: {lr_scheduler.get_last_lr()[0]:.4e}"
                )
                update_loss_sum = 0
                hidden_mse_loss_sum = 0

        # 各エポックの終わりにバリデーションを実行
        if no_inst_desp_val_dataloader:
            student_model.eval() # 生徒モデルを評価モードに
            current_eval_loss = 0
            num_eval_steps = len(no_inst_desp_val_dataloader)
            with torch.no_grad():
                print(f"\n--- Evaluating Epoch {epoch + 1} ---")
                for eval_step_idx, val_batch in enumerate(no_inst_desp_val_dataloader):
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}

                    outputs_eval = student_model(
                        input_ids=val_batch["input_ids"],
                        attention_mask=val_batch["attention_mask"],
                        pixel_values=val_batch["pixel_values"],
                        labels=val_batch["labels"],
                    )
                    # ここでの損失は通常のCrossEntropyLoss (labelsに基づいて計算される)
                    current_eval_loss += outputs_eval.loss.item() # PyTorchの損失はバッチ平均なので、そのまま加算

                current_eval_loss /= num_eval_steps # 全体の平均損失

                if current_eval_loss < best_val_loss:
                    print(f"Validation loss improved from {best_val_loss:.4f} to {current_eval_loss:.4f}. Saving model...")
                    best_val_loss = current_eval_loss
                else:
                    print(f"Validation loss did not improve. Current: {current_eval_loss:.4f}, Best: {best_val_loss:.4f}")

                # エポック終了時に常にモデルを保存
                print(f"Saving model to {output_dir} regardless of validation loss...")
                student_model.save_pretrained(output_dir)

                print(f"Eval | Epoch: {epoch + 1} | Eval Loss: {current_eval_loss:.4f} | Best Eval Loss: {best_val_loss:.4f}")

            student_model.train() # 評価後、学習モードに戻す
            student_model.zero_grad() # 勾配をゼロにリセット


    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)