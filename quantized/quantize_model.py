import os
import torch
from transformers import LlavaNextForConditionalGeneration, AutoTokenizer, LlavaNextProcessor
from peft import PeftModel
from awq import AutoAWQForCausalLM
from datasets import load_dataset  # 動作したコードから採用

def merge_lora(base_model_id, lora_weights_path, merged_model_path):
    """
    ベースモデルにLoRAの重みをマージして保存する関数
    """
    print("--- LoRAのマージを開始します ---")
    # マージ処理はCPUで行うことで、後続の量子化でGPUメモリを温存します
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_path)
    print(f"LoRA統合済みモデルを {merged_model_path} に保存しました。")
    return merged_model_path

def quantize_awq(merged_model_path, quantized_model_path):
    """
    指定されたモデルをAWQで量子化する関数
    """
    print("--- AWQ量子化を開始します ---")
    
    # 量子化設定
    quant_config = {
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
        "version": "GEMM"
    }

    # 【成功の鍵①】動作したコードを参考に、高品質な短いキャリブレーションデータを準備
    print("高品質なキャリブレーションデータを準備します...")
    calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # 10文字以上512文字未満のサンプルを32個使用
    calib_data = [d["text"] for d in calib_dataset if len(d["text"]) > 10 and len(d["text"]) < 512][:32]
    print(f"{len(calib_data)}個のサンプルデータを準備しました。")

    # 【成功の鍵②】動作したコードを参考に、効率的なモデル読み込みオプションを使用
    print(f"マージ済みモデル '{merged_model_path}' を読み込みます...")
    model = AutoAWQForCausalLM.from_pretrained(
        merged_model_path,
        trust_remote_code=True,
        safetensors=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"  # GPUに自動で配置
    )

    tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=True)

    # 手動で準備した「短い」データを使用して量子化
    print("\nモデルの量子化を実行します。この処理には時間がかかります...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data
    )

    # 量子化されたモデルを保存
    print(f"\n量子化されたモデルを '{quantized_model_path}' に保存します...")
    model.save_quantized(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)
    
    print(f"\n🎉 最終的な量子化が完了しました！'{quant_path}' にモデルが保存されています。")

def copy_tokenizer_and_processor(base_model_id, save_dir):
    """
    ベースモデルからトークナイザーとプロセッサーの設定をコピーする関数
    """
    print("--- トークナイザーとプロセッサーをコピーします ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.save_pretrained(save_dir)
    except Exception as e:
        print(f"Tokenizerのコピーに失敗: {e}")

    try:
        processor = LlavaNextProcessor.from_pretrained(base_model_id)
        processor.save_pretrained(save_dir)
    except Exception as e:
        print(f"Processorのコピーに失敗: {e}")

if __name__ == "__main__":
    # --- 設定項目 ---
    base_model_id = "llava-v1.6-vicuna-7b-hf"
    # ご自身のLoRAチェックポイントのパスを指定してください
    lora_weights_path = "checkpoints/llava-v1.6-vicuna-7b-lora-okvqa/checkpoint-2252"
    merged_model_path = "merged_model"
    quantized_model_path = "quantized_model_vllm" # 保存先名を変更
    
    # --- 実行プロセス ---
    # 1. LoRA層をマージ
    merge_lora(base_model_id, lora_weights_path, merged_model_path)
    
    # 2. トークナイザーとプロセッサーの設定をマージ済みモデルのディレクトリにコピー
    copy_tokenizer_and_processor(base_model_id, merged_model_path)
    
    # 3. AWQで量子化
    quantize_awq(merged_model_path, quantized_model_path)