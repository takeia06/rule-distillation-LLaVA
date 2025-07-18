import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset  # <-- データセットライブラリを追加

# モデルと保存先のパスa
model_path = 'llava-hf/llava-1.5-7b-hf'
quant_path = 'llava-v1.5-7b-awq-vllm'

# 量子化の設定
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM"
}

# --- 【最重要】ここが短いサンプルデータを作成する部分です ---
print("メモリ節約のため、短いキャリブレーションデータを手動で準備します...")
calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_data = [d["text"] for d in calib_dataset if len(d["text"]) > 10 and len(d["text"]) < 512][:32]
print(f"{len(calib_data)}個の短いサンプルデータを準備しました。")
# --- ここまで ---

print(f"'{model_path}' の量子化を開始します...")
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    safetensors=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True, 
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("\nモデルの量子化を実行します。この処理には時間がかかります...")
# 手動で準備した「短い」データを使用します
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data
)

print(f"\n量子化されたモデルを '{quant_path}' に保存します...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"\n🎉 量子化が完了しました！'{quant_path}' にモデルが保存されています。")