import torch
from awq.models.llava_next import LlavaNextAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# --- 設定項目 ---
# 【重要】ステップAで保存した「マージ済みモデル」のパスを指定します
model_path = 'llava-v1.6-vicuna-7b-merged'
# 量子化済みモデルの保存先パス
quant_path = 'llava-v1.6-7b-quantized-awq'

# 量子化の設定
quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM"
}

# --- キャリブレーションデータ作成 ---
print("メモリ節約のため、短いキャリブレーションデータを手動で準備します...")
# 注意：Llavaモデルのキャリブレーションには、テキストだけでなく画像も考慮するのが理想ですが、
# ここではまず動作させるためにテキストデータを使用します。
# もし量子化の品質が低い場合は、この部分を実際のタスクに近いデータ（画像＋テキスト）で
# 準備する必要があります。
calib_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
calib_data = [d["text"] for d in calib_dataset if len(d["text"]) > 10 and len(d["text"]) < 512][:32]
print(f"{len(calib_data)}個の短いサンプルデータを準備しました。")
# --- ここまで ---

print(f"'{model_path}' の量子化を開始します...")
# マージ済みモデルをAWQ用のクラスで読み込む
model = LlavaNextAWQForCausalLM.from_pretrained(
    model_path,
    model_type="llava_next",
    trust_remote_code=True,
    safetensors=True, # マージ済みモデルはsafetensors形式で保存されているはず
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True, 
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("\nモデルの量子化を実行します。この処理には時間がかかります...")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data
)

print(f"\n量子化されたモデルを '{quant_path}' に保存します...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"\n🎉 量子化が完了しました！'{quant_path}' にモデルが保存されています。")