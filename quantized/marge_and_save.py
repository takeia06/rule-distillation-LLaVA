import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 設定項目 ---
# LoRAでファインチューニングされたモデルのパス
peft_model_path = '/home/takei/LLaVA/checkpoints/llava-v1.6-vicuna-7b-mvtec/checkpoint-907' 
# マージ済みモデルの保存先パス
merged_model_path = 'checkpoints/llava-v1.6-vicuna-7b-mvtec-merged'

print(f"'{peft_model_path}' からモデルを読み込んでいます...")

# ベースモデルを読み込む
base_model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5", # ここはご自身のベースモデルに合わせてください
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# PEFTモデルを読み込み、ベースモデルに適用する
model = PeftModel.from_pretrained(
    base_model,
    peft_model_path
)

print("LoRAレイヤーをマージしています...")
# LoRAレイヤーをマージして、ベースモデルを返す
model = model.merge_and_unload()
print("マージが完了しました。")

print(f"マージ済みモデルを '{merged_model_path}' に保存します...")
model.save_pretrained(merged_model_path)

# トークナイザーもコピーする
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"\n🎉 マージ済みモデルの保存が完了しました！'{merged_model_path}' を確認してください。")