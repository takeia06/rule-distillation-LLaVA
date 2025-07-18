# Rule-Distillation LLaVA: ルール蒸留によるLLaVAの性能向上プロジェクト

このプロジェクトは、大規模言語モデル（LLM）であるLLaVAに対して、特定のルールを「蒸留」させることで、性能の向上や挙動の制御を目指す研究・開発プロジェクトです。

## 🚀 プロジェクトの目的・特徴

このプロジェクトでは、主に以下の3つのアプローチを試みています。

* **ファインチューニング (Finetuning):** `finetune/` ディレクトリにて、特定のデータセットに対するモデルの適応・性能向上を図ります。
* **ルール蒸留 (Distillation):** `distill/` ディレクトリ内のコードにより、特定のタスクを解くためのルール知識をLLaVAに学習させます。
* **量子化 (Quantization):** `quantized/` ディレクトリのスクリプトを使い、モデルを軽量化し、より少ない計算資源で動作させることを目指します。

## 🛠️ 使用技術

* **言語:** Python
* **主要ライブラリ:** PyTorch, Transformers (Hugging Face), BitsandBytes
* **ベースモデル:** LLaVA (Large Language and Vision Assistant)

## 使い方

### 1. セットアップ

まず、リポジトリをクローンし、必要なライブラリをインストールします。

```bash
git clone [https://github.com/takeia06/rule-distillation-LLaVA.git](https://github.com/takeia06/rule-distillation-LLaVA.git)
cd rule-distillation-LLaVA
pip install -r requirements.txt  
