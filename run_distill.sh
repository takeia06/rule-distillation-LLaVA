#!/bin/bash

# 仮想環境をアクティベート
source rule-distill.venv/bin/activate

# 出力ディレクトリの作成
mkdir -p output/distill_model

# 蒸留の実行
python distill.py \
    --base_model llava-1.5-7b-hf \
    --teacher_model llava-1.5-7b-hf \
    --full_inst_desp_data_path data/Road_inspection/Road_inspection_full_train.json \
    --no_inst_desp_data_path data/Road_inspection/Road_inspection_no_train.json \
    --valid_data_path data/Road_inspection/Road_inspection_full_val.json \
    --image_base_path data \
    --output_dir output/distill_model \
    --batch_size 4 \
    --micro_batch_size 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --temperature 2 \
    --distill_loss_type KL \
    --distill_from_hidden_states True \
    --hidden_beta 100.0 \
    --cutoff_len 512