#!/bin/bash

# 联邦学习训练脚本 (支持隐私保护)

# Stage 1: Encoder模式隐私保护联邦学习
echo "开始Stage 1: Encoder模式隐私保护联邦学习..."
python federated_train.py \
    --data_file output/va_pair_balanced_train.json \
    --training_mode encoder \
    --freeze_backbone \
    --use_lora \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_clients 3 \
    --num_rounds 3 \
    --local_epochs 1 \
    --split_type iid \
    --use_privacy \
    --noise_multiplier 1.2 \
    --max_grad_norm 1.0 \
    --dp_sensitivity 0.8 \
    --use_encryption \
    --server_noise

echo "Stage 1 完成！"

# Stage 2: Instruction模式隐私保护联邦学习
echo "开始Stage 2: Instruction模式隐私保护联邦学习..."
python federated_train.py \
    --data_file output/va_pair_balanced_train.json \
    --training_mode instruction \
    --use_lora \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --lora_r 8 \
    --lora_alpha 16 \
    --num_clients 3 \
    --num_rounds 2 \
    --local_epochs 1 \
    --split_type iid \
    --use_privacy \
    --noise_multiplier 1.0 \
    --max_grad_norm 0.8 \
    --dp_sensitivity 0.5 \
    --use_encryption

echo "隐私保护联邦学习训练完成！"
echo "🔒 已启用差分隐私、梯度裁剪和安全聚合功能"
echo "📊 训练完成后请查看隐私预算报告"
