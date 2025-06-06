# python generate_va_pair.py --output_dir output --questions_file dataset/questions.json --mode balanced


CUDA_VISIBLE_DEVICES=0 python train_qwen.py --training_mode encoder --use_lora --freeze_backbone --data_file output/va_pair_balanced_train.json

CUDA_VISIBLE_DEVICES=0 python train_qwen.py --training_mode encoder --num_samples 2 --batch_size 1 --data_file output/va_pair_balanced_train.json


# 使用真实数据训练（指定GPU 0）
CUDA_VISIBLE_DEVICES=0 python train_qwen.py --training_mode instruction --use_lora --data_file output/va_pair_balanced_train.json
