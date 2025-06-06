# 自定义参数
# python generate_va_pair.py --mode balanced --max_samples 20 --split_ratio 0.8 --output_dir output --questions_file dataset/questions.json

# python generate_va_pair.py --output_dir output --questions_file dataset/questions.json --mode cross --negative_ratio 1.5

# python generate_va_pair.py --output_dir output --questions_file dataset/questions.json --mode balanced



# CUDA_VISIBLE_DEVICES=0 python train_qwen.py --training_mode encoder --use_lora --freeze_backbone --data_file data/va_pair_balanced_train.json

# CUDA_VISIBLE_DEVICES=0 python train_qwen.py --training_mode encoder --num_samples 2 --batch_size 1 --data_file data/va_pair_balanced_train.json


# 使用真实数据训练（指定GPU 0）
CUDA_VISIBLE_DEVICES=0 python train_qwen.py --training_mode instruction --use_lora --data_file data/va_pair_balanced_train.json
