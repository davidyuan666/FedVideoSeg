# 自定义参数
# python generate_va_pair.py --mode balanced --max_samples 20 --split_ratio 0.8 --output_dir output --questions_file dataset/questions.json

# python generate_va_pair.py --output_dir output --questions_file dataset/questions.json --mode cross --negative_ratio 1.5

# python generate_va_pair.py --output_dir output --questions_file dataset/questions.json --mode balanced



# 使用合成数据训练（指定GPU 0）
CUDA_VISIBLE_DEVICES=0 python train_qwen.py --data_file output/va_pair_balanced_train.json --training_mode encoder --freeze_backbone --use_lora --num_epochs 5 --batch_size 4 --learning_rate 2e-5 --lora_r 16 --lora_alpha 32

# 使用真实数据训练（指定GPU 0）
CUDA_VISIBLE_DEVICES=0 python train_qwen.py --data_file output/va_pair_balanced_train.json --training_mode instruction --use_lora --num_epochs 3 --batch_size 2 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16