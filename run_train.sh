# 使用合成数据训练
python train_qwen.py --use_lora --num_epochs 3 --batch_size 2

# 使用真实数据训练
# python train_qwen.py --data_file data/frame_question_pairs.json --use_lora --num_epochs 5