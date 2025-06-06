# 使用合成数据训练
python train_qwen.py --training_mode encoder --freeze_backbone --use_lora --num_epochs 3 --batch_size 1

# 使用真实数据训练
python train_qwen.py --data_file data/frame_question_pairs.json --training_mode instruction --use_lora --learning_rate 1e-5