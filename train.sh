# 只进行微调
python train_qwen.py --training_mode finetune --num_rounds 50

# 只进行对齐调优
python train_qwen.py --training_mode alignment --dpo_beta 0.2

# 同时进行微调和对齐
python train_qwen.py --training_mode both --num_clients 10

# 自定义 LoRA 配置
python train_qwen.py --training_mode finetune --lora_r 32 --lora_alpha 64