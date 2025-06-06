"""
简化版 FedVideoQA 训练脚本
Qwen2.5-VL 二分类器微调 - 最小功能实现
"""

import argparse
import logging
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoQADataset(Dataset):
    """简单的视频问答数据集"""
    
    def __init__(self, data: List[Tuple], processor, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame, question, label = self.data[idx]
        
        # 处理输入
        inputs = self.processor(
            text=question,
            images=frame,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # 移除batch维度
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        return inputs

class QwenBinaryClassifier(nn.Module):
    """基于Qwen的二分类器"""
    
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.backbone = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir='./cache'
        )
        
        # 添加分类头
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, **inputs):
        labels = inputs.pop('labels', None)
        
        # 获取特征
        outputs = self.backbone(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # 池化得到句子表示
        pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}

class SimpleTrainer:
    """简化的训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化处理器
        self.processor = AutoProcessor.from_pretrained(args.model_name,cache_dir='./cache')
        
        # 初始化模型
        self.model = QwenBinaryClassifier(args.model_name)
        
        # 应用LoRA
        if args.use_lora:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_targets,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self.model.to(self.device)
    
    def generate_synthetic_data(self) -> List[Tuple]:
        """生成合成数据用于测试"""
        logger.info("生成合成数据...")
        
        questions = [
            "这个画面中有人在说话吗？",
            "画面中显示了文字或字幕吗？", 
            "画面中有运动或动作吗？",
            "这个画面包含主要内容吗？",
            "这个画面与问题相关吗？",
            "画面显示了重要的视觉信息吗？",
            "画面中有清晰的对象或主体吗？",
            "这个画面包含相关的上下文吗？"
        ]
        
        data = []
        for i in range(self.args.num_samples):
            # 创建随机RGB图像（模拟视频帧）
            frame = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            
            question = questions[i % len(questions)]
            label = np.random.randint(0, 2)  # 随机二分类标签
            
            data.append((frame, question, label))
        
        logger.info(f"生成了 {len(data)} 个合成样本")
        return data
    
    def load_data_from_file(self, data_file: str) -> List[Tuple]:
        """从文件加载数据"""
        logger.info(f"从 {data_file} 加载数据...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        data = []
        for item in data_json.get('finetune', []):
            try:
                frame_path = item['frame_path']
                question = item['question']
                label = item['label']
                
                # 加载图像
                if os.path.exists(frame_path):
                    frame = Image.open(frame_path).convert('RGB')
                    data.append((frame, question, label))
                else:
                    logger.warning(f"图像文件不存在: {frame_path}")
            except Exception as e:
                logger.error(f"加载数据项失败: {e}")
        
        logger.info(f"加载了 {len(data)} 个样本")
        return data
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        # 准备数据
        if self.args.data_file and os.path.exists(self.args.data_file):
            data = self.load_data_from_file(self.args.data_file)
        else:
            data = self.generate_synthetic_data()
        
        # 创建数据集和加载器
        dataset = VideoQADataset(data, self.processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.learning_rate
        )
        
        # 训练循环
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}, "
                              f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        # 保存模型
        self.save_model()
    
    def collate_fn(self, batch):
        """批处理函数"""
        batch_dict = {}
        for key in batch[0].keys():
            if key == 'labels':
                batch_dict[key] = torch.stack([item[key] for item in batch])
            else:
                # 对于其他输入，尝试堆叠
                try:
                    batch_dict[key] = torch.stack([item[key] for item in batch])
                except:
                    batch_dict[key] = [item[key] for item in batch]
        return batch_dict
    
    def save_model(self):
        """保存模型"""
        save_path = f"qwen_binary_classifier_epoch_{self.args.num_epochs}.pth"
        
        if hasattr(self.model, 'save_pretrained'):
            # 如果是PEFT模型
            self.model.save_pretrained("./saved_model")
            logger.info("LoRA模型已保存到 ./saved_model")
        else:
            # 保存整个模型
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"模型已保存到 {save_path}")
        
        # 保存训练配置
        config = {
            "model_name": self.args.model_name,
            "num_epochs": self.args.num_epochs,
            "batch_size": self.args.batch_size,
            "learning_rate": self.args.learning_rate,
            "use_lora": self.args.use_lora,
            "lora_config": {
                "r": self.args.lora_r,
                "lora_alpha": self.args.lora_alpha,
                "target_modules": self.args.lora_targets,
                "lora_dropout": self.args.lora_dropout
            } if self.args.use_lora else None
        }
        
        with open("training_config.json", "w", encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("训练配置已保存到 training_config.json")

def main():
    parser = argparse.ArgumentParser(description="简化版 Qwen 二分类器训练")
    
    # 基础参数
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Qwen模型名称")
    parser.add_argument("--data_file", type=str, default=None,
                       help="训练数据文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="合成数据样本数量")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    
    # LoRA参数
    parser.add_argument("--use_lora", action="store_true",
                       help="是否使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_targets", type=str, nargs="+", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj"],
                       help="LoRA目标模块")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    args = parser.parse_args()
    
    # 初始化训练器并开始训练
    trainer = SimpleTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()