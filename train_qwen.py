"""
改进版 FedVideoQA 训练脚本
支持两种训练方式：
1. 编码器模式：将Qwen2.5-VL作为特征提取器 + 分类头
2. 指令微调模式：使用指令模板进行端到端微调
"""

import argparse
import logging
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoQADataset(Dataset):
    """支持两种模式的视频问答数据集"""
    
    def __init__(self, data: List[Tuple], processor, tokenizer, max_length=512, training_mode="encoder"):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.training_mode = training_mode
        
        # 指令微调模式的提示模板
        self.instruction_template = "请根据图像内容回答问题：{question}\n这个问题与图像是否相关？请只回答'是'或'否'。"
        
        # 获取是/否的token id
        self.yes_token_id = self.tokenizer.encode("是", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("否", add_special_tokens=False)[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame, question, label = self.data[idx]
        
        if self.training_mode == "encoder":
            return self._get_encoder_item(frame, question, label)
        else:  # instruction mode
            return self._get_instruction_item(frame, question, label)
    
    def _get_encoder_item(self, frame, question, label):
        """编码器模式的数据处理"""
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
    
    def _get_instruction_item(self, frame, question, label):
        """指令微调模式的数据处理"""
        # 构建指令
        instruction = self.instruction_template.format(question=question)
        target = "是" if label == 1 else "否"
        
        # 处理输入
        inputs = self.processor(
            text=instruction,
            images=frame,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # 处理目标输出
        target_token_id = self.yes_token_id if label == 1 else self.no_token_id
        
        # 移除batch维度
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        # 为指令微调准备标签
        input_ids = inputs['input_ids']
        labels = input_ids.clone()
        labels[:-1] = -100  # 忽略输入部分的损失
        labels[-1] = target_token_id  # 只计算最后一个token的损失
        
        inputs['labels'] = labels
        inputs['target_token_id'] = torch.tensor(target_token_id, dtype=torch.long)
        return inputs

class QwenEncoderClassifier(nn.Module):
    """改进的基于Qwen的编码器 + 分类器"""
    
    def __init__(self, model_name: str, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir='./cache'
        )
        
        # 是否冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 改进的特征提取和分类头
        hidden_size = self.backbone.config.hidden_size
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 注意力池化层
        self.attention_pool = nn.Linear(hidden_size, 1)
    
    def forward(self, **inputs):
        labels = inputs.pop('labels', None)
        
        # 获取特征
        if hasattr(self.backbone, 'parameters') and not next(self.backbone.parameters()).requires_grad:
            with torch.no_grad():
                outputs = self.backbone(**inputs, output_hidden_states=True)
        else:
            outputs = self.backbone(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]
        
        # 注意力池化
        attention_weights = torch.softmax(
            self.attention_pool(hidden_states).squeeze(-1), dim=-1
        ).unsqueeze(-1)
        pooled_output = torch.sum(hidden_states * attention_weights, dim=1)
        
        # 分类
        logits = self.feature_projection(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}

class QwenInstructionClassifier(nn.Module):
    """基于指令微调的分类器"""
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir='./cache'
        )
    
    def forward(self, **inputs):
        labels = inputs.pop('labels', None)
        target_token_id = inputs.pop('target_token_id', None)
        
        # 标准的语言模型损失计算
        outputs = self.model(**inputs, labels=labels)
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "target_token_id": target_token_id
        }
    
    def generate_answer(self, **inputs):
        """生成回答用于推理"""
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.model.config.eos_token_id,
                temperature=0.1
            )
        return generated

class UnifiedTrainer:
    """统一的训练器，支持两种训练模式"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化处理器和分词器
        self.processor = AutoProcessor.from_pretrained(args.model_name, cache_dir='./cache')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='./cache')
        
        # 根据训练模式初始化不同的模型
        if args.training_mode == "encoder":
            self.model = QwenEncoderClassifier(
                args.model_name, 
                freeze_backbone=args.freeze_backbone
            )
            # 应用LoRA到编码器模式
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
        else:  # instruction mode
            # 对于指令微调模式，直接使用原始模型并应用LoRA
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir='./cache'
            )
            
            if args.use_lora:
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=args.lora_targets,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                self.model = get_peft_model(base_model, lora_config)
            else:
                self.model = base_model
        
        self.model.to(self.device)
    
    def generate_synthetic_data(self) -> List[Tuple]:
        """生成合成数据用于测试，专注于问题与图像的相关性"""
        logger.info("生成合成数据...")
        
        # 相关性问题 - 模拟视频问答中问题与帧的相关性判断
        questions = [
            "视频中出现了什么动物？",  # 可能相关
            "这个场景发生在什么时间？",  # 可能相关
            "画面中有几个人？",  # 可能相关
            "视频的背景音乐是什么？",  # 不相关（纯视觉无法判断）
            "这个品牌的历史是什么？",  # 不相关
            "画面中的天气如何？",  # 可能相关
            "视频的制作成本是多少？",  # 不相关
            "画面中显示的文字内容是什么？",  # 可能相关
        ]
        
        data = []
        for i in range(self.args.num_samples):
            # 创建随机RGB图像（模拟视频帧）
            frame = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            
            question = questions[i % len(questions)]
            
            # 模拟相关性标签：前4个问题类型更可能相关，后4个不相关
            if i % len(questions) < 4:
                label = np.random.choice([0, 1], p=[0.3, 0.7])  # 70%概率相关
            else:
                label = np.random.choice([0, 1], p=[0.8, 0.2])  # 20%概率相关
            
            data.append((frame, question, label))
        
        logger.info(f"生成了 {len(data)} 个合成样本")
        return data
    
    def load_data_from_file(self, data_file: str) -> List[Tuple]:
        """从文件加载数据 - 适配va_pair格式"""
        logger.info(f"从 {data_file} 加载数据...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        
        data = []
        # 如果是列表格式（va_pair格式）
        if isinstance(data_json, list):
            for item in data_json:
                try:
                    frame_path = item['image_path']
                    # 修复路径分隔符问题
                    frame_path = frame_path.replace('\\', os.sep)
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
        
        # 如果是字典格式（原来的格式）
        elif isinstance(data_json, dict):
            for item in data_json.get('finetune', []):
                try:
                    frame_path = item['frame_path']
                    # 修复路径分隔符问题
                    frame_path = frame_path.replace('\\', os.sep)
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
        logger.info(f"开始{self.args.training_mode}模式训练...")
        
        # 准备数据
        if self.args.data_file and os.path.exists(self.args.data_file):
            data = self.load_data_from_file(self.args.data_file)
        else:
            data = self.generate_synthetic_data()
        
        # 创建数据集和加载器
        dataset = VideoQADataset(
            data, 
            self.processor, 
            self.tokenizer,
            training_mode=self.args.training_mode
        )
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
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # 前向传播 - 根据训练模式选择不同的计算方式
                if self.args.training_mode == "encoder":
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    # 计算准确率
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    labels = batch['labels']
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
                else:  # instruction mode
                    labels = batch.pop('labels', None)
                    target_token_id = batch.pop('target_token_id', None)
                    
                    # 标准的语言模型损失计算
                    outputs = self.model(**batch, labels=labels)
                    loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}, "
                              f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            if self.args.training_mode == "encoder" and total_samples > 0:
                accuracy = correct_predictions / total_samples
                logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}, "
                          f"准确率: {accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        # 保存模型
        self.save_model()
    
    def collate_fn(self, batch):
        """改进的批处理函数，正确处理变长序列"""
        # 分离不同类型的数据
        batch_dict = {}
        
        # 处理标签和target_token_id - 这些可以直接堆叠
        for key in ['labels', 'target_token_id']:
            if key in batch[0]:
                batch_dict[key] = torch.stack([item[key] for item in batch])
        
        # 对于需要padding的序列数据，使用tokenizer的pad功能
        sequence_keys = ['input_ids', 'attention_mask']
        for key in sequence_keys:
            if key in batch[0]:
                sequences = [item[key] for item in batch]
                # 手动pad序列到相同长度
                max_len = max(len(seq) for seq in sequences)
                padded_sequences = []
                
                for seq in sequences:
                    if key == 'input_ids':
                        # 使用tokenizer的pad_token_id进行padding
                        pad_token_id = self.tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = self.tokenizer.eos_token_id
                        padded = torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=pad_token_id)
                    else:  # attention_mask
                        # attention_mask用0进行padding
                        padded = torch.nn.functional.pad(seq, (0, max_len - len(seq)), value=0)
                    padded_sequences.append(padded)
                
                batch_dict[key] = torch.stack(padded_sequences)
        
        # 处理其他tensor类型的数据
        for key in batch[0].keys():
            if key not in batch_dict:
                values = [item[key] for item in batch]
                
                # 检查是否都是tensor
                if all(isinstance(v, torch.Tensor) for v in values):
                    try:
                        # 对于图像相关的tensor，应该形状一致，直接堆叠
                        if key.startswith('pixel_values') or key == 'image_patches':
                            batch_dict[key] = torch.stack(values)
                        else:
                            # 其他tensor类型，尝试堆叠
                            batch_dict[key] = torch.stack(values)
                    except Exception as e:
                        # 如果是关键的图像数据堆叠失败，抛出错误
                        if key.startswith('pixel_values') or key == 'image_patches':
                            logger.error(f"Failed to stack {key}: {e}")
                            logger.error(f"Tensor shapes: {[v.shape for v in values]}")
                            raise RuntimeError(f"Critical tensor {key} cannot be stacked")
                        else:
                            # 其他数据保持为列表
                            batch_dict[key] = values
                else:
                    # 非tensor数据保持为列表
                    batch_dict[key] = values
        
        return batch_dict
    
    def save_model(self):
        """保存训练好的模型"""
        save_dir = f"./saved_models/{self.args.training_mode}_model"
        os.makedirs(save_dir, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            # 如果是PEFT模型或标准transformers模型
            self.model.save_pretrained(save_dir)
            logger.info(f"模型已保存到 {save_dir}")
        else:
            # 如果是自定义模型，保存state_dict
            torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))
            logger.info(f"模型权重已保存到 {save_dir}/model.pth")
        
        # 同时保存processor和tokenizer
        self.processor.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FedVideoQA Qwen训练脚本")
    
    # 基本参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="预训练模型名称")
    parser.add_argument("--training_mode", type=str, choices=["encoder", "instruction"], 
                       default="encoder", help="训练模式: encoder 或 instruction")
    parser.add_argument("--data_file", type=str, default=None,
                       help="训练数据文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="合成数据样本数量（当没有数据文件时使用）")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批处理大小")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    
    # 模型参数
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="是否冻结backbone（仅encoder模式）")
    
    # LoRA参数
    parser.add_argument("--use_lora", action="store_true",
                       help="是否使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora_targets", type=str, nargs="+", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj"],
                       help="LoRA目标模块")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    logger.info("="*50)
    logger.info("FedVideoQA Qwen训练脚本启动")
    logger.info(f"训练模式: {args.training_mode}")
    logger.info(f"模型: {args.model_name}")
    logger.info(f"数据文件: {args.data_file}")
    logger.info(f"使用LoRA: {args.use_lora}")
    logger.info("="*50)
    
    try:
        # 创建训练器
        trainer = UnifiedTrainer(args)
        
        # 开始训练
        trainer.train()
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
    

