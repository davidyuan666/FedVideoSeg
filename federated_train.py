"""
FedVideoQA 联邦学习训练脚本
基于现有的 train_qwen.py 实现联邦学习功能
增加隐私保护：差分隐私 + 安全聚合
"""

import argparse
import logging
import json
import torch
import copy
import hashlib
import random
from typing import List, Dict, Any
from train_qwen import UnifiedTrainer, VideoQADataset
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyEngine:
    """隐私保护引擎"""
    
    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
    def add_differential_privacy_noise(self, weights: Dict[str, torch.Tensor], 
                                     sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """添加差分隐私噪声"""
        noisy_weights = {}
        
        for key, weight in weights.items():
            if weight.requires_grad:
                # 计算噪声标准差
                noise_std = self.noise_multiplier * sensitivity
                
                # 生成高斯噪声
                noise = torch.normal(0, noise_std, size=weight.shape, 
                                   dtype=weight.dtype, device=weight.device)
                
                # 添加噪声
                noisy_weights[key] = weight + noise
            else:
                noisy_weights[key] = weight
                
        return noisy_weights
    
    def clip_gradients(self, model: torch.nn.Module):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
    
    def encrypt_weights(self, weights: Dict[str, torch.Tensor], 
                       client_id: int) -> Dict[str, torch.Tensor]:
        """简单的权重加密/混淆"""
        encrypted_weights = {}
        
        # 使用客户端ID生成随机种子
        random.seed(client_id * 42)
        
        for key, weight in weights.items():
            # 生成随机掩码
            mask = torch.rand_like(weight) * 0.1  # 小幅度的随机掩码
            encrypted_weights[key] = weight + mask
            
        # 重置随机种子
        random.seed()
        
        return encrypted_weights
    
    def decrypt_weights(self, encrypted_weights: Dict[str, torch.Tensor], 
                       client_id: int) -> Dict[str, torch.Tensor]:
        """解密权重"""
        decrypted_weights = {}
        
        # 使用相同的随机种子
        random.seed(client_id * 42)
        
        for key, weight in encrypted_weights.items():
            # 生成相同的随机掩码
            mask = torch.rand_like(weight) * 0.1
            decrypted_weights[key] = weight - mask
            
        # 重置随机种子
        random.seed()
        
        return decrypted_weights

class PrivateFederatedClient:
    """支持隐私保护的联邦学习客户端"""
    
    def __init__(self, client_id: int, args, data_subset: List, privacy_engine: PrivacyEngine):
        self.client_id = client_id
        self.args = args
        self.data = data_subset
        # 不在初始化时创建trainer，避免重复加载模型
        self.trainer = None
        self.privacy_engine = privacy_engine
        
        logger.info(f"隐私保护客户端 {client_id} 初始化完成，数据量: {len(data_subset)}")
    
    def _get_trainer(self):
        """延迟初始化trainer"""
        if self.trainer is None:
            # 修改args以避免device_map冲突
            client_args = copy.deepcopy(self.args)
            # 在federated learning中，我们手动管理设备分配
            self.trainer = UnifiedTrainer(client_args)
        return self.trainer
    
    def local_train(self, global_weights: Dict[str, torch.Tensor], epochs: int = 1):
        """本地训练（含隐私保护）"""
        logger.info(f"客户端 {self.client_id} 开始隐私保护本地训练...")
        
        # 获取trainer实例
        trainer = self._get_trainer()
        
        # 加载全局权重
        try:
            trainer.model.load_state_dict(global_weights, strict=False)
        except Exception as e:
            logger.warning(f"加载权重时出现警告: {e}")
            # 尝试更宽松的加载
            missing_keys, unexpected_keys = trainer.model.load_state_dict(global_weights, strict=False)
            if missing_keys:
                logger.info(f"缺失的键: {missing_keys}")
            if unexpected_keys:
                logger.info(f"意外的键: {unexpected_keys}")
        
        # 创建本地数据集
        dataset = VideoQADataset(
            self.data, 
            trainer.processor, 
            trainer.tokenizer,
            training_mode=self.args.training_mode
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            collate_fn=trainer.collate_fn
        )
        
        # 本地优化器
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(), 
            lr=self.args.learning_rate
        )
        
        # 训练
        trainer.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            for batch in dataloader:
                batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = trainer.model(**batch)
                loss = outputs['loss']
                loss.backward()
                
                # 梯度裁剪（隐私保护）
                if self.args.use_privacy:
                    self.privacy_engine.clip_gradients(trainer.model)
                
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / (epochs * len(dataloader))
        
        # 获取训练后的权重
        trained_weights = trainer.model.state_dict()
        
        # 应用隐私保护
        if self.args.use_privacy:
            # 添加差分隐私噪声
            trained_weights = self.privacy_engine.add_differential_privacy_noise(
                trained_weights, sensitivity=self.args.dp_sensitivity
            )
            
            # 简单加密
            if self.args.use_encryption:
                trained_weights = self.privacy_engine.encrypt_weights(
                    trained_weights, self.client_id
                )
        
        logger.info(f"客户端 {self.client_id} 隐私保护本地训练完成，平均损失: {avg_loss:.4f}")
        
        return trained_weights, len(self.data)

class PrivateFederatedServer:
    """支持隐私保护的联邦学习服务器"""
    
    def __init__(self, args, privacy_engine: PrivacyEngine):
        self.args = args
        # 只在服务器端创建一个全局模型实例
        try:
            trainer = UnifiedTrainer(args)
            self.global_model = trainer.model
            self.device = trainer.device
        except Exception as e:
            logger.error(f"初始化全局模型失败: {e}")
            # 降级方案：使用简单的模型初始化
            if args.training_mode == "encoder":
                from train_qwen import QwenEncoderClassifier
                self.global_model = QwenEncoderClassifier(args.model_name, freeze_backbone=args.freeze_backbone)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.global_model.to(self.device)
            else:
                raise e
        
        self.clients = []
        self.privacy_engine = privacy_engine
        
    def add_client(self, client: PrivateFederatedClient):
        """添加客户端"""
        self.clients.append(client)
    
    def secure_federated_averaging(self, client_weights: List[Dict], 
                                 client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """安全联邦平均算法"""
        logger.info("执行安全联邦平均...")
        
        # 如果启用了加密，先解密
        if self.args.use_encryption:
            decrypted_weights = []
            for i, weights in enumerate(client_weights):
                decrypted = self.privacy_engine.decrypt_weights(weights, i)
                decrypted_weights.append(decrypted)
            client_weights = decrypted_weights
        
        # 计算权重
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
        
        # 初始化平均权重
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] * weights[0]
        
        # 加权平均
        for i in range(1, len(client_weights)):
            for key in avg_weights.keys():
                if key in client_weights[i]:
                    avg_weights[key] += client_weights[i][key] * weights[i]
        
        # 可选：在服务器端也添加一些噪声
        if self.args.use_privacy and self.args.server_noise:
            avg_weights = self.privacy_engine.add_differential_privacy_noise(
                avg_weights, sensitivity=self.args.dp_sensitivity * 0.5
            )
        
        return avg_weights
    
    def train_round(self, round_num: int, local_epochs: int = 1):
        """单轮联邦训练"""
        logger.info(f"开始第 {round_num} 轮隐私保护联邦训练...")
        
        # 获取当前全局权重
        global_weights = self.global_model.state_dict()
        
        # 客户端本地训练
        client_weights = []
        client_sizes = []
        
        for client in self.clients:
            weights, size = client.local_train(global_weights, local_epochs)
            client_weights.append(weights)
            client_sizes.append(size)
        
        # 安全联邦平均
        new_global_weights = self.secure_federated_averaging(client_weights, client_sizes)
        
        # 更新全局模型
        self.global_model.load_state_dict(new_global_weights)
        
        logger.info(f"第 {round_num} 轮隐私保护联邦训练完成")
    
    def save_global_model(self, round_num: int):
        """保存全局模型"""
        save_path = f"private_federated_model_round_{round_num}.pth"
        torch.save(self.global_model.state_dict(), save_path)
        logger.info(f"隐私保护全局模型已保存到 {save_path}")
    
    def privacy_audit(self, round_num: int):
        """隐私审计报告"""
        if self.args.use_privacy:
            epsilon = self.calculate_privacy_budget(round_num)
            logger.info(f"隐私审计 - 第 {round_num} 轮:")
            logger.info(f"  估算隐私预算 (ε): {epsilon:.4f}")
            logger.info(f"  噪声倍数: {self.privacy_engine.noise_multiplier}")
            logger.info(f"  梯度裁剪范数: {self.privacy_engine.max_grad_norm}")
    
    def calculate_privacy_budget(self, round_num: int) -> float:
        """计算隐私预算（简化版）"""
        # 简化的隐私预算计算
        # 实际应用中需要更精确的计算
        base_epsilon = 1.0 / (self.privacy_engine.noise_multiplier ** 2)
        total_epsilon = base_epsilon * round_num
        return total_epsilon

def split_data_federated(data: List, num_clients: int, split_type: str = "iid"):
    """将数据分割给多个客户端"""
    logger.info(f"将数据分割给 {num_clients} 个客户端，分割类型: {split_type}")
    
    if split_type == "iid":
        # IID分割：随机分配
        np.random.shuffle(data)
        client_data = []
        chunk_size = len(data) // num_clients
        
        for i in range(num_clients):
            start = i * chunk_size
            end = start + chunk_size if i < num_clients - 1 else len(data)
            client_data.append(data[start:end])
    
    else:  # non-iid
        # Non-IID分割：按标签分布
        # 简化实现：让每个客户端主要拥有某种标签的数据
        positive_data = [item for item in data if item[2] == 1]
        negative_data = [item for item in data if item[2] == 0]
        
        client_data = [[] for _ in range(num_clients)]
        
        # 主要数据分配
        for i in range(num_clients):
            if i % 2 == 0:  # 偶数客户端主要分配正样本
                main_data = positive_data[i//2 * len(positive_data)//((num_clients+1)//2):(i//2+1) * len(positive_data)//((num_clients+1)//2)]
                minor_data = negative_data[i//2 * len(negative_data)//(num_clients*2):(i//2+1) * len(negative_data)//(num_clients*2)]
            else:  # 奇数客户端主要分配负样本
                main_data = negative_data[i//2 * len(negative_data)//(num_clients//2):(i//2+1) * len(negative_data)//(num_clients//2)]
                minor_data = positive_data[i//2 * len(positive_data)//(num_clients*2):(i//2+1) * len(positive_data)//(num_clients*2)]
            
            client_data[i] = main_data + minor_data
    
    # 打印分割统计
    for i, data_subset in enumerate(client_data):
        pos_count = sum(1 for item in data_subset if item[2] == 1)
        neg_count = len(data_subset) - pos_count
        logger.info(f"客户端 {i}: 总数据 {len(data_subset)}, 正样本 {pos_count}, 负样本 {neg_count}")
    
    return client_data

def main():
    parser = argparse.ArgumentParser(description="FedVideoQA 隐私保护联邦学习训练")
    
        # 基础参数（复用train_qwen.py的参数）
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Qwen模型名称")
    parser.add_argument("--data_file", type=str, required=True,
                       help="训练数据文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="合成数据样本数量")
    
    # 训练模式
    parser.add_argument("--training_mode", type=str, 
                       choices=["encoder", "instruction"], 
                       default="encoder",
                       help="训练模式")
    parser.add_argument("--freeze_backbone", action="store_true",
                       help="冻结backbone")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=2,
                       help="批大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    
    # LoRA参数
    parser.add_argument("--use_lora", action="store_true",
                       help="使用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_targets", type=str, nargs="+", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj"],
                       help="LoRA目标模块")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # 联邦学习参数
    parser.add_argument("--num_clients", type=int, default=3,
                       help="客户端数量")
    parser.add_argument("--num_rounds", type=int, default=5,
                       help="联邦学习轮数")
    parser.add_argument("--local_epochs", type=int, default=1,
                       help="每轮本地训练轮数")
    parser.add_argument("--split_type", type=str, 
                       choices=["iid", "non_iid"], 
                       default="iid",
                       help="数据分割类型")
    
    # 隐私保护参数
    parser.add_argument("--use_privacy", action="store_true",
                       help="启用隐私保护")
    parser.add_argument("--noise_multiplier", type=float, default=1.0,
                       help="差分隐私噪声倍数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪最大范数")
    parser.add_argument("--dp_sensitivity", type=float, default=1.0,
                       help="差分隐私敏感度")
    parser.add_argument("--use_encryption", action="store_true",
                       help="启用简单加密")
    parser.add_argument("--server_noise", action="store_true",
                       help="在服务器端也添加噪声")
    
    args = parser.parse_args()
    
    logger.info("开始FedVideoQA隐私保护联邦学习训练...")
    logger.info(f"客户端数量: {args.num_clients}")
    logger.info(f"联邦学习轮数: {args.num_rounds}")
    logger.info(f"数据分割类型: {args.split_type}")
    logger.info(f"隐私保护: {'启用' if args.use_privacy else '禁用'}")
    if args.use_privacy:
        logger.info(f"  噪声倍数: {args.noise_multiplier}")
        logger.info(f"  梯度裁剪: {args.max_grad_norm}")
        logger.info(f"  简单加密: {'启用' if args.use_encryption else '禁用'}")
    
    # 加载数据
    if not os.path.exists(args.data_file):
        logger.error(f"数据文件不存在: {args.data_file}")
        return
    
    trainer = UnifiedTrainer(args)
    data = trainer.load_data_from_file(args.data_file)
    
    # 分割数据给客户端
    client_data_splits = split_data_federated(data, args.num_clients, args.split_type)
    
    # 初始化隐私引擎
    privacy_engine = PrivacyEngine(
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm
    ) if args.use_privacy else None
    
    # 初始化服务器
    server = PrivateFederatedServer(args, privacy_engine)
    
    # 创建客户端
    for i in range(args.num_clients):
        client = PrivateFederatedClient(i, args, client_data_splits[i], privacy_engine)
        server.add_client(client)
    
    # 联邦学习训练
    for round_num in range(1, args.num_rounds + 1):
        server.train_round(round_num, args.local_epochs)
        
        # 隐私审计
        if args.use_privacy:
            server.privacy_audit(round_num)
        
        # 每几轮保存一次模型
        if round_num % 2 == 0:
            server.save_global_model(round_num)
    
    # 最终保存
    server.save_global_model(args.num_rounds)
    
    # 最终隐私报告
    if args.use_privacy:
        final_epsilon = server.calculate_privacy_budget(args.num_rounds)
        logger.info(f"🔒 最终隐私预算: ε = {final_epsilon:.4f}")
        logger.info(f"🔒 隐私保护级别: {'强' if final_epsilon < 1.0 else '中等' if final_epsilon < 10.0 else '较弱'}")
    
    logger.info("隐私保护联邦学习训练完成！")

if __name__ == "__main__":
    main() 