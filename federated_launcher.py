"""
FedVideoQA 联邦学习框架启动器
整合二分搜索、Qwen2.5-VL微调和联邦学习
"""

import argparse
import logging
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
import flwr as fl
from datetime import datetime

from src.core.binary_search import BinarySearchLocalizer
from src.core.deepseek_client import DeepSeekClient
from src.federated.fed_client import FedClient
from src.federated.fed_server import FedServer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'federated_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FedVideoQALauncher:
    """FedVideoQA联邦学习启动器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_data = []
        self.client_data = {}
        
        # 初始化组件
        self.deepseek_client = DeepSeekClient()
        self.localizer = BinarySearchLocalizer(self.deepseek_client)
        
    def load_dataset(self) -> List[Tuple[str, str]]:
        """加载视频数据集"""
        data_dir = Path(self.config['data_dir'])
        
        # 示例数据加载逻辑
        video_question_pairs = []
        
        if (data_dir / 'dataset.json').exists():
            # 从JSON文件加载
            with open(data_dir / 'dataset.json', 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                for item in dataset:
                    video_question_pairs.append((item['video_path'], item['question']))
        else:
            # 生成示例数据
            logger.warning("未找到dataset.json，使用示例数据")
            for i in range(self.config.get('num_videos', 10)):
                video_path = str(data_dir / f'video_{i}.mp4')
                question = f"这个视频片段中讨论了什么内容？{i}"
                video_question_pairs.append((video_path, question))
        
        logger.info(f"加载了 {len(video_question_pairs)} 个视频-问题对")
        return video_question_pairs
    
    def generate_training_data(self, video_question_pairs: List[Tuple[str, str]]) -> List[Tuple]:
        """使用二分搜索生成训练数据"""
        logger.info("开始生成训练数据...")
        
        all_training_data = []
        
        for i, (video_path, question) in enumerate(video_question_pairs):
            logger.info(f"处理视频 {i+1}/{len(video_question_pairs)}: {video_path}")
            
            try:
                # 检查视频文件是否存在
                if not os.path.exists(video_path):
                    logger.warning(f"视频文件不存在: {video_path}")
                    continue
                
                # 使用二分搜索生成训练数据
                training_data = self.localizer.generate_training_data(video_path, question)
                all_training_data.extend(training_data)
                
                logger.info(f"为视频 {video_path} 生成了 {len(training_data)} 个训练样本")
                
            except Exception as e:
                logger.error(f"处理视频 {video_path} 时出错: {e}")
                continue
        
        logger.info(f"总共生成了 {len(all_training_data)} 个训练样本")
        return all_training_data
    
    def distribute_data_to_clients(self, training_data: List[Tuple]) -> Dict[str, List[Tuple]]:
        """将数据分发给联邦客户端"""
        logger.info("分发数据到联邦客户端...")
        
        num_clients = self.config['num_clients']
        client_data = {}
        
        # 简单的数据分割策略
        samples_per_client = len(training_data) // num_clients
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            start_idx = i * samples_per_client
            
            if i == num_clients - 1:  # 最后一个客户端获得剩余所有数据
                end_idx = len(training_data)
            else:
                end_idx = start_idx + samples_per_client
            
            client_data[client_id] = training_data[start_idx:end_idx]
            
            # 模拟不同设备类型
            device_type = "mobile" if i % 3 == 0 else "desktop"
            
            logger.info(f"客户端 {client_id} ({device_type}): {len(client_data[client_id])} 个样本")
        
        return client_data
    
    def create_client_fn(self, client_data: Dict[str, List[Tuple]]):
        """创建客户端函数"""
        def client_fn(cid: str):
            # 获取客户端特定数据
            local_data = client_data.get(cid, [])
            device_type = "mobile" if "0" in cid or "3" in cid or "6" in cid else "desktop"
            
            return FedClient(
                client_id=cid, 
                local_data=local_data, 
                device_type=device_type,
                training_mode=self.config.get('training_mode', 'finetune'),
                model_name=self.config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                tuner_configs={
                    "lora": {
                        "r": self.config.get('lora_r', 16),
                        "lora_alpha": self.config.get('lora_alpha', 32),
                        "target_modules": self.config.get('lora_targets', ["q_proj", "v_proj"]),
                        "lora_dropout": self.config.get('lora_dropout', 0.1)
                    },
                    "training": {
                        "local_epochs": self.config.get('local_epochs', 3),
                        "batch_size": self.config.get('batch_size', 4),
                        "learning_rate": self.config.get('learning_rate', 1e-4)
                    }
                }
            )
        
        return client_fn
    
    def start_federated_training(self, client_data: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """启动联邦学习训练"""
        logger.info("启动联邦学习训练...")
        
        # 创建增强的联邦学习策略
        strategy = FedServer(
            fraction_fit=self.config.get('fraction_fit', 0.8),
            fraction_evaluate=self.config.get('fraction_evaluate', 0.5),
            min_fit_clients=self.config.get('min_fit_clients', 3),
            min_evaluate_clients=self.config.get('min_evaluate_clients', 3),
            min_available_clients=self.config.get('min_available_clients', 3),
            training_mode=self.config.get('training_mode', 'finetune'),
            model_save_path=self.config.get('model_save_path', './global_models'),
            convergence_threshold=self.config.get('convergence_threshold', 0.001),
            convergence_patience=self.config.get('convergence_patience', 5),
            client_selection_strategy=self.config.get('client_selection_strategy', 'random'),
            adaptive_lr=self.config.get('adaptive_lr', True)
        )
        
        # 配置客户端资源
        client_resources = {
            "num_cpus": self.config.get('client_cpus', 1),
            "num_gpus": self.config.get('client_gpus', 0.5) if torch.cuda.is_available() else 0
        }
        
        # 启动联邦学习仿真
        try:
            logger.info(f"启动 {self.config['num_clients']} 个客户端，运行 {self.config['num_rounds']} 轮训练")
            
            history = fl.simulation.start_simulation(
                client_fn=self.create_client_fn(client_data),
                num_clients=self.config['num_clients'],
                config=fl.server.ServerConfig(num_rounds=self.config['num_rounds']),
                strategy=strategy,
                client_resources=client_resources,
                ray_init_args={"include_dashboard": False}  # 禁用Ray dashboard以避免端口冲突
            )
            
            logger.info("联邦学习训练完成！")
            
            # 整理结果
            results = {
                "training_history": {
                    "losses": dict(history.losses_distributed) if history.losses_distributed else {},
                    "metrics": dict(history.metrics_distributed) if history.metrics_distributed else {},
                    "losses_centralized": dict(history.losses_centralized) if history.losses_centralized else {},
                    "metrics_centralized": dict(history.metrics_centralized) if history.metrics_centralized else {}
                },
                "round_metrics": getattr(strategy, 'round_metrics', []),
                "global_metrics": getattr(strategy, 'global_metrics', {}),
                "final_performance": {
                    "best_accuracy": getattr(strategy, 'global_metrics', {}).get('best_accuracy', 0.0),
                    "total_rounds": self.config['num_rounds'],
                    "convergence_round": getattr(strategy, 'global_metrics', {}).get('convergence_round'),
                    "converged": getattr(strategy, 'converged', False)
                },
                "config": self.config
            }
            
            # 保存全局模型
            if hasattr(strategy, 'save_global_model'):
                model_path = f"global_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                strategy.save_global_model(model_path)
                results["model_path"] = model_path
                logger.info(f"全局模型已保存到: {model_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"联邦学习训练失败: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"federated_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"训练结果已保存到: {results_file}")
            
            # 输出关键指标摘要
            final_perf = results.get("final_performance", {})
            logger.info("=== 训练结果摘要 ===")
            logger.info(f"最佳准确率: {final_perf.get('best_accuracy', 0):.4f}")
            logger.info(f"总训练轮数: {final_perf.get('total_rounds', 0)}")
            logger.info(f"收敛轮次: {final_perf.get('convergence_round', '未收敛')}")
            logger.info(f"是否收敛: {'是' if final_perf.get('converged', False) else '否'}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def run(self) -> None:
        """运行完整的联邦学习流程"""
        logger.info("=== FedVideoQA 联邦学习框架启动 ===")
        logger.info(f"配置参数: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        
        try:
            # 步骤1: 加载数据集
            logger.info("步骤1: 加载视频数据集")
            video_question_pairs = self.load_dataset()
            
            if not video_question_pairs:
                logger.error("没有加载到任何视频数据，退出")
                return
            
            # 步骤2: 生成训练数据
            logger.info("步骤2: 使用二分搜索生成训练数据")
            training_data = self.generate_training_data(video_question_pairs)
            
            if not training_data:
                logger.error("没有生成任何训练数据，退出")
                return
            
            # 步骤3: 分发数据到客户端
            logger.info("步骤3: 分发数据到联邦客户端")
            client_data = self.distribute_data_to_clients(training_data)
            
            # 验证客户端数据分发
            total_samples = sum(len(data) for data in client_data.values())
            logger.info(f"总共分发了 {total_samples} 个样本到 {len(client_data)} 个客户端")
            
            # 步骤4: 启动联邦学习
            logger.info("步骤4: 启动联邦学习训练")
            results = self.start_federated_training(client_data)
            
            # 步骤5: 保存结果
            logger.info("步骤5: 保存训练结果")
            self.save_results(results)
            
            logger.info("=== 联邦学习流程完成 ===")
            
        except Exception as e:
            logger.error(f"联邦学习流程失败: {e}")
            import traceback
            traceback.print_exc()
            raise

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        # 数据配置
        "data_dir": "data/",
        "num_videos": 10,
        
        # 联邦学习配置
        "num_clients": 5,
        "num_rounds": 20,
        "fraction_fit": 0.8,
        "fraction_evaluate": 0.5,
        "min_fit_clients": 3,
        "min_evaluate_clients": 3,
        "min_available_clients": 3,
        
        # 客户端资源配置
        "client_cpus": 1,
        "client_gpus": 0.5,
        
        # 模型配置
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "training_mode": "finetune",
        "model_save_path": "./global_models",
        
        # LoRA配置
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.1,
        
        # 训练配置
        "local_epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-4,
        
        # 策略配置
        "convergence_threshold": 0.001,
        "convergence_patience": 5,
        "client_selection_strategy": "random",
        "adaptive_lr": True,
    }

def main():
    parser = argparse.ArgumentParser(description="FedVideoQA 联邦学习框架启动器")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data/", help="数据目录")
    parser.add_argument("--num_clients", type=int, default=5, help="客户端数量")
    parser.add_argument("--num_rounds", type=int, default=20, help="联邦学习轮数")
    parser.add_argument("--num_videos", type=int, default=10, help="视频数量")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # 命令行参数覆盖配置
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.num_clients:
        config['num_clients'] = args.num_clients
    if args.num_rounds:
        config['num_rounds'] = args.num_rounds
    if args.num_videos:
        config['num_videos'] = args.num_videos
    
    # 创建数据目录
    os.makedirs(config['data_dir'], exist_ok=True)
    
    # 启动联邦学习框架
    launcher = FedVideoQALauncher(config)
    launcher.run()

if __name__ == "__main__":
    main() 