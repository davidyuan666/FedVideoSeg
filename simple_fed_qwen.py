"""
简化版联邦学习Qwen-VL微调框架
集成CLIP和DeepSeek自动生成训练数据
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from transformers import AutoModel, AutoProcessor, CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
import cv2
import os
from pathlib import Path
import requests
import json
import time
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if not self.api_key:
            logger.warning("DeepSeek API key未设置，将使用模拟响应")
    
    def generate_question(self, frame_description: str) -> str:
        """根据帧描述生成问题"""
        if not self.api_key:
            # 模拟响应
            return f"这个场景中{frame_description.split('，')[0]}吗？"
        
        prompt = f"""
        基于以下视频帧的描述，生成一个相关的问题。问题应该是关于视频内容的，可以通过观看这个帧来回答。
        
        帧描述：{frame_description}
        
        请生成一个简洁、清晰的问题：
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                question = result["choices"][0]["message"]["content"].strip()
                return question
            else:
                logger.error(f"DeepSeek API错误: {response.status_code}")
                return f"这个场景中显示了什么？"
                
        except Exception as e:
            logger.error(f"调用DeepSeek API失败: {e}")
            return f"这个场景中显示了什么？"
    
    def evaluate_relevance(self, frame_description: str, question: str) -> float:
        """评估帧描述与问题的相关性"""
        if not self.api_key:
            # 模拟相关性评分
            return 0.8 if any(word in question for word in frame_description.split()) else 0.3
        
        prompt = f"""
        评估以下视频帧描述与问题的相关性，返回0-1之间的分数。
        
        帧描述：{frame_description}
        问题：{question}
        
        请只返回一个0-1之间的数字，表示相关性分数：
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                score_text = result["choices"][0]["message"]["content"].strip()
                try:
                    score = float(score_text)
                    return max(0.0, min(1.0, score))
                except ValueError:
                    return 0.5
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"评估相关性失败: {e}")
            return 0.5

class CLIPFrameAnalyzer:
    """使用CLIP分析视频帧"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        
        logger.info(f"CLIP模型已加载: {model_name}")
    
    def describe_frame(self, frame: np.ndarray) -> str:
        """使用CLIP生成帧的描述"""
        # 转换为PIL图像
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        # 预定义的描述模板
        descriptions = [
            "一个人在说话",
            "多个人在交谈", 
            "室内场景",
            "户外场景",
            "会议或讨论",
            "演讲或展示",
            "日常活动",
            "工作场景",
            "学习环境",
            "娱乐活动"
        ]
        
        try:
            # 处理图像和文本
            inputs = self.processor(
                text=descriptions,
                images=frame,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 获取相似度分数
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # 选择最相关的描述
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            
            best_description = descriptions[best_idx]
            
            # 如果置信度较低，使用通用描述
            if confidence < 0.3:
                best_description = "视频场景"
            
            logger.debug(f"帧描述: {best_description} (置信度: {confidence:.3f})")
            return best_description
            
        except Exception as e:
            logger.error(f"CLIP分析帧失败: {e}")
            return "视频场景"

class SimpleVideoDataset(Dataset):
    """简化的视频数据集"""
    def __init__(self, data: List[Tuple]):
        self.data = data  # (frame, question, label)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame, question, label = self.data[idx]
        return frame, question, torch.tensor(label, dtype=torch.long)

class SimpleQwenVL(nn.Module):
    """简化的Qwen-VL模型"""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # 添加LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # 分类头
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        
    def forward(self, frames: List, questions: List[str]):
        # 处理输入
        inputs = self.processor(text=questions, images=frames, return_tensors="pt", padding=True)
        
        # 获取特征
        outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled_output)
        return logits

class SimpleFedClient(fl.client.NumPyClient):
    """简化的联邦学习客户端"""
    def __init__(self, client_id: str, data: List[Tuple]):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型
        self.model = SimpleQwenVL()
        self.model.to(self.device)
        
        # 数据
        self.dataset = SimpleVideoDataset(data)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)
        
        # 优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
    def get_parameters(self, config):
        """获取模型参数"""
        return [param.detach().cpu().numpy() for param in self.model.parameters() if param.requires_grad]
    
    def set_parameters(self, parameters):
        """设置模型参数"""
        param_dict = dict(zip([name for name, param in self.model.named_parameters() if param.requires_grad], parameters))
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in param_dict:
                param.data = torch.from_numpy(param_dict[name]).to(param.device)
    
    def fit(self, parameters, config):
        """本地训练"""
        self.set_parameters(parameters)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for frames, questions, labels in self.dataloader:
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(frames, questions)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 5:  # 限制训练批次
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Client {self.client_id} - Training Loss: {avg_loss:.4f}")
        
        return self.get_parameters({}), len(self.dataset), {"train_loss": avg_loss}
    
    def evaluate(self, parameters, config):
        """本地评估"""
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, questions, labels in self.dataloader:
                labels = labels.to(self.device)
                logits = self.model(frames, questions)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0
        
        return avg_loss, len(self.dataset), {"accuracy": accuracy}

def extract_frame(video_path: str, timestamp: float = 0.0):
    """从视频提取帧"""
    if not os.path.exists(video_path):
        logger.warning(f"视频文件不存在: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # 转换BGR到RGB (OpenCV默认是BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    else:
        logger.warning(f"无法从视频 {video_path} 的时间戳 {timestamp} 提取帧")
        return None

def generate_data_from_videos(
    video_dir: str = "data/videos", 
    num_samples_per_video: int = 5,
    deepseek_api_key: str = None
) -> List[Tuple]:
    """使用CLIP和DeepSeek从真实视频生成训练数据"""
    
    # 初始化组件
    clip_analyzer = CLIPFrameAnalyzer()
    deepseek_client = DeepSeekClient(deepseek_api_key)
    
    data = []
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        logger.warning(f"视频目录不存在: {video_dir}")
        return generate_simple_data()
    
    # 查找视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning(f"在 {video_dir} 中未找到视频文件")
        return generate_simple_data()
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    for video_file in video_files:
        logger.info(f"处理视频: {video_file}")
        
        # 获取视频时长
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration <= 0:
            continue
        
        # 从视频中提取多个帧并生成问题
        for i in range(num_samples_per_video):
            # 在视频时长内选择时间戳
            timestamp = (i / num_samples_per_video) * duration
            frame = extract_frame(str(video_file), timestamp)
            
            if frame is not None:
                try:
                    # 使用CLIP分析帧
                    logger.debug(f"分析帧 {i+1}/{num_samples_per_video}")
                    frame_description = clip_analyzer.describe_frame(frame)
                    
                    # 使用DeepSeek生成问题
                    question = deepseek_client.generate_question(frame_description)
                    
                    # 评估相关性作为标签
                    relevance_score = deepseek_client.evaluate_relevance(frame_description, question)
                    label = 1 if relevance_score > 0.5 else 0
                    
                    data.append((frame, question, label))
                    
                    logger.info(f"生成数据 - 描述: {frame_description}, 问题: {question}, 标签: {label}")
                    
                    # 避免API调用过于频繁
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"处理帧时出错: {e}")
                    continue
                
        logger.info(f"从 {video_file} 生成了 {num_samples_per_video} 个训练样本")
    
    if not data:
        logger.warning("未能从视频中生成任何数据，使用模拟数据")
        return generate_simple_data()
    
    logger.info(f"总共生成了 {len(data)} 个训练样本")
    return data

def generate_simple_data(num_samples: int = 20) -> List[Tuple]:
    """生成简单的示例数据（回退选项）"""
    logger.info("生成模拟数据...")
    data = []
    for i in range(num_samples):
        # 创建随机帧（模拟视频帧）
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        question = f"这个画面显示了什么内容？{i}"
        label = i % 2  # 随机标签
        data.append((frame, question, label))
    return data

def create_client_fn(client_data_map: Dict[str, List[Tuple]]):
    """创建客户端函数"""
    def client_fn(cid: str):
        return SimpleFedClient(cid, client_data_map.get(cid, []))
    return client_fn

def main():
    """主函数 - 运行简化的联邦学习"""
    logger.info("=== 简化版联邦学习Qwen-VL微调启动 ===")
    
    # 从环境变量或参数获取DeepSeek API密钥
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        logger.warning("未设置DEEPSEEK_API_KEY环境变量，将使用模拟响应")
    
    # 使用CLIP和DeepSeek生成训练数据
    logger.info("使用CLIP和DeepSeek生成训练数据...")
    all_data = generate_data_from_videos(
        "data/videos", 
        num_samples_per_video=3,  # 减少样本数以避免API调用过多
        deepseek_api_key=deepseek_api_key
    )
    
    if not all_data:
        logger.error("无法生成任何训练数据")
        return
    
    # 分发数据到客户端
    num_clients = 3
    samples_per_client = len(all_data) // num_clients
    client_data_map = {}
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(all_data)
        client_data_map[f"client_{i}"] = all_data[start_idx:end_idx]
        logger.info(f"客户端 client_{i}: {len(client_data_map[f'client_{i}'])} 个样本")
    
    # 联邦学习配置
    config = fl.server.ServerConfig(num_rounds=3)  # 减少轮数以便快速测试
    
    # 启动联邦学习
    logger.info("启动联邦学习训练...")
    try:
        history = fl.simulation.start_simulation(
            client_fn=create_client_fn(client_data_map),
            num_clients=num_clients,
            config=config,
            strategy=fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_available_clients=num_clients
            ),
            client_resources={"num_cpus": 1, "num_gpus": 0.3 if torch.cuda.is_available() else 0}
        )
        
        logger.info("=== 联邦学习完成 ===")
        logger.info(f"训练轮数: {len(history.losses_distributed)}")
        
        if history.losses_distributed:
            final_loss = list(history.losses_distributed.values())[-1]
            logger.info(f"最终损失: {final_loss}")
            
    except Exception as e:
        logger.error(f"联邦学习失败: {e}")
        raise

if __name__ == "__main__":
    main()