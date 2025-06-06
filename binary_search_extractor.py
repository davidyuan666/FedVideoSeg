"""
基于二分法搜索的视频关键帧提取器
使用CLIP和DeepSeek进行智能帧选择
"""

import torch
import cv2
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

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
    
    def evaluate_relevance(self, frame_description: str, question: str) -> float:
        """评估帧描述与问题的相关性 (0-1)"""
        if not self.api_key:
            # 模拟相关性评分
            keywords = question.lower().split()
            desc_lower = frame_description.lower()
            matches = sum(1 for word in keywords if word in desc_lower)
            return min(0.9, matches / len(keywords) * 0.8 + 0.1)
        
        prompt = f"""
        请评估以下视频帧描述与问题的相关性，返回0-1之间的分数。
        1.0表示完全相关，0.0表示完全不相关。
        
        问题：{question}
        帧描述：{frame_description}
        
        请只返回一个0-1之间的数字：
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
                logger.error(f"DeepSeek API错误: {response.status_code}")
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
        
        # 预定义的描述模板
        self.descriptions = [
            "一个人在说话或演讲",
            "多个人在交谈或讨论", 
            "会议或正式讨论场景",
            "演示或展示内容",
            "教学或培训场景",
            "工作或办公环境",
            "室内正式场合",
            "户外活动场景",
            "技术或产品展示",
            "问答或互动环节",
            "文档或屏幕内容",
            "图表或数据展示"
        ]
        
        logger.info(f"CLIP模型已加载: {model_name}")
    
    def describe_frame(self, frame: np.ndarray) -> str:
        """使用CLIP生成帧的描述"""
        try:
            # 转换为PIL图像
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            # 处理图像和文本
            inputs = self.processor(
                text=self.descriptions,
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
            
            best_description = self.descriptions[best_idx]
            
            # 如果置信度较低，使用通用描述
            if confidence < 0.2:
                best_description = "视频场景内容"
            
            logger.debug(f"帧描述: {best_description} (置信度: {confidence:.3f})")
            return best_description
            
        except Exception as e:
            logger.error(f"CLIP分析帧失败: {e}")
            return "视频场景内容"

class BinarySearchExtractor:
    """二分法视频关键帧提取器"""
    
    def __init__(self, deepseek_api_key: str = None):
        self.clip_analyzer = CLIPFrameAnalyzer()
        self.deepseek_client = DeepSeekClient(deepseek_api_key)
        self.min_segment_duration = 2.0  # 最小片段时长(秒)
        self.max_frames = 5  # 最大提取帧数
        
        # 预定义问题
        self.predefined_questions = [
            "视频中的主要讨论内容是什么？",
            "视频中出现了哪些重要信息？", 
            "视频的核心观点或结论是什么？",
            "视频中有哪些关键的演示或展示？",
            "视频中的重点时刻在哪里？"
        ]
    
    def extract_frame_at_time(self, video_path: str, timestamp: float) -> np.ndarray:
        """在指定时间提取视频帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            raise ValueError(f"无法在时间戳 {timestamp} 提取帧")
    
    def get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return duration
    
    def evaluate_segment_relevance(self, video_path: str, start_time: float, 
                                 end_time: float, question: str) -> float:
        """评估视频片段与问题的相关性"""
        try:
            # 提取片段首尾帧
            start_frame = self.extract_frame_at_time(video_path, start_time)
            end_frame = self.extract_frame_at_time(video_path, end_time)
            
            # 获取帧描述
            start_desc = self.clip_analyzer.describe_frame(start_frame)
            end_desc = self.clip_analyzer.describe_frame(end_frame)
            
            # 组合描述
            combined_desc = f"片段开始: {start_desc}, 片段结束: {end_desc}"
            
            # 评估相关性
            relevance = self.deepseek_client.evaluate_relevance(combined_desc, question)
            
            logger.debug(f"片段 [{start_time:.1f}-{end_time:.1f}s] 相关性: {relevance:.3f}")
            return relevance
            
        except Exception as e:
            logger.error(f"评估片段相关性失败: {e}")
            return 0.0
    
    def binary_search_segments(self, video_path: str, question: str, 
                             start_time: float = 0, end_time: float = None) -> List[Tuple[float, float, float]]:
        """使用二分法搜索相关片段"""
        if end_time is None:
            end_time = self.get_video_duration(video_path)
        
        segments = []  # (start_time, end_time, relevance_score)
        search_queue = [(start_time, end_time)]
        
        while search_queue and len(segments) < self.max_frames:
            current_start, current_end = search_queue.pop(0)
            
            # 如果片段太短，跳过
            if current_end - current_start < self.min_segment_duration:
                continue
            
            # 评估当前片段的相关性
            relevance = self.evaluate_segment_relevance(
                video_path, current_start, current_end, question
            )
            
            logger.info(f"评估片段 [{current_start:.1f}-{current_end:.1f}s]: 相关性 {relevance:.3f}")
            
            # 如果相关性高，记录这个片段
            if relevance > 0.6:
                segments.append((current_start, current_end, relevance))
                logger.info(f"✓ 找到相关片段: [{current_start:.1f}-{current_end:.1f}s]")
            
            # 如果相关性中等，继续二分搜索
            elif relevance > 0.3:
                mid_time = (current_start + current_end) / 2
                
                # 将片段分为两半继续搜索
                search_queue.append((current_start, mid_time))
                search_queue.append((mid_time, current_end))
                
                logger.debug(f"继续搜索子片段: [{current_start:.1f}-{mid_time:.1f}s] 和 [{mid_time:.1f}-{current_end:.1f}s]")
            
            # 避免API调用过于频繁
            time.sleep(0.5)
        
        # 按相关性排序并返回最好的片段
        segments.sort(key=lambda x: x[2], reverse=True)
        return segments[:self.max_frames]
    
    def extract_key_frames(self, video_path: str, question: str, output_dir: str) -> List[Dict]:
        """提取关键帧并保存"""
        logger.info(f"开始处理视频: {video_path}")
        logger.info(f"问题: {question}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 搜索相关片段
        relevant_segments = self.binary_search_segments(video_path, question)
        
        if not relevant_segments:
            logger.warning("未找到相关片段")
            return []
        
        # 提取关键帧
        extracted_frames = []
        
        for i, (start_time, end_time, relevance) in enumerate(relevant_segments):
            # 选择片段中间的时间点作为关键帧
            key_timestamp = (start_time + end_time) / 2
            
            try:
                # 提取帧
                frame = self.extract_frame_at_time(video_path, key_timestamp)
                
                # 保存帧
                frame_filename = f"frame_{i+1}_{key_timestamp:.1f}s.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # 转换为PIL图像并保存
                pil_image = Image.fromarray(frame)
                pil_image.save(frame_path, "JPEG", quality=95)
                
                # 获取帧描述
                frame_desc = self.clip_analyzer.describe_frame(frame)
                
                frame_info = {
                    "frame_id": i + 1,
                    "timestamp": key_timestamp,
                    "segment_start": start_time,
                    "segment_end": end_time,
                    "relevance_score": relevance,
                    "description": frame_desc,
                    "filename": frame_filename,
                    "file_path": frame_path
                }
                
                extracted_frames.append(frame_info)
                
                logger.info(f"✓ 提取关键帧 {i+1}: {key_timestamp:.1f}s (相关性: {relevance:.3f})")
                
            except Exception as e:
                logger.error(f"提取帧失败: {e}")
                continue
        
        return extracted_frames
    
    def process_video_with_questions(self, video_path: str, output_base_dir: str) -> Dict:
        """使用所有预定义问题处理视频"""
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(output_base_dir, video_name)
        
        logger.info(f"=" * 60)
        logger.info(f"处理视频: {video_name}")
        logger.info(f"输出目录: {video_output_dir}")
        
        results = {
            "video_name": video_name,
            "video_path": video_path,
            "output_directory": video_output_dir,
            "questions_results": []
        }
        
        for i, question in enumerate(self.predefined_questions):
            logger.info(f"\n--- 问题 {i+1}/{len(self.predefined_questions)} ---")
            
            # 为每个问题创建子目录
            question_dir = os.path.join(video_output_dir, f"question_{i+1}")
            
            # 提取关键帧
            extracted_frames = self.extract_key_frames(video_path, question, question_dir)
            
            question_result = {
                "question_id": i + 1,
                "question": question,
                "output_directory": question_dir,
                "extracted_frames": extracted_frames,
                "frame_count": len(extracted_frames)
            }
            
            results["questions_results"].append(question_result)
            
            logger.info(f"问题 {i+1} 完成，提取了 {len(extracted_frames)} 个关键帧")
        
        # 保存结果到JSON文件
        results_file = os.path.join(video_output_dir, "extraction_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到: {results_file}")
        return results

def main():
    """主函数"""
    logger.info("=== 二分法视频关键帧提取器 ===")
    
    # 配置路径
    video_dir = "dataset/video"
    output_dir = "extracted_frames"
    
    # 检查视频目录
    if not os.path.exists(video_dir):
        logger.error(f"视频目录不存在: {video_dir}")
        logger.info("请创建目录并放入视频文件:")
        logger.info(f"mkdir -p {video_dir}")
        return
    
    # 获取DeepSeek API密钥
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        logger.warning("未设置DEEPSEEK_API_KEY环境变量，将使用模拟响应")
    
    # 初始化提取器
    extractor = BinarySearchExtractor(deepseek_api_key)
    
    # 查找视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
        video_files.extend(Path(video_dir).glob(f"*{ext.upper()}"))
    
    if not video_files:
        logger.error(f"在 {video_dir} 中未找到视频文件")
        logger.info(f"支持的格式: {', '.join(video_extensions)}")
        return
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    all_results = []
    
    for video_file in video_files:
        try:
            result = extractor.process_video_with_questions(str(video_file), output_dir)
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"处理视频 {video_file} 失败: {e}")
            continue
    
    # 保存总体结果
    summary_file = os.path.join(output_dir, "extraction_summary.json")
    summary = {
        "total_videos": len(video_files),
        "processed_videos": len(all_results),
        "predefined_questions": extractor.predefined_questions,
        "results": all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 60)
    logger.info("处理完成!")
    logger.info(f"总视频数: {len(video_files)}")
    logger.info(f"成功处理: {len(all_results)}")
    logger.info(f"结果摘要: {summary_file}")
    
    # 打印提取统计
    total_frames = sum(
        sum(q["frame_count"] for q in result["questions_results"]) 
        for result in all_results
    )
    logger.info(f"总提取帧数: {total_frames}")

if __name__ == "__main__":
    main() 