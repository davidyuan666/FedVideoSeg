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
    """使用CLIP和图像描述生成模型分析视频帧"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name, cache_dir='cache')
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir='cache')
        self.model.to(self.device)
        
        # 初始化图像描述生成模型
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir='cache')
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir='cache')
            self.caption_model.to(self.device)
            self.use_blip = True
            logger.info("BLIP图像描述模型已加载")
        except ImportError:
            logger.warning("BLIP模型不可用，将使用增强的CLIP分析")
            self.use_blip = False
        
        logger.info(f"CLIP模型已加载: {model_name}")
    
    def describe_frame_with_blip(self, frame: np.ndarray) -> str:
        """使用BLIP模型生成帧的开放式描述"""
        try:
            # 转换为PIL图像
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            # 使用BLIP生成描述
            inputs = self.caption_processor(frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            logger.debug(f"BLIP帧描述: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"BLIP分析帧失败: {e}")
            return self._fallback_describe(frame)
    
    def _fallback_describe(self, frame: np.ndarray) -> str:
        """增强的CLIP描述方法：使用更详细的场景分析"""
        try:
            # 转换为PIL图像
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            # 使用更详细和具体的描述模板
            detailed_descriptions = [
                "a person cooking food in a kitchen with pots and pans",
                "people eating a meal together at a dining table", 
                "someone talking on a mobile phone or device",
                "people having a conversation in a living room",
                "a person working at a desk with a computer",
                "people watching television in a comfortable room",
                "someone cleaning or organizing household items",
                "people exercising or doing physical activities",
                "a person reading a book or studying materials",
                "people playing games or socializing together",
                "outdoor activities in a natural environment",
                "street scenes with cars and pedestrians",
                "shopping activities in a store or market",
                "business meeting or educational discussion",
                "family gathering or celebration event",
                "kitchen activities with food preparation",
                "bedroom or private room activities",
                "bathroom or personal care activities",
                "garden or outdoor home activities",
                "workshop or garage activities"
            ]
            
            # 处理图像和文本
            inputs = self.processor(
                text=detailed_descriptions,
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
            
            # 选择置信度最高的描述
            best_idx = probs.argmax().item()
            confidence = probs[0][best_idx].item()
            
            best_description = detailed_descriptions[best_idx]
            
            # 如果置信度较低，尝试组合描述
            if confidence < 0.15:
                # 获取前3个候选并组合
                top_k = 3
                top_indices = probs[0].topk(top_k).indices
                top_probs = probs[0].topk(top_k).values
                
                description_parts = []
                for idx, prob in zip(top_indices, top_probs):
                    if prob > 0.08:  # 较低的阈值
                        desc = detailed_descriptions[idx]
                        # 提取关键词
                        key_words = desc.split()[:3]  # 取前3个词作为关键描述
                        description_parts.extend(key_words)
                
                if description_parts:
                    best_description = " ".join(description_parts[:5])  # 组合前5个关键词
                else:
                    best_description = "indoor daily life scene"
            
            logger.debug(f"增强CLIP帧描述: {best_description} (置信度: {confidence:.3f})")
            return best_description
                
        except Exception as e:
            logger.error(f"增强CLIP分析帧失败: {e}")
            return "video scene content"
    
    def describe_frame(self, frame: np.ndarray, deepseek_client=None) -> str:
        """生成帧的开放式描述"""
        # 优先使用BLIP模型进行开放式描述
        if self.use_blip:
            return self.describe_frame_with_blip(frame)
        # 使用增强的CLIP分析
        else:
            return self._fallback_describe(frame)

class BinarySearchExtractor:
    """二分法视频关键帧提取器"""
    
    def __init__(self, deepseek_api_key: str = None):
        self.clip_analyzer = CLIPFrameAnalyzer()
        self.deepseek_client = DeepSeekClient(deepseek_api_key)
        self.min_segment_duration = 2.0  # 最小片段时长(秒)
        self.max_frames = 5  # 最大提取帧数
        self.initial_segment_duration = 5.0  # 初步筛选的片段时长(秒)
        self.top_segments_for_binary_search = 3  # 选择前N个片段进行二分法搜索
        
        # 预定义问题
        self.predefined_questions = [
            "when does the man boil milk",
            "when does the man make cake"
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
            logger.info(f"   🖼️  提取片段首尾帧: [{start_time:.1f}s] 和 [{end_time:.1f}s]")
            
            # 提取片段首尾帧
            start_frame = self.extract_frame_at_time(video_path, start_time)
            end_frame = self.extract_frame_at_time(video_path, end_time)
            
            # 获取帧描述（传入deepseek_client以支持更好的描述）
            start_desc = self.clip_analyzer.describe_frame(start_frame, self.deepseek_client)
            end_desc = self.clip_analyzer.describe_frame(end_frame, self.deepseek_client)
            
            logger.info(f"   📝 帧描述 - 开始帧: {start_desc}")
            logger.info(f"   📝 帧描述 - 结束帧: {end_desc}")
            
            # 组合描述
            combined_desc = f"片段开始: {start_desc}, 片段结束: {end_desc}"
            
            # 评估相关性
            logger.info(f"   🤖 正在调用DeepSeek评估相关性...")
            relevance = self.deepseek_client.evaluate_relevance(combined_desc, question)
            
            logger.info(f"   ⭐ DeepSeek评估结果: {relevance:.3f}")
            
            return relevance
            
        except Exception as e:
            logger.error(f"   ❌ 评估片段相关性失败: {e}")
            return 0.0
    
    def initial_coarse_screening(self, video_path: str, question: str) -> List[Tuple[float, float, float]]:
        """初步粗筛选：每隔5秒创建片段并评估相关性"""
        duration = self.get_video_duration(video_path)
        segments = []
        
        logger.info(f"🔍 开始初步粗筛选 - 视频总时长: {duration:.1f}s")
        logger.info(f"📏 片段间隔: {self.initial_segment_duration}s")
        logger.info(f"📋 问题: {question}")
        
        # 创建每隔5秒的片段
        current_time = 0
        segment_id = 0
        
        while current_time < duration:
            end_time = min(current_time + self.initial_segment_duration, duration)
            
            # 确保片段长度足够
            if end_time - current_time >= self.min_segment_duration:
                segment_id += 1
                
                logger.info(f"🎯 [粗筛 {segment_id}] 评估片段 [{current_time:.1f}-{end_time:.1f}s] (时长: {end_time-current_time:.1f}s)")
                
                # 评估片段相关性
                relevance = self.evaluate_segment_relevance(video_path, current_time, end_time, question)
                
                segments.append((current_time, end_time, relevance))
                
                logger.info(f"📊 [粗筛 {segment_id}] 片段 [{current_time:.1f}-{end_time:.1f}s] 相关性分数: {relevance:.3f}")
                
                # 避免API调用过于频繁
                time.sleep(0.3)
            
            current_time += self.initial_segment_duration
        
        # 按相关性排序
        segments.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"🏁 粗筛选完成，共评估 {len(segments)} 个片段")
        
        # 显示所有片段的相关性分数
        if segments:
            logger.info(f"📋 所有片段相关性排序:")
            for i, (start, end, score) in enumerate(segments):
                logger.info(f"   {i+1}. [{start:.1f}-{end:.1f}s] 分数: {score:.3f}")
        
        # 选择相关性最高的前几个片段进行二分法搜索
        top_segments = segments[:self.top_segments_for_binary_search]
        
        if top_segments:
            logger.info(f"🎯 选择前 {len(top_segments)} 个片段进行二分法细化搜索:")
            for i, (start, end, score) in enumerate(top_segments):
                logger.info(f"   选中 {i+1}. [{start:.1f}-{end:.1f}s] 分数: {score:.3f}")
        
        return top_segments

    def binary_search_on_segment(self, video_path: str, question: str, 
                                start_time: float, end_time: float) -> List[Tuple[float, float, float]]:
        """对指定片段进行二分法搜索"""
        segments = []  # (start_time, end_time, relevance_score)
        search_queue = [(start_time, end_time)]
        search_depth = 0
        
        logger.info(f"🔬 对片段 [{start_time:.1f}-{end_time:.1f}s] 开始二分法细化搜索")
        
        while search_queue and len(segments) < self.max_frames:
            current_start, current_end = search_queue.pop(0)
            search_depth += 1
            
            # 如果片段太短，跳过
            if current_end - current_start < self.min_segment_duration:
                logger.info(f"⏭️  [二分 {search_depth}] 片段 [{current_start:.1f}-{current_end:.1f}s] 太短，跳过 (< {self.min_segment_duration}s)")
                continue
            
            logger.info(f"🎯 [二分 {search_depth}] 评估片段 [{current_start:.1f}-{current_end:.1f}s] (时长: {current_end-current_start:.1f}s)")
            
            relevance = self.evaluate_segment_relevance(
                video_path, current_start, current_end, question
            )
            
            logger.info(f"📊 [二分 {search_depth}] 片段 [{current_start:.1f}-{current_end:.1f}s] 相关性分数: {relevance:.3f}")
            
            # 如果相关性高，记录这个片段
            if relevance > 0.6:
                segments.append((current_start, current_end, relevance))
                logger.info(f"✅ [二分 {search_depth}] 高相关性片段已保存: [{current_start:.1f}-{current_end:.1f}s] (分数: {relevance:.3f})")
            
            # 如果相关性中等，继续二分搜索
            elif relevance > 0.3:
                mid_time = (current_start + current_end) / 2
                
                # 将片段分为两半继续搜索
                search_queue.append((current_start, mid_time))
                search_queue.append((mid_time, current_end))
                
                logger.info(f"🔄 [二分 {search_depth}] 中等相关性，继续二分搜索:")
                logger.info(f"   ├─ 左半段: [{current_start:.1f}-{mid_time:.1f}s] (时长: {mid_time-current_start:.1f}s)")
                logger.info(f"   └─ 右半段: [{mid_time:.1f}-{current_end:.1f}s] (时长: {current_end-mid_time:.1f}s)")
            
            else:
                logger.info(f"❌ [二分 {search_depth}] 低相关性片段，停止搜索: [{current_start:.1f}-{current_end:.1f}s] (分数: {relevance:.3f})")
            
            # 避免API调用过于频繁
            time.sleep(0.5)
        
        return segments

    def binary_search_segments(self, video_path: str, question: str, 
                             start_time: float = 0, end_time: float = None) -> List[Tuple[float, float, float]]:
        """改进的二分法搜索：先粗筛选，再对选中的片段进行二分法"""
        logger.info(f"🚀 开始改进的二分法搜索流程")
        
        # 第一步：初步粗筛选
        top_segments = self.initial_coarse_screening(video_path, question)
        
        if not top_segments:
            logger.warning("粗筛选未找到相关片段")
            return []
        
        # 第二步：对选中的片段进行二分法细化搜索
        all_refined_segments = []
        
        for i, (seg_start, seg_end, initial_score) in enumerate(top_segments):
            logger.info(f"\n--- 二分法细化搜索 {i+1}/{len(top_segments)} ---")
            logger.info(f"🎯 目标片段: [{seg_start:.1f}-{seg_end:.1f}s] (初始分数: {initial_score:.3f})")
            
            refined_segments = self.binary_search_on_segment(
                video_path, question, seg_start, seg_end
            )
            
            # 如果二分法没有找到更好的片段，使用原始片段
            if not refined_segments and initial_score > 0.4:
                refined_segments = [(seg_start, seg_end, initial_score)]
                logger.info(f"🔄 二分法未找到更好片段，保留原始片段: [{seg_start:.1f}-{seg_end:.1f}s] (分数: {initial_score:.3f})")
            
            all_refined_segments.extend(refined_segments)
            
            logger.info(f"✓ 片段 {i+1} 细化完成，找到 {len(refined_segments)} 个精确片段")
        
        # 第三步：合并结果并排序
        all_refined_segments.sort(key=lambda x: x[2], reverse=True)
        
        # 去重：如果有重叠的片段，选择分数更高的
        final_segments = []
        for seg in all_refined_segments:
            if len(final_segments) >= self.max_frames:
                break
            
            # 检查是否与已有片段重叠
            overlap = False
            for existing_seg in final_segments:
                if (seg[0] < existing_seg[1] and seg[1] > existing_seg[0]):  # 有重叠
                    overlap = True
                    break
            
            if not overlap:
                final_segments.append(seg)
        
        logger.info(f"🏁 改进的二分法搜索完成，最终选择 {len(final_segments)} 个片段")
        
        # 显示最终结果
        if final_segments:
            logger.info(f"📋 最终选择的片段 (按相关性排序):")
            for i, (start, end, score) in enumerate(final_segments):
                logger.info(f"   {i+1}. [{start:.1f}-{end:.1f}s] 分数: {score:.3f}")
        
        return final_segments
    
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
                frame_desc = self.clip_analyzer.describe_frame(frame, self.deepseek_client)
                
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