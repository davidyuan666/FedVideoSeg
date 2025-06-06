"""
VA-pair数据集生成器
专门用于从questions.json和提取的关键帧生成适合训练QwenVL模型的数据集
"""

import os
import json
import logging
import random
from typing import List, Dict, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VAPairGenerator:
    """VA-pair数据集生成器"""
    
    def __init__(self):
        self.logger = logger
    
    def load_questions(self, questions_file: str) -> List[Dict]:
        """加载问题文件"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            self.logger.info(f"✅ 成功加载 {len(questions)} 个问题")
            return questions
        except Exception as e:
            self.logger.error(f"❌ 加载问题文件失败: {e}")
            return []
    
    def scan_available_frames(self, output_dir: str) -> Dict[str, List[Dict]]:
        """扫描输出目录，收集所有可用的图片帧"""
        all_frames = {}  # video_id -> list of frame info
        
        if not os.path.exists(output_dir):
            self.logger.error(f"❌ 输出目录不存在: {output_dir}")
            return all_frames
        
        # 扫描output目录下的所有子目录
        for item in os.listdir(output_dir):
            video_dir = os.path.join(output_dir, item)
            if os.path.isdir(video_dir):
                video_id = item
                valid_frames = []
                
                # 扫描视频目录下的所有图片文件
                try:
                    for filename in os.listdir(video_dir):
                        file_path = os.path.join(video_dir, filename)
                        # 检查是否是图片文件
                        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            valid_frames.append({
                                "path": file_path,
                                "filename": filename
                            })
                except Exception as e:
                    self.logger.warning(f"⚠️  扫描目录 {video_dir} 失败: {e}")
                    continue
                
                if valid_frames:
                    all_frames[video_id] = valid_frames
                    self.logger.info(f"📁 视频{video_id}: 发现 {len(valid_frames)} 个有效图片")
                else:
                    self.logger.info(f"📁 视频{video_id}: 没有发现图片文件，跳过")
        
        total_videos = len(all_frames)
        total_frames = sum(len(frames) for frames in all_frames.values())
        self.logger.info(f"📊 总计扫描到 {total_videos} 个视频，{total_frames} 个图片文件")
        
        return all_frames
    
    def create_video_question_pairs(self, questions: List[Dict], all_frames: Dict[str, List[Dict]]) -> List[Dict]:
        """根据问题和可用视频创建视频-问题对"""
        vq_pairs = []
        
        for question_data in questions:
            video_id = str(question_data["video_id"])
            question = question_data["question"]
            
            # 检查该视频是否有可用的图片
            if video_id in all_frames and all_frames[video_id]:
                vq_pairs.append({
                    "video_id": video_id,
                    "question": question,
                    "frames": [frame["filename"] for frame in all_frames[video_id]]
                })
                self.logger.info(f"✅ 创建VQ对: 视频{video_id}, 问题: {question[:50]}...")
            else:
                self.logger.warning(f"⚠️  跳过视频{video_id}: 没有可用图片")
        
        self.logger.info(f"📋 总计创建 {len(vq_pairs)} 个视频-问题对")
        return vq_pairs
    
    def generate_va_pair_dataset(self, output_dir: str, questions_file: str, 
                               va_pair_file: str = None) -> List[Dict]:
        """
        生成基础VA-pair数据集
        """
        if va_pair_file is None:
            va_pair_file = os.path.join(output_dir, "va_pair.json")
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始生成VA-pair训练数据集")
        
        # 加载问题文件
        questions = self.load_questions(questions_file)
        if not questions:
            return []
        
        # 扫描可用图片
        all_frames = self.scan_available_frames(output_dir)
        if not all_frames:
            self.logger.error("❌ 没有找到任何可用的图片文件")
            return []
        
        # 创建视频-问题对
        vq_pairs = self.create_video_question_pairs(questions, all_frames)
        if not vq_pairs:
            self.logger.error("❌ 没有创建任何视频-问题对")
            return []
        
        # 构建VA-pair数据集
        va_pairs = []
        positive_samples = 0
        negative_samples = 0
        
        self.logger.info(f"📋 开始处理 {len(vq_pairs)} 个视频-问题对")
        
        for vq_pair in vq_pairs:
            video_id = vq_pair["video_id"]
            question = vq_pair["question"]
            
            if video_id not in all_frames or not all_frames[video_id]:
                self.logger.warning(f"⚠️  视频{video_id}没有有效图片，跳过")
                continue
            
            self.logger.info(f"🎯 处理视频ID: {video_id}")
            self.logger.info(f"   ❓ 问题: {question}")
            
            # 正样本：当前视频的相关帧
            current_video_frames = all_frames[video_id]
            for frame_info in current_video_frames:
                va_pair = {
                    "image_path": frame_info["path"],
                    "question": question,
                    "answer": "是",  # 相关帧的答案是"是"
                    "label": 1,     # 正样本
                    "video_id": video_id,
                    "frame_filename": frame_info["filename"],
                    "sample_type": "positive"
                }
                va_pairs.append(va_pair)
                positive_samples += 1
            
            self.logger.info(f"   ✅ 添加了 {len(current_video_frames)} 个正样本")
            
            # 负样本：其他视频的帧
            negative_frames = []
            for other_video_id, other_frames in all_frames.items():
                if other_video_id != video_id and other_frames:
                    for frame_info in other_frames:
                        negative_frames.append({
                            "path": frame_info["path"],
                            "filename": frame_info["filename"],
                            "source_video_id": other_video_id
                        })
            
            # 随机选择负样本，数量与正样本相等
            if negative_frames:
                negative_count = min(len(current_video_frames), len(negative_frames))
                selected_negatives = random.sample(negative_frames, negative_count)
                
                for neg_frame in selected_negatives:
                    va_pair = {
                        "image_path": neg_frame["path"],
                        "question": question,
                        "answer": "否",  # 无关帧的答案是"否"
                        "label": 0,     # 负样本
                        "video_id": video_id,
                        "frame_filename": neg_frame["filename"],
                        "source_video_id": neg_frame["source_video_id"],
                        "sample_type": "negative"
                    }
                    va_pairs.append(va_pair)
                    negative_samples += 1
                
                self.logger.info(f"   ❌ 添加了 {negative_count} 个负样本")
            
            self.logger.info(f"   📊 视频{video_id}完成")
        
        # 打乱数据集
        random.shuffle(va_pairs)
        
        # 保存数据集
        self._save_dataset(va_pairs, va_pair_file, positive_samples, negative_samples)
        
        return va_pairs
    
    def generate_balanced_va_pair_dataset(self, output_dir: str, questions_file: str,
                                        va_pair_file: str = None, 
                                        max_samples_per_video: int = 10,
                                        split_ratio: float = 0.8) -> List[Dict]:
        """
        生成平衡的VA-pair数据集
        确保正负样本数量平衡，自动分割训练集和验证集
        """
        if va_pair_file is None:
            va_pair_file = os.path.join(output_dir, "va_pair_balanced.json")
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始生成平衡的VA-pair训练数据集")
        
        # 加载问题文件
        questions = self.load_questions(questions_file)
        if not questions:
            return []
        
        # 扫描可用图片
        all_frames = self.scan_available_frames(output_dir)
        if not all_frames:
            self.logger.error("❌ 没有找到任何可用的图片文件")
            return []
        
        # 创建视频-问题对
        vq_pairs = self.create_video_question_pairs(questions, all_frames)
        if not vq_pairs:
            self.logger.error("❌ 没有创建任何视频-问题对")
            return []
        
        # 构建训练样本
        va_pairs = []
        positive_samples = 0
        negative_samples = 0
        
        for vq_pair in vq_pairs:
            video_id = vq_pair["video_id"]
            question = vq_pair["question"]
            
            if video_id not in all_frames or not all_frames[video_id]:
                self.logger.warning(f"⚠️  视频{video_id}没有有效图片，跳过")
                continue
            
            self.logger.info(f"🎯 处理视频ID: {video_id}")
            self.logger.info(f"   ❓ 问题: {question}")
            
            # 正样本：本视频的相关帧
            current_video_frames = all_frames[video_id]
            positive_count = min(len(current_video_frames), max_samples_per_video // 2)
            
            # 如果图片太多，随机选择
            if len(current_video_frames) > positive_count:
                selected_frames = random.sample(current_video_frames, positive_count)
            else:
                selected_frames = current_video_frames
            
            for frame_info in selected_frames:
                va_pair = {
                    "image_path": frame_info["path"],
                    "question": question,
                    "answer": "是",
                    "label": 1,
                    "video_id": video_id,
                    "frame_filename": frame_info["filename"],
                    "sample_type": "positive"
                }
                va_pairs.append(va_pair)
                positive_samples += 1
            
            self.logger.info(f"   ✅ 添加了 {positive_count} 个正样本")
            
            # 负样本：其他视频的帧
            negative_frames = []
            for other_video_id, other_frames in all_frames.items():
                if other_video_id != video_id and other_frames:
                    for frame_info in other_frames:
                        negative_frames.append({
                            "path": frame_info["path"],
                            "filename": frame_info["filename"],
                            "source_video_id": other_video_id
                        })
            
            # 随机选择负样本，数量与正样本相等
            if negative_frames:
                negative_count = min(positive_count, len(negative_frames))
                selected_negatives = random.sample(negative_frames, negative_count)
                
                for neg_frame in selected_negatives:
                    va_pair = {
                        "image_path": neg_frame["path"],
                        "question": question,
                        "answer": "否",
                        "label": 0,
                        "video_id": video_id,
                        "frame_filename": neg_frame["filename"],
                        "source_video_id": neg_frame["source_video_id"],
                        "sample_type": "negative"
                    }
                    va_pairs.append(va_pair)
                    negative_samples += 1
                
                self.logger.info(f"   ❌ 添加了 {negative_count} 个负样本")
            
            self.logger.info(f"   📊 视频{video_id}完成，总样本: {positive_count + negative_count}")
        
        # 打乱数据集
        random.shuffle(va_pairs)
        
        # 分割数据集为训练集和验证集
        split_index = int(len(va_pairs) * split_ratio)
        train_data = va_pairs[:split_index]
        val_data = va_pairs[split_index:]
        
        # 保存数据集
        self._save_split_dataset(va_pairs, train_data, val_data, va_pair_file, 
                                positive_samples, negative_samples)
        
        return va_pairs
    
    def generate_cross_video_negative_dataset(self, output_dir: str, questions_file: str,
                                            va_pair_file: str = None,
                                            negative_ratio: float = 1.0) -> List[Dict]:
        """
        生成跨视频负样本数据集
        为每个问题生成来自其他视频的负样本，提高模型的判别能力
        """
        if va_pair_file is None:
            va_pair_file = os.path.join(output_dir, "va_pair_cross_negative.json")
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始生成跨视频负样本VA-pair数据集")
        
        # 加载数据
        questions = self.load_questions(questions_file)
        if not questions:
            return []
        
        # 扫描可用图片
        all_frames = self.scan_available_frames(output_dir)
        if not all_frames:
            self.logger.error("❌ 没有找到任何可用的图片文件")
            return []
        
        # 创建视频-问题对
        vq_pairs = self.create_video_question_pairs(questions, all_frames)
        if not vq_pairs:
            self.logger.error("❌ 没有创建任何视频-问题对")
            return []
        
        va_pairs = []
        positive_samples = 0
        negative_samples = 0
        
        # 创建问题到视频的映射
        question_to_videos = {}
        for q in questions:
            question_text = q["question"]
            if question_text not in question_to_videos:
                question_to_videos[question_text] = []
            question_to_videos[question_text].append(str(q["video_id"]))
        
        for vq_pair in vq_pairs:
            video_id = vq_pair["video_id"]
            question = vq_pair["question"]
            
            if video_id not in all_frames or not all_frames[video_id]:
                continue
            
            self.logger.info(f"🎯 处理视频ID: {video_id}, 问题: {question}")
            
            # 正样本：当前视频的帧
            current_frames = all_frames[video_id]
            for frame_info in current_frames:
                va_pair = {
                    "image_path": frame_info["path"],
                    "question": question,
                    "answer": "是",
                    "label": 1,
                    "video_id": video_id,
                    "frame_filename": frame_info["filename"],
                    "sample_type": "positive"
                }
                va_pairs.append(va_pair)
                positive_samples += 1
            
            # 负样本1：同一问题的其他视频的帧（更难的负样本）
            same_question_videos = question_to_videos.get(question, [])
            same_question_negatives = []
            for other_video_id in same_question_videos:
                if other_video_id != video_id and other_video_id in all_frames:
                    for frame_info in all_frames[other_video_id]:
                        same_question_negatives.append({
                            "path": frame_info["path"],
                            "filename": frame_info["filename"],
                            "source_video_id": other_video_id,
                            "negative_type": "same_question"
                        })
            
            # 负样本2：不同问题的视频帧（更容易的负样本）
            different_question_negatives = []
            for other_video_id, other_frames in all_frames.items():
                if other_video_id != video_id and other_video_id not in same_question_videos:
                    for frame_info in other_frames:
                        different_question_negatives.append({
                            "path": frame_info["path"],
                            "filename": frame_info["filename"],
                            "source_video_id": other_video_id,
                            "negative_type": "different_question"
                        })
            
            # 按比例选择负样本
            total_positive = len(current_frames)
            total_negative_needed = int(total_positive * negative_ratio)
            
            # 30%来自同问题其他视频，70%来自不同问题视频
            same_q_needed = int(total_negative_needed * 0.3)
            diff_q_needed = total_negative_needed - same_q_needed
            
            selected_negatives = []
            
            # 选择同问题负样本
            if same_question_negatives and same_q_needed > 0:
                selected_same_q = random.sample(
                    same_question_negatives, 
                    min(same_q_needed, len(same_question_negatives))
                )
                selected_negatives.extend(selected_same_q)
            
            # 选择不同问题负样本
            if different_question_negatives and diff_q_needed > 0:
                selected_diff_q = random.sample(
                    different_question_negatives,
                    min(diff_q_needed, len(different_question_negatives))
                )
                selected_negatives.extend(selected_diff_q)
            
            # 添加负样本
            for neg_frame in selected_negatives:
                va_pair = {
                    "image_path": neg_frame["path"],
                    "question": question,
                    "answer": "否",
                    "label": 0,
                    "video_id": video_id,
                    "frame_filename": neg_frame["filename"],
                    "source_video_id": neg_frame["source_video_id"],
                    "sample_type": "negative",
                    "negative_type": neg_frame["negative_type"]
                }
                va_pairs.append(va_pair)
                negative_samples += 1
            
            self.logger.info(f"   ✅ {total_positive}正样本, {len(selected_negatives)}负样本")
        
        # 打乱并保存
        random.shuffle(va_pairs)
        self._save_dataset(va_pairs, va_pair_file, positive_samples, negative_samples)
        
        return va_pairs
    
    def _save_dataset(self, va_pairs: List[Dict], va_pair_file: str, 
                     positive_samples: int, negative_samples: int):
        """保存单个数据集文件"""
        try:
            with open(va_pair_file, 'w', encoding='utf-8') as f:
                json.dump(va_pairs, f, indent=2, ensure_ascii=False)
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 VA-pair数据集生成完成!")
            self.logger.info(f"📊 数据集统计:")
            self.logger.info(f"   - 总样本数: {len(va_pairs)}")
            self.logger.info(f"   - 正样本数: {positive_samples}")
            self.logger.info(f"   - 负样本数: {negative_samples}")
            self.logger.info(f"   - 正负比例: {positive_samples}:{negative_samples}")
            self.logger.info(f"📁 数据集文件: {va_pair_file}")
            
            self._show_samples(va_pairs)
            
        except Exception as e:
            self.logger.error(f"❌ 保存数据集失败: {e}")
    
    def _save_split_dataset(self, va_pairs: List[Dict], train_data: List[Dict], 
                           val_data: List[Dict], va_pair_file: str,
                           positive_samples: int, negative_samples: int):
        """保存分割的数据集文件"""
        try:
            # 保存完整数据集
            with open(va_pair_file, 'w', encoding='utf-8') as f:
                json.dump(va_pairs, f, indent=2, ensure_ascii=False)
            
            # 保存训练集
            train_file = va_pair_file.replace('.json', '_train.json')
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            
            # 保存验证集
            val_file = va_pair_file.replace('.json', '_val.json')
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 平衡VA-pair数据集生成完成!")
            self.logger.info(f"📊 数据集统计:")
            self.logger.info(f"   - 总样本数: {len(va_pairs)}")
            self.logger.info(f"   - 正样本数: {positive_samples}")
            self.logger.info(f"   - 负样本数: {negative_samples}")
            self.logger.info(f"   - 正负比例: {positive_samples}:{negative_samples}")
            self.logger.info(f"   - 训练集大小: {len(train_data)}")
            self.logger.info(f"   - 验证集大小: {len(val_data)}")
            self.logger.info(f"📁 数据集文件:")
            self.logger.info(f"   - 完整数据集: {va_pair_file}")
            self.logger.info(f"   - 训练集: {train_file}")
            self.logger.info(f"   - 验证集: {val_file}")
            
            self._show_samples(va_pairs)
            
        except Exception as e:
            self.logger.error(f"❌ 保存数据集失败: {e}")
    
    def _show_samples(self, va_pairs: List[Dict], num_samples: int = 3):
        """显示数据集样例"""
        if va_pairs:
            self.logger.info(f"📋 数据集样例:")
            for i, sample in enumerate(va_pairs[:num_samples]):
                self.logger.info(f"   样例 {i+1}:")
                self.logger.info(f"     - 图片: {os.path.basename(sample['image_path'])}")
                self.logger.info(f"     - 问题: {sample['question']}")
                self.logger.info(f"     - 答案: {sample['answer']}")
                self.logger.info(f"     - 标签: {sample['label']}")
                self.logger.info(f"     - 类型: {sample['sample_type']}")
                if 'source_video_id' in sample:
                    self.logger.info(f"     - 来源视频: {sample['source_video_id']}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VA-pair数据集生成器')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--questions_file', default='dataset/questions.json', help='问题文件路径')
    parser.add_argument('--mode', choices=['basic', 'balanced', 'cross'], default='balanced',
                       help='生成模式: basic(基础), balanced(平衡), cross(跨视频负样本)')
    parser.add_argument('--max_samples', type=int, default=10, help='每个视频最大样本数')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--negative_ratio', type=float, default=1.0, help='负样本比例')
    
    args = parser.parse_args()
    
    # 检查必要文件和目录
    if not os.path.exists(args.output_dir):
        logger.error(f"❌ 输出目录不存在: {args.output_dir}")
        return
    
    if not os.path.exists(args.questions_file):
        logger.error(f"❌ 问题文件不存在: {args.questions_file}")
        return
    
    # 初始化生成器
    generator = VAPairGenerator()
    
    logger.info("=== VA-pair数据集生成器 ===")
    logger.info(f"🎯 生成模式: {args.mode}")
    
    try:
        if args.mode == 'basic':
            # 基础模式
            va_pairs = generator.generate_va_pair_dataset(
                args.output_dir, args.questions_file
            )
        elif args.mode == 'balanced':
            # 平衡模式
            va_pairs = generator.generate_balanced_va_pair_dataset(
                args.output_dir, args.questions_file,
                max_samples_per_video=args.max_samples,
                split_ratio=args.split_ratio
            )
        elif args.mode == 'cross':
            # 跨视频负样本模式
            va_pairs = generator.generate_cross_video_negative_dataset(
                args.output_dir, args.questions_file,
                negative_ratio=args.negative_ratio
            )
        
        if va_pairs:
            logger.info("🎉 数据集生成成功!")
            logger.info("💡 使用方法:")
            if args.mode == 'balanced':
                logger.info(f"   python train_qwen.py --data_file {args.output_dir}/va_pair_balanced_train.json")
            else:
                logger.info(f"   python train_qwen.py --data_file {args.output_dir}/va_pair*.json")
        else:
            logger.warning("⚠️  数据集生成失败")
            
    except Exception as e:
        logger.error(f"❌ 生成过程中发生错误: {e}")


if __name__ == "__main__":
    main()