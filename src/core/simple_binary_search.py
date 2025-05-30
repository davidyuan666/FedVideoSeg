"""
Simplified Binary Search Video Localizer
Core implementation for video segment localization using DeepSeek evaluation
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class SimpleBinarySearchLocalizer:
    """
    Simple binary search for video segment localization
    Reduces complexity from O(T) to O(log T)
    """
    
    def __init__(self, deepseek_client, min_segment_length: float = 5.0):
        self.deepseek_client = deepseek_client
        self.min_segment_length = min_segment_length
        
    def extract_frames(self, video_path: str, start_time: float, end_time: float) -> List[np.ndarray]:
        """Extract frames from video segment"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame, int(fps)):  # 1 frame per second
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
                
        cap.release()
        return frames
    
    def evaluate_segment_relevance(self, frames: List[np.ndarray], question: str) -> float:
        """Use DeepSeek to evaluate if segment can answer question"""
        if not frames:
            return 0.0
            
        # Sample a few frames for evaluation
        sample_frames = frames[::max(1, len(frames)//3)]  # Sample 3 frames max
        
        try:
            relevance = self.deepseek_client.evaluate_frames_relevance(sample_frames, question)
            return relevance
        except Exception as e:
            logger.error(f"DeepSeek evaluation failed: {e}")
            return 0.0
    
    def binary_search_localize(self, video_path: str, question: str) -> Tuple[float, float, float]:
        """
        Binary search to find most relevant video segment
        Returns: (start_time, end_time, relevance_score)
        """
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        
        start_time = 0.0
        end_time = duration
        best_segment = (0.0, duration, 0.0)
        
        iteration = 0
        while (end_time - start_time) > self.min_segment_length and iteration < 10:
            mid_time = (start_time + end_time) / 2
            
            # Evaluate left half
            left_frames = self.extract_frames(video_path, start_time, mid_time)
            left_relevance = self.evaluate_segment_relevance(left_frames, question)
            
            # Evaluate right half  
            right_frames = self.extract_frames(video_path, mid_time, end_time)
            right_relevance = self.evaluate_segment_relevance(right_frames, question)
            
            # Choose better half
            if left_relevance >= right_relevance:
                end_time = mid_time
                current_relevance = left_relevance
            else:
                start_time = mid_time
                current_relevance = right_relevance
                
            # Update best segment
            if current_relevance > best_segment[2]:
                best_segment = (start_time, end_time, current_relevance)
                
            iteration += 1
            
        return best_segment
    
    def generate_training_data(self, video_path: str, question: str) -> List[Tuple]:
        """
        Generate frame-question training pairs for Qwen2.5-VL
        Returns: List of (frame, question, relevance_label)
        """
        # First find the most relevant segment
        start_time, end_time, _ = self.binary_search_localize(video_path, question)
        
        # Extract frames from entire video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        
        training_data = []
        
        # Sample frames every 5 seconds
        for t in range(0, int(duration), 5):
            frames = self.extract_frames(video_path, t, t+1)
            if frames:
                frame = frames[0]
                # Label as relevant if within the found segment
                is_relevant = 1 if start_time <= t <= end_time else 0
                training_data.append((frame, question, is_relevant))
                
        cap.release()
        return training_data 