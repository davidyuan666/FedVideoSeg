"""
Enhanced Binary Search Video Localizer
Supports both fine-tuning and preference data generation
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
import time
import random

logger = logging.getLogger(__name__)

class BinarySearchLocalizer:
    """
    Enhanced binary search for video segment localization
    Supports preference data generation for alignment tuning
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
        Generate frame-question training pairs for fine-tuning
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
    
    def generate_preference_data(self, video_path: str, question: str, num_pairs: int = 10) -> List[Tuple]:
        """
        Generate preference pairs for alignment tuning
        Returns: List of (preferred_frame, preferred_question, rejected_frame, rejected_question)
        """
        # Get the most relevant segment
        start_time, end_time, _ = self.binary_search_localize(video_path, question)
        
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        
        preference_data = []
        
        # Generate preference pairs
        for _ in range(num_pairs):
            # Select a time within the relevant segment for preferred
            preferred_time = random.uniform(start_time, end_time)
            preferred_frames = self.extract_frames(video_path, preferred_time, preferred_time + 1)
            
            # Select a time outside the relevant segment for rejected
            rejected_time = self._sample_outside_segment(0, duration, start_time, end_time)
            rejected_frames = self.extract_frames(video_path, rejected_time, rejected_time + 1)
            
            if preferred_frames and rejected_frames:
                # Create variations of the question
                preferred_question = question
                rejected_question = self._generate_question_variant(question)
                
                preference_data.append((
                    preferred_frames[0],   # preferred frame
                    preferred_question,    # preferred question
                    rejected_frames[0],    # rejected frame  
                    rejected_question      # rejected question
                ))
        
        return preference_data
    
    def _sample_outside_segment(self, video_start: float, video_end: float, 
                               segment_start: float, segment_end: float) -> float:
        """Sample a time point outside the given segment"""
        # Create ranges outside the segment
        ranges = []
        if segment_start > video_start:
            ranges.append((video_start, segment_start))
        if segment_end < video_end:
            ranges.append((segment_end, video_end))
        
        if not ranges:
            # If no ranges available, sample from the beginning
            return video_start
        
        # Choose a random range and sample from it
        chosen_range = random.choice(ranges)
        return random.uniform(chosen_range[0], chosen_range[1])
    
    def _generate_question_variant(self, original_question: str) -> str:
        """Generate a variant of the question for preference learning"""
        # Simple question variants for demonstration
        variants = [
            original_question,  # Sometimes use original
            f"Can you see {original_question.lower()}",
            f"What about {original_question.lower()}",
            f"Is there {original_question.lower()}",
            original_question.replace("What", "Where").replace("what", "where"),
            original_question.replace("How", "Why").replace("how", "why")
        ]
        
        return random.choice(variants)
    
    def generate_multi_level_preference_data(self, video_path: str, question: str) -> List[Tuple]:
        """
        Generate more sophisticated preference data with multiple relevance levels
        """
        # Find segments with different relevance levels
        segments_with_scores = self._find_multiple_segments(video_path, question)
        
        preference_data = []
        
        # Create preference pairs from different relevance levels
        for i, (seg1_start, seg1_end, score1) in enumerate(segments_with_scores):
            for seg2_start, seg2_end, score2 in segments_with_scores[i+1:]:
                if abs(score1 - score2) > 0.2:  # Significant difference
                    # Higher score is preferred
                    if score1 > score2:
                        preferred_frames = self.extract_frames(video_path, seg1_start, seg1_start + 1)
                        rejected_frames = self.extract_frames(video_path, seg2_start, seg2_start + 1)
                    else:
                        preferred_frames = self.extract_frames(video_path, seg2_start, seg2_start + 1)
                        rejected_frames = self.extract_frames(video_path, seg1_start, seg1_start + 1)
                    
                    if preferred_frames and rejected_frames:
                        preference_data.append((
                            preferred_frames[0],
                            question,
                            rejected_frames[0], 
                            self._generate_question_variant(question)
                        ))
        
        return preference_data
    
    def _find_multiple_segments(self, video_path: str, question: str, num_segments: int = 5) -> List[Tuple[float, float, float]]:
        """Find multiple segments with their relevance scores"""
        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        
        segments = []
        segment_length = duration / num_segments
        
        # Evaluate each segment
        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, duration)
            
            frames = self.extract_frames(video_path, start_time, end_time)
            relevance = self.evaluate_segment_relevance(frames, question)
            
            segments.append((start_time, end_time, relevance))
        
        # Sort by relevance score
        segments.sort(key=lambda x: x[2], reverse=True)
        
        return segments