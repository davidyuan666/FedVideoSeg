"""
Binary Search Video Localizer with DeepSeek LLM Evaluation
Implements Section 3.2 of the FedVideoQA paper
"""

import torch
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio
import time

from .deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)

@dataclass
class VideoSegment:
    """Video segment with relevance information"""
    start_time: float
    end_time: float
    frames: List[np.ndarray]
    audio_features: Optional[np.ndarray] = None
    relevance_score: float = 0.0
    confidence: float = 0.0

class BinarySearchLocalizer:
    """
    Binary search video localizer using LLM evaluation
    Reduces complexity from O(T) to O(log T) as described in the paper
    """
    
    def __init__(
        self,
        deepseek_client: DeepSeekClient,
        min_segment_length: float = 5.0,  # τ_min in seconds
        max_iterations: int = 10,
        relevance_threshold: float = 0.8
    ):
        self.deepseek_client = deepseek_client
        self.min_segment_length = min_segment_length
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold
        
    async def localize_segments(
        self,
        video_path: str,
        question: str,
        max_segments: int = 3
    ) -> List[VideoSegment]:
        """
        Localize relevant video segments using binary search with LLM evaluation
        
        Implements Algorithm 1 from the paper:
        BinarySearch(V, q, t_start, t_end)
        
        Args:
            video_path: Path to video file
            question: Target question for localization
            max_segments: Maximum number of segments to return
            
        Returns:
            List of relevant video segments sorted by relevance score
        """
        logger.info(f"Starting binary search localization for video: {video_path}")
        
        # Step 1: Extract video metadata and initial frames
        video_duration = self._get_video_duration(video_path)
        
        # Step 2: Initialize search with full video
        relevant_segments = []
        search_queue = [(0.0, video_duration, 0)]  # (start, end, depth)
        
        while search_queue and len(relevant_segments) < max_segments:
            start_time, end_time, depth = search_queue.pop(0)
            
            # Check termination conditions
            if depth >= self.max_iterations:
                continue
                
            segment_duration = end_time - start_time
            if segment_duration < self.min_segment_length:
                continue
            
            # Extract segment for evaluation
            segment = await self._extract_segment(video_path, start_time, end_time)
            
            # Evaluate relevance using DeepSeek (Equation 2)
            relevance_score = await self._evaluate_relevance(segment, question)
            segment.relevance_score = relevance_score
            
            logger.debug(
                f"Segment [{start_time:.1f}s - {end_time:.1f}s]: "
                f"relevance = {relevance_score:.3f}"
            )
            
            # Decision logic based on relevance score
            if relevance_score >= self.relevance_threshold:
                # High relevance: add to results and explore nearby regions
                relevant_segments.append(segment)
                
                # Explore adjacent segments for completeness
                if start_time > 0:
                    search_queue.append((
                        max(0, start_time - 10), start_time, depth + 1
                    ))
                if end_time < video_duration:
                    search_queue.append((
                        end_time, min(video_duration, end_time + 10), depth + 1
                    ))
                    
            elif relevance_score > 0.4:  # Medium relevance: continue binary search
                # Binary search: divide segment and search both halves
                mid_time = (start_time + end_time) / 2
                search_queue.append((start_time, mid_time, depth + 1))
                search_queue.append((mid_time, end_time, depth + 1))
            
            # Low relevance (< 0.4): discard segment
        
        # Sort segments by relevance score
        relevant_segments.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(
            f"Binary search completed: found {len(relevant_segments)} "
            f"relevant segments in {depth + 1} iterations"
        )
        
        return relevant_segments[:max_segments]
    
    async def _extract_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> VideoSegment:
        """
        Extract video segment between start and end times
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            VideoSegment object with frames and metadata
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Seek to start time
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        current_time = start_time
        
        # Extract frames at 1 FPS for efficiency
        frame_interval = max(1, int(fps))
        frame_count = 0
        
        while current_time < end_time:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Resize frame for efficiency
                frame_resized = cv2.resize(frame, (224, 224))
                frames.append(frame_resized)
            
            frame_count += 1
            current_time = start_time + frame_count / fps
        
        cap.release()
        
        return VideoSegment(
            start_time=start_time,
            end_time=end_time,
            frames=frames,
            relevance_score=0.0
        )
    
    async def _evaluate_relevance(
        self,
        segment: VideoSegment,
        question: str
    ) -> float:
        """
        Evaluate segment relevance using DeepSeek LLM
        
        Implements Equation 2: R(V[t1,t2], q) = DeepSeek(Prompt(...))
        
        Args:
            segment: Video segment to evaluate
            question: Target question
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Create structured prompt for DeepSeek
            prompt = self._create_evaluation_prompt(segment, question)
            
            # Get relevance score from DeepSeek
            response = await self.deepseek_client.evaluate_segment_relevance(
                prompt=prompt,
                frames=segment.frames
            )
            
            # Parse relevance score
            relevance_score = self._parse_relevance_score(response)
            
            return max(0.0, min(1.0, relevance_score))
            
        except Exception as e:
            logger.error(f"Error evaluating segment relevance: {e}")
            return 0.0
    
    def _create_evaluation_prompt(
        self,
        segment: VideoSegment,
        question: str
    ) -> str:
        """
        Create structured prompt for DeepSeek evaluation
        
        Args:
            segment: Video segment
            question: Target question
            
        Returns:
            Formatted prompt string
        """
        duration = segment.end_time - segment.start_time
        num_frames = len(segment.frames)
        
        prompt = f"""
        Evaluate the relevance of this video segment to the given question.
        
        Video Segment Information:
        - Duration: {duration:.1f} seconds
        - Time range: {segment.start_time:.1f}s to {segment.end_time:.1f}s
        - Number of frames: {num_frames}
        
        Question: {question}
        
        Please analyze the visual content and determine how relevant this segment 
        is to answering the question. Consider:
        1. Visual elements that relate to the question
        2. Potential audio/speech content (if applicable)
        3. Educational context and content
        
        Provide a relevance score between 0.0 (not relevant) and 1.0 (highly relevant).
        
        Response format: {{"relevance_score": <float>, "reasoning": "<explanation>"}}
        """
        
        return prompt.strip()
    
    def _parse_relevance_score(self, response: Dict[str, Any]) -> float:
        """
        Parse relevance score from DeepSeek response
        
        Args:
            response: DeepSeek API response
            
        Returns:
            Parsed relevance score
        """
        try:
            if isinstance(response, dict):
                return float(response.get('relevance_score', 0.0))
            elif isinstance(response, str):
                # Try to extract score from text response
                import re
                score_match = re.search(r'relevance_score["\s]*:\s*([0-9.]+)', response)
                if score_match:
                    return float(score_match.group(1))
            
            return 0.0
            
        except (ValueError, TypeError):
            return 0.0
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    
    def get_complexity_analysis(self, video_duration: float) -> Dict[str, Any]:
        """
        Analyze computational complexity improvements
        
        Args:
            video_duration: Video duration in seconds
            
        Returns:
            Complexity analysis results
        """
        # Traditional exhaustive search: O(T)
        traditional_operations = video_duration
        
        # Binary search: O(log T)
        binary_operations = math.log2(video_duration) if video_duration > 0 else 1
        
        # Theoretical speedup (Equation 12)
        speedup = traditional_operations / binary_operations if binary_operations > 0 else 1
        
        return {
            'video_duration': video_duration,
            'traditional_complexity': traditional_operations,
            'binary_complexity': binary_operations,
            'theoretical_speedup': speedup,
            'efficiency_gain': (1 - binary_operations / traditional_operations) * 100
        }

class SegmentQualityAssessment:
    """
    Quality assessment for video segments
    """
    
    def __init__(self, min_quality_score: float = 0.6):
        self.min_quality_score = min_quality_score
    
    def assess_segment_quality(self, segment: VideoSegment) -> Dict[str, float]:
        """
        Assess the quality of a video segment
        
        Args:
            segment: Video segment to assess
            
        Returns:
            Quality assessment scores
        """
        scores = {}
        
        # Visual quality assessment
        if segment.frames:
            scores['visual_quality'] = self._assess_visual_quality(segment.frames)
        else:
            scores['visual_quality'] = 0.0
        
        # Duration appropriateness
        duration = segment.end_time - segment.start_time
        scores['duration_quality'] = self._assess_duration_quality(duration)
        
        # Overall quality score
        scores['overall_quality'] = (
            0.7 * scores['visual_quality'] + 
            0.3 * scores['duration_quality']
        )
        
        return scores
    
    def _assess_visual_quality(self, frames: List[np.ndarray]) -> float:
        """Assess visual quality of frames"""
        if not frames:
            return 0.0
        
        # Simple quality metrics
        quality_scores = []
        
        for frame in frames[:5]:  # Sample first 5 frames
            # Compute image sharpness using Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize sharpness score
            normalized_sharpness = min(1.0, sharpness / 1000.0)
            quality_scores.append(normalized_sharpness)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _assess_duration_quality(self, duration: float) -> float:
        """Assess if duration is appropriate for educational content"""
        # Optimal duration range: 5-30 seconds
        if 5 <= duration <= 30:
            return 1.0
        elif duration < 5:
            return duration / 5.0
        else:  # duration > 30
            return max(0.3, 30.0 / duration) 