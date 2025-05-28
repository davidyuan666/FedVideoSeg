"""
Binary Search Video Processor with DeepSeek Integration
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import asyncio
from dataclasses import dataclass

from .deepseek_client import DeepSeekClient, VideoSegment

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Binary search result for video segments."""
    segment: VideoSegment
    relevance_score: float
    confidence: float
    search_depth: int

class BinarySearchVideoProcessor:
    """
    Binary search-based video processor for efficient question-relevant segment location.
    """
    
    def __init__(self, deepseek_client: DeepSeekClient, min_segment_length: float = 5.0):
        self.deepseek_client = deepseek_client
        self.min_segment_length = min_segment_length
        self.search_cache = {}  # Cache for repeated searches
        
    async def locate_relevant_segments(
        self, 
        video_path: str, 
        question: str,
        max_segments: int = 3,
        relevance_threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Use binary search to locate question-relevant video segments.
        
        Args:
            video_path: Path to video file
            question: Target question for segment location
            max_segments: Maximum number of segments to return
            relevance_threshold: Minimum relevance score threshold
            
        Returns:
            List of relevant video segments with search metadata
        """
        
        # Extract video metadata
        video_duration = self._get_video_duration(video_path)
        
        # Initialize search space
        search_results = []
        search_queue = [(0, video_duration, 0)]  # (start, end, depth)
        
        while search_queue and len(search_results) < max_segments:
            start_time, end_time, depth = search_queue.pop(0)
            
            # Skip if segment too small
            if end_time - start_time < self.min_segment_length:
                continue
                
            # Extract segment
            segment = await self._extract_segment(video_path, start_time, end_time)
            
            # Evaluate relevance with DeepSeek
            relevance_result = await self._evaluate_segment_relevance(segment, question)
            
            if relevance_result['relevance_score'] >= relevance_threshold:
                search_result = SearchResult(
                    segment=segment,
                    relevance_score=relevance_result['relevance_score'],
                    confidence=relevance_result['confidence'],
                    search_depth=depth
                )
                search_results.append(search_result)
                
                # If highly relevant, search nearby regions
                if relevance_result['relevance_score'] > 0.8:
                    mid_time = (start_time + end_time) / 2
                    
                    # Add adjacent segments to search queue
                    if start_time > 0:
                        search_queue.append((max(0, start_time - 10), start_time, depth + 1))
                    if end_time < video_duration:
                        search_queue.append((end_time, min(video_duration, end_time + 10), depth + 1))
            
            else:
                # Binary search: split segment and search halves
                if depth < 4:  # Limit search depth
                    mid_time = (start_time + end_time) / 2
                    search_queue.append((start_time, mid_time, depth + 1))
                    search_queue.append((mid_time, end_time, depth + 1))
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Binary search found {len(search_results)} relevant segments for question")
        return search_results[:max_segments]
    
    async def _extract_segment(self, video_path: str, start_time: float, end_time: float) -> VideoSegment:
        """Extract video segment between start and end times."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Seek to start time
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        frames = []
        current_time = start_time
        
        while current_time < end_time:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            current_time += 1.0 / fps
        
        cap.release()
        
        return VideoSegment(
            start_time=start_time,
            end_time=end_time,
            frames=frames,
            relevance_score=0.0
        )
    
    async def _evaluate_segment_relevance(self, segment: VideoSegment, question: str) -> Dict[str, float]:
        """Evaluate segment relevance using DeepSeek."""
        try:
            analysis = await self.deepseek_client.analyze_video_segment(
                segment, context=f"Question: {question}"
            )
            
            return {
                'relevance_score': analysis.get('relevance_score', 0.0),
                'confidence': analysis.get('confidence', 0.5)
            }
            
        except Exception as e:
            logger.error(f"DeepSeek evaluation failed: {e}")
            return {'relevance_score': 0.0, 'confidence': 0.0}
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return duration 