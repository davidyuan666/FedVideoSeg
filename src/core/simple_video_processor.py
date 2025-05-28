"""
Simplified Video Processing Module
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import asyncio

from .deepseek_client import DeepSeekClient, VideoSegment

logger = logging.getLogger(__name__)

class SimpleVideoProcessor:
    """Simplified video processing without complex DFS."""
    
    def __init__(self, deepseek_client: DeepSeekClient):
        self.deepseek_client = deepseek_client
        self.segment_length = 15  # Fixed 15-second segments
        
    def extract_frames(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """Extract frames at 1 FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract 1 frame per second
            if frame_count % int(fps) == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def create_fixed_segments(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoSegment]:
        """Create fixed-length segments."""
        segments = []
        
        for i in range(0, len(frames), self.segment_length):
            segment_frames = frames[i:i + self.segment_length]
            
            if len(segment_frames) < 3:  # Skip very short segments
                continue
                
            start_time = segment_frames[0][0]
            end_time = segment_frames[-1][0]
            frame_data = [frame for _, frame in segment_frames]
            
            segment = VideoSegment(
                start_time=start_time,
                end_time=end_time,
                frames=frame_data,
                relevance_score=0.5  # Default score
            )
            segments.append(segment)
        
        return segments
    
    async def process_video_simple(
        self, 
        video_path: str, 
        question: str = ""
    ) -> Dict[str, any]:
        """Simplified video processing."""
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Create fixed segments
        segments = self.create_fixed_segments(frames)
        
        # Optional: analyze with DeepSeek (simplified)
        if question and self.deepseek_client:
            try:
                # Analyze only first few segments to save API calls
                sample_segments = segments[:3]
                results = await self.deepseek_client.batch_analyze_segments(
                    sample_segments, question, max_concurrent=1
                )
                
                # Update relevance scores
                for segment, result in zip(sample_segments, results):
                    if isinstance(result, dict):
                        segment.relevance_score = result.get('relevance_score', 0.5)
                        
            except Exception as e:
                logger.warning(f"DeepSeek analysis failed: {e}")
        
        return {
            'segments': segments,
            'total_segments': len(segments),
            'processing_time': len(frames) * 0.01  # Estimated
        } 