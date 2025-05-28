"""
Video Processing Module with DFS-based Segmentation
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from .deepseek_client import DeepSeekClient, VideoSegment

logger = logging.getLogger(__name__)

@dataclass
class SegmentationConfig:
    """Configuration for video segmentation."""
    min_segment_length: float = 5.0  # seconds
    max_segment_length: float = 30.0  # seconds
    similarity_threshold: float = 0.8
    scene_change_threshold: float = 0.3
    max_depth: int = 5
    frame_sample_rate: int = 1  # frames per second

class DFSSegmentationEngine:
    """
    Depth-First Search based video segmentation engine.
    Automatically identifies question-relevant segments.
    """
    
    def __init__(self, config: SegmentationConfig, deepseek_client: DeepSeekClient):
        self.config = config
        self.deepseek_client = deepseek_client
        self.frame_cache = {}
        
    def extract_frames(self, video_path: str) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video with timestamps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / self.config.frame_sample_rate)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames using histogram comparison."""
        # Convert to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Compare histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, similarity)  # Ensure non-negative
    
    def detect_scene_changes(self, frames: List[Tuple[float, np.ndarray]]) -> List[int]:
        """Detect scene changes in the video."""
        scene_changes = [0]  # Always start with first frame
        
        for i in range(1, len(frames)):
            similarity = self.calculate_frame_similarity(frames[i-1][1], frames[i][1])
            
            if similarity < self.config.scene_change_threshold:
                scene_changes.append(i)
                logger.debug(f"Scene change detected at frame {i} (similarity: {similarity:.3f})")
        
        scene_changes.append(len(frames) - 1)  # Always end with last frame
        return scene_changes
    
    async def dfs_segment_video(
        self, 
        frames: List[Tuple[float, np.ndarray]], 
        question: str = "",
        context: str = ""
    ) -> List[VideoSegment]:
        """
        Use DFS to find optimal video segments for question answering.
        
        Args:
            frames: List of (timestamp, frame) tuples
            question: Target question for relevance scoring
            context: Additional context for segmentation
            
        Returns:
            List of optimized video segments
        """
        logger.info(f"Starting DFS segmentation for {len(frames)} frames")
        
        # Detect initial scene boundaries
        scene_changes = self.detect_scene_changes(frames)
        logger.info(f"Detected {len(scene_changes)} scene boundaries")
        
        # Initialize DFS with scene-based segments
        initial_segments = self._create_initial_segments(frames, scene_changes)
        
        # DFS exploration for optimal segmentation
        optimal_segments = await self._dfs_explore(
            initial_segments, question, context, depth=0
        )
        
        # Post-process and filter segments
        final_segments = self._post_process_segments(optimal_segments)
        
        logger.info(f"DFS segmentation complete: {len(final_segments)} segments")
        return final_segments
    
    def _create_initial_segments(
        self, 
        frames: List[Tuple[float, np.ndarray]], 
        scene_changes: List[int]
    ) -> List[VideoSegment]:
        """Create initial segments based on scene changes."""
        segments = []
        
        for i in range(len(scene_changes) - 1):
            start_idx = scene_changes[i]
            end_idx = scene_changes[i + 1]
            
            start_time = frames[start_idx][0]
            end_time = frames[end_idx][0]
            
            # Check segment length constraints
            duration = end_time - start_time
            if duration < self.config.min_segment_length:
                continue
            
            # Extract frames for this segment
            segment_frames = [frames[j][1] for j in range(start_idx, end_idx + 1)]
            
            segment = VideoSegment(
                start_time=start_time,
                end_time=end_time,
                frames=segment_frames,
                relevance_score=0.0
            )
            
            segments.append(segment)
        
        return segments
    
    async def _dfs_explore(
        self, 
        segments: List[VideoSegment], 
        question: str, 
        context: str, 
        depth: int
    ) -> List[VideoSegment]:
        """
        DFS exploration to find optimal segments.
        
        Args:
            segments: Current segments to explore
            question: Target question
            context: Additional context
            depth: Current recursion depth
            
        Returns:
            Optimized segments
        """
        if depth >= self.config.max_depth or not segments:
            return segments
        
        # Analyze current segments
        analyzed_segments = await self._analyze_segments_batch(segments, question, context)
        
        # Sort by relevance score
        analyzed_segments.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Select top segments for further exploration
        top_segments = analyzed_segments[:min(3, len(analyzed_segments))]
        
        # Recursively explore promising segments
        refined_segments = []
        for segment in top_segments:
            if segment.relevance_score > 0.5:  # Threshold for further exploration
                sub_segments = await self._split_segment(segment)
                if len(sub_segments) > 1:
                    # Recursively explore sub-segments
                    refined_sub = await self._dfs_explore(
                        sub_segments, question, context, depth + 1
                    )
                    refined_segments.extend(refined_sub)
                else:
                    refined_segments.append(segment)
            else:
                refined_segments.append(segment)
        
        return refined_segments
    
    async def _analyze_segments_batch(
        self, 
        segments: List[VideoSegment], 
        question: str, 
        context: str
    ) -> List[VideoSegment]:
        """Analyze segments in batch using DeepSeek API."""
        try:
            # Prepare context for analysis
            analysis_context = f"Question: {question}\nContext: {context}"
            
            # Analyze segments
            results = await self.deepseek_client.batch_analyze_segments(
                segments, analysis_context
            )
            
            # Update segments with analysis results
            for i, (segment, result) in enumerate(zip(segments, results)):
                if isinstance(result, dict) and 'relevance_score' in result:
                    segment.relevance_score = result['relevance_score']
                    if 'questions' in result:
                        segment.questions = [q.get('question', '') for q in result['questions']]
                else:
                    segment.relevance_score = 0.1  # Default low score for failed analysis
            
            return segments
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # Return segments with default scores
            for segment in segments:
                segment.relevance_score = 0.1
            return segments
    
    async def _split_segment(self, segment: VideoSegment) -> List[VideoSegment]:
        """Split a segment into smaller sub-segments."""
        duration = segment.end_time - segment.start_time
        
        # Don't split if already at minimum length
        if duration <= self.config.min_segment_length * 2:
            return [segment]
        
        # Split into two equal parts
        mid_time = segment.start_time + duration / 2
        mid_frame_idx = len(segment.frames) // 2
        
        segment1 = VideoSegment(
            start_time=segment.start_time,
            end_time=mid_time,
            frames=segment.frames[:mid_frame_idx],
            relevance_score=0.0
        )
        
        segment2 = VideoSegment(
            start_time=mid_time,
            end_time=segment.end_time,
            frames=segment.frames[mid_frame_idx:],
            relevance_score=0.0
        )
        
        return [segment1, segment2]
    
    def _post_process_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Post-process segments to remove duplicates and merge similar ones."""
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: x.start_time)
        
        # Remove overlapping segments, keeping higher relevance scores
        filtered_segments = []
        for segment in segments:
            if not filtered_segments:
                filtered_segments.append(segment)
                continue
            
            last_segment = filtered_segments[-1]
            
            # Check for overlap
            if segment.start_time < last_segment.end_time:
                # Keep segment with higher relevance score
                if segment.relevance_score > last_segment.relevance_score:
                    filtered_segments[-1] = segment
            else:
                filtered_segments.append(segment)
        
        # Filter by minimum relevance score
        final_segments = [s for s in filtered_segments if s.relevance_score > 0.3]
        
        return final_segments

class VideoProcessor:
    """Main video processing class."""
    
    def __init__(self, deepseek_client: DeepSeekClient, config: Optional[SegmentationConfig] = None):
        self.deepseek_client = deepseek_client
        self.config = config or SegmentationConfig()
        self.dfs_engine = DFSSegmentationEngine(self.config, deepseek_client)
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    def validate_video(self, video_path: str) -> bool:
        """Validate video file."""
        path = Path(video_path)
        
        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        if path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported video format: {path.suffix}")
            return False
        
        # Check if video can be opened
        cap = cv2.VideoCapture(video_path)
        is_valid = cap.isOpened()
        cap.release()
        
        return is_valid
    
    async def process_video(
        self, 
        video_path: str, 
        question: str = "",
        context: str = "",
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process video and extract question-relevant segments.
        
        Args:
            video_path: Path to video file
            question: Target question for segmentation
            context: Additional context
            output_dir: Directory to save results
            
        Returns:
            Processing results including segments and metadata
        """
        if not self.validate_video(video_path):
            raise ValueError(f"Invalid video file: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        try:
            # Extract frames
            frames = self.dfs_engine.extract_frames(video_path)
            
            # Perform DFS segmentation
            segments = await self.dfs_engine.dfs_segment_video(frames, question, context)
            
            # Generate metadata
            metadata = self._generate_metadata(video_path, segments, question, context)
            
            # Save results if output directory specified
            if output_dir:
                await self._save_results(output_dir, segments, metadata)
            
            logger.info(f"Video processing complete: {len(segments)} segments extracted")
            
            return {
                'segments': segments,
                'metadata': metadata,
                'total_segments': len(segments),
                'total_duration': sum(s.end_time - s.start_time for s in segments),
                'average_relevance': np.mean([s.relevance_score for s in segments]) if segments else 0.0
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
    
    def _generate_metadata(
        self, 
        video_path: str, 
        segments: List[VideoSegment], 
        question: str, 
        context: str
    ) -> Dict[str, Any]:
        """Generate metadata for processed video."""
        return {
            'video_path': video_path,
            'video_hash': self._calculate_file_hash(video_path),
            'question': question,
            'context': context,
            'processing_timestamp': np.datetime64('now').isoformat(),
            'num_segments': len(segments),
            'total_duration': sum(s.end_time - s.start_time for s in segments),
            'config': asdict(self.config),
            'segment_summary': [
                {
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration': s.end_time - s.start_time,
                    'relevance_score': s.relevance_score,
                    'num_questions': len(s.questions)
                }
                for s in segments
            ]
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _save_results(
        self, 
        output_dir: str, 
        segments: List[VideoSegment], 
        metadata: Dict[str, Any]
    ):
        """Save processing results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save segment information
        segments_file = output_path / "segments.json"
        segments_data = []
        
        for i, segment in enumerate(segments):
            segment_data = {
                'id': i,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'relevance_score': segment.relevance_score,
                'questions': segment.questions,
                'num_frames': len(segment.frames)
            }
            segments_data.append(segment_data)
        
        with open(segments_file, 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}") 