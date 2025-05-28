"""
DeepSeek API Client for Video Analysis and Question Generation
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass
import base64
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class VideoSegment:
    """Represents a video segment with metadata."""
    start_time: float
    end_time: float
    frames: List[np.ndarray]
    audio_features: Optional[np.ndarray] = None
    relevance_score: float = 0.0
    questions: List[str] = None
    
    def __post_init__(self):
        if self.questions is None:
            self.questions = []

class DeepSeekClient:
    """Client for interacting with DeepSeek API for video analysis."""
    
    def __init__(self, api_key: str, api_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64 for API transmission."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    async def analyze_video_segment(
        self, 
        segment: VideoSegment,
        context: str = "",
        max_questions: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze a video segment and generate relevant questions.
        
        Args:
            segment: Video segment to analyze
            context: Additional context for analysis
            max_questions: Maximum number of questions to generate
            
        Returns:
            Analysis results including questions and relevance scores
        """
        try:
            # Sample frames from segment for analysis
            sample_frames = self._sample_frames(segment.frames, max_frames=5)
            encoded_frames = [self.encode_frame(frame) for frame in sample_frames]
            
            prompt = self._create_analysis_prompt(
                segment, context, max_questions, encoded_frames
            )
            
            response = await self._make_api_call(prompt)
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Error analyzing video segment: {e}")
            return {"questions": [], "relevance_score": 0.0, "error": str(e)}
    
    def _sample_frames(self, frames: List[np.ndarray], max_frames: int = 5) -> List[np.ndarray]:
        """Sample representative frames from the segment."""
        if len(frames) <= max_frames:
            return frames
        
        # Use uniform sampling
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def _create_analysis_prompt(
        self, 
        segment: VideoSegment, 
        context: str, 
        max_questions: int,
        encoded_frames: List[str]
    ) -> Dict[str, Any]:
        """Create prompt for DeepSeek API."""
        
        system_prompt = """You are an expert educational video analyst. Your task is to:
1. Analyze video segments for educational content
2. Generate relevant questions that test understanding
3. Assess the educational value and relevance of content
4. Provide detailed analysis for question-answering systems

Focus on creating questions that are:
- Pedagogically sound
- Appropriate for the content level
- Clear and unambiguous
- Testable through video content
"""
        
        user_prompt = f"""
Analyze this video segment (duration: {segment.end_time - segment.start_time:.2f}s):

Context: {context}

Based on the provided frames, please:
1. Identify the main educational concepts
2. Generate up to {max_questions} relevant questions
3. Rate the educational relevance (0-1 scale)
4. Suggest key timestamps for answers

Respond in JSON format:
{{
    "concepts": ["concept1", "concept2", ...],
    "questions": [
        {{
            "question": "question text",
            "difficulty": "easy|medium|hard",
            "type": "factual|conceptual|analytical",
            "expected_answer_location": "timestamp_range"
        }}
    ],
    "relevance_score": 0.0-1.0,
    "summary": "brief content summary"
}}
"""
        
        return {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.7
        }
    
    async def _make_api_call(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Make asynchronous API call to DeepSeek."""
        try:
            response = self.session.post(
                f"{self.api_url}/chat/completions",
                json=prompt,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def _parse_analysis_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate API response."""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Try to parse as JSON
            if content.strip().startswith('{'):
                return json.loads(content)
            else:
                # Fallback parsing for non-JSON responses
                return self._fallback_parse(content)
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse API response: {e}")
            return self._fallback_parse(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
    
    def _fallback_parse(self, content: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses."""
        return {
            "concepts": [],
            "questions": [],
            "relevance_score": 0.5,
            "summary": content[:200] + "..." if len(content) > 200 else content,
            "raw_response": content
        }
    
    async def batch_analyze_segments(
        self, 
        segments: List[VideoSegment],
        context: str = "",
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Analyze multiple video segments concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(segment):
            async with semaphore:
                return await self.analyze_video_segment(segment, context)
        
        tasks = [analyze_with_semaphore(segment) for segment in segments]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def validate_api_key(self) -> bool:
        """Validate the API key by making a test call."""
        try:
            test_prompt = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = self.session.post(
                f"{self.api_url}/chat/completions",
                json=test_prompt,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False 