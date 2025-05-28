"""Core modules for FedVideoQA framework."""

from .video_processor import VideoProcessor, DFSSegmentationEngine
from .multimodal_extractor import MultimodalFeatureExtractor
from .qa_system import VideoQASystem
from .deepseek_client import DeepSeekClient

__all__ = [
    'VideoProcessor',
    'DFSSegmentationEngine', 
    'MultimodalFeatureExtractor',
    'VideoQASystem',
    'DeepSeekClient'
] 