"""Models module for FedVideoQA framework."""

from .multimodal_model import MultimodalVideoQAModel
from .attention_mechanisms import DynamicAttentionFusion
from .feature_extractors import CLIPExtractor, WhisperExtractor, BERTExtractor

__all__ = [
    'MultimodalVideoQAModel',
    'DynamicAttentionFusion', 
    'CLIPExtractor',
    'WhisperExtractor',
    'BERTExtractor'
] 