"""
Multimodal Feature Extractors for Video, Audio, and Text
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import whisper
import clip
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def extract_features(self, input_data: Any) -> torch.Tensor:
        """Extract features from input data."""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        pass

class CLIPExtractor(BaseFeatureExtractor):
    """CLIP-based visual feature extractor."""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        super().__init__(device)
        self.model_name = model_name
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_features = self.model.encode_image(dummy_input)
            self.feature_dim = dummy_features.shape[-1]
    
    def extract_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Extract CLIP features from video frames.
        
        Args:
            frames: List of video frames as numpy arrays
            
        Returns:
            Tensor of shape (num_frames, feature_dim)
        """
        if not frames:
            return torch.zeros(0, self.feature_dim).to(self.device)
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Preprocess for CLIP
            processed_frame = self.preprocess(frame_rgb).unsqueeze(0)
            processed_frames.append(processed_frame)
        
        # Stack frames
        batch_frames = torch.cat(processed_frames, dim=0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(batch_frames)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize
        
        return features
    
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """Extract CLIP text features."""
        if not texts:
            return torch.zeros(0, self.feature_dim).to(self.device)
        
        # Tokenize texts
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Extract features
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

class WhisperExtractor(BaseFeatureExtractor):
    """Whisper-based audio feature extractor."""
    
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        super().__init__(device)
        self.model_name = model_name
        self.model = whisper.load_model(model_name, device=self.device)
        self.feature_dim = self.model.dims.n_audio_state  # Whisper feature dimension
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract Whisper features from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Tensor of audio features
        """
        if len(audio_data) == 0:
            return torch.zeros(0, self.feature_dim).to(self.device)
        
        try:
            # Ensure audio is in correct format for Whisper
            if sample_rate != 16000:
                # Resample to 16kHz (Whisper's expected sample rate)
                audio_data = self._resample_audio(audio_data, sample_rate, 16000)
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data) + 1e-8)
            
            # Extract features using Whisper encoder
            with torch.no_grad():
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio_data).to(self.device)
                
                # Pad or trim to 30 seconds (Whisper's expected length)
                target_length = 16000 * 30  # 30 seconds at 16kHz
                if len(audio_tensor) > target_length:
                    audio_tensor = audio_tensor[:target_length]
                else:
                    audio_tensor = torch.nn.functional.pad(
                        audio_tensor, (0, target_length - len(audio_tensor))
                    )
                
                # Extract mel spectrogram
                mel = whisper.log_mel_spectrogram(audio_tensor)
                
                # Encode with Whisper
                features = self.model.encoder(mel.unsqueeze(0))
                
                # Average pool over time dimension
                features = features.mean(dim=1)  # (1, feature_dim)
                
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return torch.zeros(1, self.feature_dim).to(self.device)
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple audio resampling."""
        # This is a simplified resampling - in practice, use librosa or similar
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Transcribe audio to text."""
        try:
            # Prepare audio for Whisper
            if sample_rate != 16000:
                audio_data = self._resample_audio(audio_data, sample_rate, 16000)
            
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data) + 1e-8)
            
            # Transcribe
            result = self.model.transcribe(audio_data)
            
            return {
                'text': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return {'text': '', 'segments': [], 'language': 'unknown'}
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

class BERTExtractor(BaseFeatureExtractor):
    """BERT-based text feature extractor."""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cuda"):
        super().__init__(device)
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.feature_dim = self.model.config.hidden_size
    
    def extract_features(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Extract BERT features from texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Tensor of shape (num_texts, feature_dim)
        """
        if not texts:
            return torch.zeros(0, self.feature_dim).to(self.device)
        
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use [CLS] token representation
            features = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        return features
    
    def extract_contextual_features(
        self, 
        texts: List[str], 
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """Extract contextual features including attention weights."""
        if not texts:
            return {
                'features': torch.zeros(0, self.feature_dim).to(self.device),
                'attention_weights': torch.zeros(0, 12, max_length, max_length).to(self.device)
            }
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Extract features with attention
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Features from [CLS] token
            features = outputs.last_hidden_state[:, 0, :]
            
            # Attention weights from all layers
            attention_weights = torch.stack(outputs.attentions, dim=1)  # (batch, layers, heads, seq, seq)
            
        return {
            'features': features,
            'attention_weights': attention_weights,
            'tokens': encoded['input_ids'],
            'attention_mask': attention_mask
        }
    
    def get_feature_dim(self) -> int:
        return self.feature_dim

class MultimodalFeatureExtractor:
    """Combined multimodal feature extractor."""
    
    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        whisper_model: str = "base", 
        bert_model: str = "bert-base-uncased",
        device: str = "cuda"
    ):
        self.device = device
        
        # Initialize extractors
        self.clip_extractor = CLIPExtractor(clip_model, device)
        self.whisper_extractor = WhisperExtractor(whisper_model, device)
        self.bert_extractor = BERTExtractor(bert_model, device)
        
        # Feature dimensions
        self.visual_dim = self.clip_extractor.get_feature_dim()
        self.audio_dim = self.whisper_extractor.get_feature_dim()
        self.text_dim = self.bert_extractor.get_feature_dim()
        
        logger.info(f"Multimodal extractor initialized:")
        logger.info(f"  Visual dim: {self.visual_dim}")
        logger.info(f"  Audio dim: {self.audio_dim}")
        logger.info(f"  Text dim: {self.text_dim}")
    
    def extract_all_features(
        self,
        frames: List[np.ndarray],
        audio_data: Optional[np.ndarray] = None,
        texts: Optional[List[str]] = None,
        sample_rate: int = 16000
    ) -> Dict[str, torch.Tensor]:
        """Extract features from all modalities."""
        features = {}
        
        # Visual features
        if frames:
            features['visual'] = self.clip_extractor.extract_features(frames)
        else:
            features['visual'] = torch.zeros(0, self.visual_dim).to(self.device)
        
        # Audio features
        if audio_data is not None and len(audio_data) > 0:
            features['audio'] = self.whisper_extractor.extract_features(audio_data, sample_rate)
        else:
            features['audio'] = torch.zeros(1, self.audio_dim).to(self.device)
        
        # Text features
        if texts:
            features['text'] = self.bert_extractor.extract_features(texts)
        else:
            features['text'] = torch.zeros(0, self.text_dim).to(self.device)
        
        return features
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions for each modality."""
        return {
            'visual': self.visual_dim,
            'audio': self.audio_dim,
            'text': self.text_dim
        } 