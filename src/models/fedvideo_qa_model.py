"""
FedVideoQA: Enhanced Multimodal Video Question Answering Model
Based on the paper methodology combining LLM, Federated Learning, and Video Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import math

from .attention_mechanisms import DynamicAttentionFusion, CrossModalAttention
from .feature_extractors import MultimodalFeatureExtractor

logger = logging.getLogger(__name__)

class DeviceCapabilityConfig:
    """Device capability configuration for adaptive processing"""
    
    MOBILE = {
        'visual': 0.5,
        'audio': 0.8, 
        'text': 0.9,
        'compression_ratio': 0.5,
        'sparsification_ratio': 0.1
    }
    
    DESKTOP = {
        'visual': 0.8,
        'audio': 0.9,
        'text': 0.9, 
        'compression_ratio': 0.8,
        'sparsification_ratio': 0.3
    }
    
    SERVER = {
        'visual': 1.0,
        'audio': 1.0,
        'text': 1.0,
        'compression_ratio': 1.0,
        'sparsification_ratio': 1.0
    }

class AdaptiveMultimodalFusion(nn.Module):
    """
    Adaptive multimodal fusion based on device capabilities
    Implements equations from Section 3.3 of the paper
    """
    
    def __init__(
        self,
        visual_dim: int = 512,  # CLIP ViT-B/32
        audio_dim: int = 768,   # Whisper-base
        text_dim: int = 768,    # BERT-base
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Projection layers to common dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.visual_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.audio_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.text_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Dynamic weight learning network
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor, 
        text_features: torch.Tensor,
        device_capabilities: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with device-adaptive fusion
        
        Args:
            visual_features: CLIP features [batch_size, seq_len, 512]
            audio_features: Whisper features [batch_size, seq_len, 768]
            text_features: BERT features [batch_size, seq_len, 768]
            device_capabilities: Device capability scores
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        batch_size = visual_features.size(0)
        
        # Project to common dimension
        h_v = self.visual_proj(visual_features)  # [batch_size, seq_len, hidden_dim]
        h_a = self.audio_proj(audio_features)
        h_t = self.text_proj(text_features)
        
        # Apply device capability scaling
        h_v = h_v * device_capabilities['visual']
        h_a = h_a * device_capabilities['audio'] 
        h_t = h_t * device_capabilities['text']
        
        # Cross-modal attention
        h_vt, vt_attn = self.visual_cross_attn(h_v, h_t)
        h_at, at_attn = self.audio_cross_attn(h_a, h_t)
        h_va, va_attn = self.text_cross_attn(h_v, h_a)
        
        # Pool features (mean pooling over sequence dimension)
        h_v_pooled = h_vt.mean(dim=1)  # [batch_size, hidden_dim]
        h_a_pooled = h_at.mean(dim=1)
        h_t_pooled = h_va.mean(dim=1)
        
        # Dynamic weight computation (Equation 4 from paper)
        combined_features = torch.cat([h_v_pooled, h_a_pooled, h_t_pooled], dim=-1)
        alpha = self.weight_network(combined_features)  # [batch_size, 3]
        
        # Apply dynamic weights (Equation 5 from paper)
        weighted_visual = h_v_pooled * alpha[:, 0:1]
        weighted_audio = h_a_pooled * alpha[:, 1:2]
        weighted_text = h_t_pooled * alpha[:, 2:3]
        
        # Final fusion
        fused_features = torch.cat([weighted_visual, weighted_audio, weighted_text], dim=-1)
        output = self.fusion_layer(fused_features)
        
        return {
            'fused_features': output,
            'modality_weights': alpha,
            'attention_weights': {
                'visual_text': vt_attn,
                'audio_text': at_attn,
                'visual_audio': va_attn
            },
            'individual_features': {
                'visual': h_v_pooled,
                'audio': h_a_pooled,
                'text': h_t_pooled
            }
        }

class AnswerGenerationModule(nn.Module):
    """
    Answer generation module supporting both classification and generation
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        vocab_size: int = 30522,  # BERT vocab size
        max_answer_length: int = 50,
        num_answer_choices: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_answer_length = max_answer_length
        
        # Classification head for multiple choice questions
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_answer_choices)
        )
        
        # Generation head for open-ended questions
        self.generation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        fused_features: torch.Tensor,
        task_type: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate answers based on fused multimodal features
        
        Args:
            fused_features: Fused multimodal features [batch_size, hidden_dim]
            task_type: "classification" or "generation"
            
        Returns:
            Dictionary containing answer predictions and confidence scores
        """
        outputs = {}
        
        # Confidence estimation
        confidence = self.confidence_estimator(fused_features)
        outputs['confidence'] = confidence
        
        if task_type == "classification":
            # Multiple choice answer prediction
            logits = self.classification_head(fused_features)
            outputs['answer_logits'] = logits
            outputs['answer_probs'] = F.softmax(logits, dim=-1)
            
        elif task_type == "generation":
            # Open-ended answer generation
            logits = self.generation_head(fused_features)
            outputs['generation_logits'] = logits
            outputs['generation_probs'] = F.softmax(logits, dim=-1)
            
        return outputs

class FedVideoQAModel(nn.Module):
    """
    Main FedVideoQA model implementing the complete architecture from the paper
    Combines binary search localization, adaptive fusion, and federated learning
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        audio_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        vocab_size: int = 30522,
        max_answer_length: int = 50,
        num_answer_choices: int = 4,
        device_type: str = "desktop"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device_type = device_type
        
        # Get device capabilities
        if device_type == "mobile":
            self.device_config = DeviceCapabilityConfig.MOBILE
        elif device_type == "desktop":
            self.device_config = DeviceCapabilityConfig.DESKTOP
        else:  # server
            self.device_config = DeviceCapabilityConfig.SERVER
            
        # Adaptive multimodal fusion
        self.multimodal_fusion = AdaptiveMultimodalFusion(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Answer generation module
        self.answer_generator = AnswerGenerationModule(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_answer_length=max_answer_length,
            num_answer_choices=num_answer_choices,
            dropout=dropout
        )
        
        # Question-aware attention
        self.question_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        question_features: torch.Tensor,
        task_type: str = "classification",
        answer_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete FedVideoQA model
        
        Args:
            visual_features: CLIP features [batch_size, seq_len, 512]
            audio_features: Whisper features [batch_size, seq_len, 768]
            text_features: BERT features [batch_size, seq_len, 768]
            question_features: BERT question features [batch_size, seq_len, 768]
            task_type: "classification" or "generation"
            answer_labels: Ground truth labels for training
            
        Returns:
            Dictionary containing predictions, losses, and attention weights
        """
        
        # Step 1: Adaptive multimodal fusion (Section 3.3)
        fusion_output = self.multimodal_fusion(
            visual_features=visual_features,
            audio_features=audio_features,
            text_features=text_features,
            device_capabilities=self.device_config
        )
        
        fused_features = fusion_output['fused_features']
        
        # Step 2: Temporal modeling with transformer
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)
            
        temporal_features = self.transformer_encoder(fused_features)
        
        # Step 3: Question-aware attention
        if question_features.dim() == 2:
            question_features = question_features.unsqueeze(1)
            
        attended_features, qa_attention = self.question_attention(
            query=temporal_features,
            key=question_features,
            value=question_features
        )
        
        # Residual connection and normalization
        final_features = self.layer_norm(temporal_features + attended_features)
        pooled_features = final_features.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Step 4: Answer generation
        answer_output = self.answer_generator(
            fused_features=pooled_features,
            task_type=task_type
        )
        
        # Combine all outputs
        outputs = {
            'pooled_features': pooled_features,
            'modality_weights': fusion_output['modality_weights'],
            'attention_weights': fusion_output['attention_weights'],
            'qa_attention': qa_attention,
            'confidence': answer_output['confidence']
        }
        
        # Add task-specific outputs
        if task_type == "classification":
            outputs.update({
                'answer_logits': answer_output['answer_logits'],
                'answer_probs': answer_output['answer_probs']
            })
        else:
            outputs.update({
                'generation_logits': answer_output['generation_logits'],
                'generation_probs': answer_output['generation_probs']
            })
        
        # Compute loss if labels provided
        if answer_labels is not None:
            outputs['loss'] = self._compute_loss(
                answer_output, answer_labels, task_type
            )
            
        return outputs
    
    def _compute_loss(
        self,
        answer_output: Dict[str, torch.Tensor],
        answer_labels: torch.Tensor,
        task_type: str
    ) -> torch.Tensor:
        """
        Compute combined loss function (Equation 7 from paper)
        """
        if task_type == "classification":
            answer_loss = F.cross_entropy(
                answer_output['answer_logits'], 
                answer_labels
            )
        else:
            answer_loss = F.cross_entropy(
                answer_output['generation_logits'].view(-1, answer_output['generation_logits'].size(-1)),
                answer_labels.view(-1)
            )
        
        # Confidence regularization
        confidence_loss = F.mse_loss(
            answer_output['confidence'].squeeze(),
            torch.ones_like(answer_output['confidence'].squeeze())
        )
        
        # Combined loss with weights
        total_loss = answer_loss + 0.1 * confidence_loss
        
        return total_loss
    
    def compress_model(self, compression_ratio: float) -> None:
        """
        Apply device-specific model compression
        """
        if compression_ratio < 1.0:
            # Apply parameter pruning or quantization
            for name, param in self.named_parameters():
                if 'weight' in name:
                    # Simple magnitude-based pruning
                    threshold = torch.quantile(torch.abs(param), 1 - compression_ratio)
                    mask = torch.abs(param) >= threshold
                    param.data *= mask.float()
    
    def get_model_size(self) -> int:
        """Get model size in parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class QualityAssuranceFilter:
    """
    Quality assurance filtering based on confidence and consistency scores
    Implements Equation 11 from the paper
    """
    
    def __init__(self, gamma: float = 0.7, threshold: float = 0.85):
        self.gamma = gamma
        self.threshold = threshold
    
    def filter_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        consistency_scores: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Filter predictions based on quality assurance criteria
        
        Args:
            predictions: Model predictions
            consistency_scores: Cross-modal consistency scores
            
        Returns:
            Filtered predictions above quality threshold
        """
        confidence_scores = predictions['confidence']
        
        # Compute quality assurance score (Equation 11)
        qa_scores = (
            self.gamma * confidence_scores + 
            (1 - self.gamma) * consistency_scores
        )
        
        # Filter based on threshold
        valid_mask = qa_scores.squeeze() > self.threshold
        
        filtered_predictions = {}
        for key, value in predictions.items():
            if torch.is_tensor(value):
                filtered_predictions[key] = value[valid_mask]
            else:
                filtered_predictions[key] = value
                
        filtered_predictions['qa_scores'] = qa_scores[valid_mask]
        filtered_predictions['valid_mask'] = valid_mask
        
        return filtered_predictions 