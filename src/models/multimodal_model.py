"""
Multimodal Video Question Answering Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from .attention_mechanisms import DynamicAttentionFusion, TemporalAttention, QuestionAwareAttention
from .feature_extractors import MultimodalFeatureExtractor

logger = logging.getLogger(__name__)

class MultimodalVideoQAModel(nn.Module):
    """
    Main multimodal video question answering model with dynamic attention fusion.
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        audio_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_classes: int = 1000,  # For answer classification
        dropout: float = 0.1,
        alpha_v: float = 0.4,
        alpha_a: float = 0.3,
        alpha_t: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature extractor (can be used externally)
        self.feature_extractor = None  # Will be set externally
        
        # Dynamic attention fusion
        self.attention_fusion = DynamicAttentionFusion(
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            alpha_v=alpha_v,
            alpha_a=alpha_a,
            alpha_t=alpha_t
        )
        
        # Temporal attention for video sequences
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads, dropout)
        
        # Question-aware attention
        self.question_attention = QuestionAwareAttention(hidden_dim, num_heads, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Answer generation/classification heads
        self.answer_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Answer generation (for open-ended questions)
        self.answer_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim)  # Generate text embeddings
        )
        
        # Question type classifier
        self.question_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 4)  # factual, conceptual, analytical, other
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
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        question_features: torch.Tensor,
        answer_labels: Optional[torch.Tensor] = None,
        task_type: str = "classification"  # "classification" or "generation"
    ) -> Dict[str, torch.Tensor]:
        
        # Multimodal fusion with dynamic attention
        fusion_output = self.attention_fusion(
            visual_features, audio_features, text_features
        )
        
        fused_features = fusion_output['fused_features']  # (batch_size, hidden_dim)
        
        # Expand for sequence processing if needed
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Temporal attention (if we have video sequences)
        temporal_output, temporal_attn = self.temporal_attention(fused_features)
        
        # Question-aware attention
        question_aware_output, qa_attn = self.question_attention(
            temporal_output, question_features
        )
        
        # Transformer encoding
        encoded_features = self.transformer_encoder(question_aware_output)
        
        # Pool the sequence (mean pooling)
        pooled_features = encoded_features.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Generate outputs based on task type
        outputs = {
            'pooled_features': pooled_features,
            'attention_weights': fusion_output['attention_weights'],
            'modality_weights': fusion_output['modality_weights'],
            'temporal_attention': temporal_attn,
            'qa_attention': qa_attn
        }
        
        if task_type == "classification":
            # Answer classification
            answer_logits = self.answer_classifier(pooled_features)
            outputs['answer_logits'] = answer_logits
            
            if answer_labels is not None:
                loss = F.cross_entropy(answer_logits, answer_labels)
                outputs['loss'] = loss
                
        elif task_type == "generation":
            # Answer generation
            generated_embeddings = self.answer_generator(pooled_features)
            outputs['generated_embeddings'] = generated_embeddings
            
            if answer_labels is not None:
                # MSE loss for embedding generation
                loss = F.mse_loss(generated_embeddings, answer_labels)
                outputs['loss'] = loss
        
        # Question type classification
        question_type_logits = self.question_type_classifier(pooled_features)
        outputs['question_type_logits'] = question_type_logits
        
        # Confidence estimation
        confidence = self.confidence_estimator(pooled_features)
        outputs['confidence'] = confidence
        
        return outputs
    
    def predict(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        question_features: torch.Tensor,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """Make predictions without computing loss."""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                visual_features, audio_features, text_features, 
                question_features, task_type=task_type
            )
            
            predictions = {
                'confidence': outputs['confidence'].cpu().numpy(),
                'modality_weights': outputs['modality_weights'].cpu().numpy(),
            }
            
            if task_type == "classification":
                answer_probs = F.softmax(outputs['answer_logits'], dim=-1)
                predicted_answers = torch.argmax(answer_probs, dim=-1)
                
                predictions.update({
                    'answer_probabilities': answer_probs.cpu().numpy(),
                    'predicted_answers': predicted_answers.cpu().numpy(),
                })
                
            elif task_type == "generation":
                predictions['generated_embeddings'] = outputs['generated_embeddings'].cpu().numpy()
            
            # Question type prediction
            question_type_probs = F.softmax(outputs['question_type_logits'], dim=-1)
            predicted_question_types = torch.argmax(question_type_probs, dim=-1)
            
            predictions.update({
                'question_type_probabilities': question_type_probs.cpu().numpy(),
                'predicted_question_types': predicted_question_types.cpu().numpy(),
            })
            
        return predictions
    
    def get_attention_maps(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        question_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract attention maps for visualization."""
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                visual_features, audio_features, text_features, question_features
            )
            
            attention_maps = {
                'multimodal_attention': outputs['attention_weights'],
                'temporal_attention': outputs['temporal_attention'],
                'question_attention': outputs['qa_attention'],
                'modality_weights': outputs['modality_weights']
            }
            
        return attention_maps
    
    def compute_feature_importance(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        question_features: torch.Tensor
    ) -> Dict[str, float]:
        """Compute feature importance scores."""
        self.eval()
        
        # Baseline prediction
        baseline_output = self.forward(
            visual_features, audio_features, text_features, question_features
        )
        baseline_confidence = baseline_output['confidence'].item()
        
        # Zero out each modality and measure impact
        zero_visual = torch.zeros_like(visual_features)
        zero_audio = torch.zeros_like(audio_features)
        zero_text = torch.zeros_like(text_features)
        
        with torch.no_grad():
            # Impact of removing visual features
            no_visual_output = self.forward(
                zero_visual, audio_features, text_features, question_features
            )
            visual_importance = baseline_confidence - no_visual_output['confidence'].item()
            
            # Impact of removing audio features
            no_audio_output = self.forward(
                visual_features, zero_audio, text_features, question_features
            )
            audio_importance = baseline_confidence - no_audio_output['confidence'].item()
            
            # Impact of removing text features
            no_text_output = self.forward(
                visual_features, audio_features, zero_text, question_features
            )
            text_importance = baseline_confidence - no_text_output['confidence'].item()
        
        return {
            'visual_importance': visual_importance,
            'audio_importance': audio_importance,
            'text_importance': text_importance,
            'baseline_confidence': baseline_confidence
        } 