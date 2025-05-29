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
    简化的多模态视频问答模型 - 仅支持文本生成
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
        alpha_v: float = 0.4,
        alpha_a: float = 0.3,
        alpha_t: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        
        # 保留多模态融合组件
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
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads, dropout)
        self.question_attention = QuestionAwareAttention(hidden_dim, num_heads, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers
        )
        
        # 只保留答案生成器
        self.answer_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim)  # 生成文本嵌入
        )
        
        # 可选：保留置信度估计器
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
        answer_embeddings: Optional[torch.Tensor] = None  # 目标答案嵌入
    ) -> Dict[str, torch.Tensor]:
        
        # 多模态融合
        fusion_output = self.attention_fusion(visual_features, audio_features, text_features)
        fused_features = fusion_output['fused_features']
        
        # 时序和问题感知注意力
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)
        
        temporal_output, temporal_attn = self.temporal_attention(fused_features)
        question_aware_output, qa_attn = self.question_attention(temporal_output, question_features)
        
        # Transformer编码
        encoded_features = self.transformer_encoder(question_aware_output)
        pooled_features = encoded_features.mean(dim=1)
        
        # 生成答案嵌入
        generated_embeddings = self.answer_generator(pooled_features)
        
        outputs = {
            'generated_embeddings': generated_embeddings,
            'pooled_features': pooled_features,
            'attention_weights': fusion_output['attention_weights'],
            'modality_weights': fusion_output['modality_weights'],
            'temporal_attention': temporal_attn,
            'qa_attention': qa_attn
        }
        
        # 计算损失（如果提供目标嵌入）
        if answer_embeddings is not None:
            # 使用余弦相似度损失或MSE损失
            loss = F.mse_loss(generated_embeddings, answer_embeddings)
            outputs['loss'] = loss
        
        # 置信度估计
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