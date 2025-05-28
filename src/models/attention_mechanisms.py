"""
Dynamic Attention Mechanisms for Multimodal Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self._attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def _attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing different modalities."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        query_modality: torch.Tensor, 
        key_value_modality: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross-modal attention
        attended, attention_weights = self.attention(
            query_modality, key_value_modality, key_value_modality, mask
        )
        
        # Residual connection and normalization
        output = self.norm1(query_modality + attended)
        
        # Feed-forward network
        ffn_output = self.ffn(output)
        output = self.norm2(output + ffn_output)
        
        return output, attention_weights

class DynamicAttentionFusion(nn.Module):
    """
    Dynamic attention fusion mechanism with learnable weights (α_v, α_a, α_t).
    """
    
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha_v: float = 0.4,
        alpha_a: float = 0.3,
        alpha_t: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Projection layers to common dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.visual_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.audio_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.text_cross_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Self-attention for each modality
        self.visual_self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.audio_self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.text_self_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Dynamic weight learning
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Initialize static weights
        self.register_buffer('static_weights', torch.tensor([alpha_v, alpha_a, alpha_t]))
        
        # Final fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        use_dynamic_weights: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size = max(
            visual_features.size(0) if visual_features.numel() > 0 else 0,
            audio_features.size(0) if audio_features.numel() > 0 else 0,
            text_features.size(0) if text_features.numel() > 0 else 0
        )
        
        # Handle empty features
        if visual_features.numel() == 0:
            visual_features = torch.zeros(batch_size, 1, visual_features.size(-1)).to(visual_features.device)
        if audio_features.numel() == 0:
            audio_features = torch.zeros(batch_size, 1, audio_features.size(-1)).to(audio_features.device)
        if text_features.numel() == 0:
            text_features = torch.zeros(batch_size, 1, text_features.size(-1)).to(text_features.device)
        
        # Ensure 3D tensors (batch_size, seq_len, feature_dim)
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # Project to common dimension
        visual_proj = self.visual_proj(visual_features)
        audio_proj = self.audio_proj(audio_features)
        text_proj = self.text_proj(text_features)
        
        # Self-attention for each modality
        visual_self, visual_self_attn = self.visual_self_attn(visual_proj, visual_proj, visual_proj)
        audio_self, audio_self_attn = self.audio_self_attn(audio_proj, audio_proj, audio_proj)
        text_self, text_self_attn = self.text_self_attn(text_proj, text_proj, text_proj)
        
        # Cross-modal attention
        # Visual attending to audio and text
        visual_audio, va_attn = self.visual_cross_attn(visual_self, audio_self)
        visual_text, vt_attn = self.visual_cross_attn(visual_audio, text_self)
        
        # Audio attending to visual and text
        audio_visual, av_attn = self.audio_cross_attn(audio_self, visual_self)
        audio_text, at_attn = self.audio_cross_attn(audio_visual, text_self)
        
        # Text attending to visual and audio
        text_visual, tv_attn = self.text_cross_attn(text_self, visual_self)
        text_audio, ta_attn = self.text_cross_attn(text_visual, audio_self)
        
        # Pool features (mean pooling over sequence dimension)
        visual_pooled = visual_text.mean(dim=1)  # (batch_size, hidden_dim)
        audio_pooled = audio_text.mean(dim=1)
        text_pooled = text_audio.mean(dim=1)
        
        # Dynamic weight computation
        if use_dynamic_weights:
            # Concatenate all modality features
            combined_features = torch.cat([visual_pooled, audio_pooled, text_pooled], dim=-1)
            dynamic_weights = self.weight_network(combined_features)  # (batch_size, 3)
            
            # Apply dynamic weights
            weighted_visual = visual_pooled * dynamic_weights[:, 0:1]
            weighted_audio = audio_pooled * dynamic_weights[:, 1:2]
            weighted_text = text_pooled * dynamic_weights[:, 2:3]
        else:
            # Use static weights
            weighted_visual = visual_pooled * self.static_weights[0]
            weighted_audio = audio_pooled * self.static_weights[1]
            weighted_text = text_pooled * self.static_weights[2]
        
        # Fusion
        fused_features = torch.cat([weighted_visual, weighted_audio, weighted_text], dim=-1)
        fused_output = self.fusion_layer(fused_features)
        
        # Final output projection
        output = self.output_proj(fused_output)
        
        return {
            'fused_features': output,
            'visual_features': visual_pooled,
            'audio_features': audio_pooled,
            'text_features': text_pooled,
            'attention_weights': {
                'visual_self': visual_self_attn,
                'audio_self': audio_self_attn,
                'text_self': text_self_attn,
                'visual_audio': va_attn,
                'visual_text': vt_attn,
                'audio_visual': av_attn,
                'audio_text': at_attn,
                'text_visual': tv_attn,
                'text_audio': ta_attn
            },
            'modality_weights': dynamic_weights if use_dynamic_weights else self.static_weights.unsqueeze(0).expand(batch_size, -1)
        }

class TemporalAttention(nn.Module):
    """Temporal attention for video sequences."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x_with_pos = x + pos_enc
        
        # Self-attention
        attended, attention_weights = self.attention(x_with_pos, x_with_pos, x_with_pos, mask)
        
        # Residual connection and normalization
        output = self.norm(x + attended)
        
        return output, attention_weights

class QuestionAwareAttention(nn.Module):
    """Question-aware attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = CrossModalAttention(d_model, num_heads, dropout)
        self.question_encoder = nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)
        
    def forward(
        self, 
        video_features: torch.Tensor, 
        question_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode question
        question_encoded, _ = self.question_encoder(question_features)
        
        # Question-aware video attention
        attended_video, attention_weights = self.cross_attention(
            video_features, question_encoded
        )
        
        return attended_video, attention_weights 