"""
Multimodal Federated Learning Client for Cross-Device Collaboration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import flwr as fl
from collections import defaultdict

from ..models.multimodal_model import MultimodalVideoQAModel
from ..models.feature_extractors import MultimodalFeatureExtractor

logger = logging.getLogger(__name__)

class MultimodalFederatedClient(fl.client.NumPyClient):
    """
    Federated client supporting cross-device multimodal data collaboration.
    """
    
    def __init__(
        self,
        client_id: str,
        device_type: str,  # "mobile", "desktop", "server"
        model: MultimodalVideoQAModel,
        feature_extractor: MultimodalFeatureExtractor,
        train_loader,
        test_loader,
        local_epochs: int = 3,
        learning_rate: float = 1e-4,
        modality_weights: Optional[Dict[str, float]] = None
    ):
        self.client_id = client_id
        self.device_type = device_type
        self.model = model
        self.feature_extractor = feature_extractor
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        
        # Device-specific modality weights
        self.modality_weights = modality_weights or self._get_default_modality_weights()
        
        # Optimizer with different strategies for different devices
        self.optimizer = self._create_device_specific_optimizer(learning_rate)
        
        # Track modality-specific performance
        self.modality_performance = defaultdict(list)
        
    def _get_default_modality_weights(self) -> Dict[str, float]:
        """Get default modality weights based on device type."""
        if self.device_type == "mobile":
            # Mobile devices: prioritize efficiency
            return {"visual": 0.5, "audio": 0.3, "text": 0.2}
        elif self.device_type == "desktop":
            # Desktop: balanced approach
            return {"visual": 0.4, "audio": 0.3, "text": 0.3}
        else:  # server
            # Server: can handle complex processing
            return {"visual": 0.35, "audio": 0.35, "text": 0.3}
    
    def _create_device_specific_optimizer(self, learning_rate: float):
        """Create optimizer based on device capabilities."""
        if self.device_type == "mobile":
            # Lighter optimizer for mobile devices
            return torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            # Full Adam optimizer for more capable devices
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters with device-specific compression."""
        parameters = []
        
        for param in self.model.parameters():
            param_array = param.detach().cpu().numpy()
            
            # Device-specific parameter compression
            if self.device_type == "mobile":
                # Higher compression for mobile devices
                param_array = self._compress_parameters(param_array, compression_ratio=0.5)
            elif self.device_type == "desktop":
                # Moderate compression
                param_array = self._compress_parameters(param_array, compression_ratio=0.8)
            # No compression for servers
            
            parameters.append(param_array)
        
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set parameters with device-specific decompression."""
        decompressed_params = []
        
        for param_array in parameters:
            # Decompress if needed
            if self.device_type in ["mobile", "desktop"]:
                param_array = self._decompress_parameters(param_array)
            decompressed_params.append(param_array)
        
        # Update model parameters
        params_dict = zip(self.model.parameters(), decompressed_params)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Multimodal federated training with cross-device collaboration."""
        
        self.set_parameters(parameters)
        self.model.train()
        
        # Device-specific training metrics
        training_metrics = {
            "device_type": self.device_type,
            "modality_losses": defaultdict(float),
            "modality_accuracies": defaultdict(float),
            "total_loss": 0.0,
            "total_accuracy": 0.0
        }
        
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_metrics = self._train_epoch(epoch)
            
            # Accumulate metrics
            for key, value in epoch_metrics.items():
                if "loss" in key or "accuracy" in key:
                    training_metrics[key] += value
            
            total_samples += epoch_metrics.get("samples", 0)
        
        # Average metrics over epochs
        for key in training_metrics:
            if isinstance(training_metrics[key], (int, float)) and key != "device_type":
                training_metrics[key] /= self.local_epochs
        
        # Add device-specific information
        training_metrics.update({
            "client_id": self.client_id,
            "device_capabilities": self._get_device_capabilities(),
            "modality_weights": self.modality_weights,
            "communication_cost": self._estimate_communication_cost()
        })
        
        return (
            self.get_parameters(config),
            total_samples,
            training_metrics
        )
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with modality-specific tracking."""
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch in self.train_loader:
            # Extract multimodal features
            features = self._extract_multimodal_features(batch)
            
            # Forward pass with modality-specific weighting
            outputs = self.model(
                visual_features=features['visual'],
                audio_features=features['audio'],
                text_features=features['text'],
                question_features=batch['question_features'],
                answer_labels=batch['answer_labels']
            )
            
            # Compute modality-specific losses
            modality_losses = self._compute_modality_losses(outputs, batch)
            
            # Weighted total loss
            total_loss = sum(
                self.modality_weights.get(modality, 1.0) * loss 
                for modality, loss in modality_losses.items()
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Device-specific gradient clipping
            if self.device_type == "mobile":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_metrics["total_loss"] += total_loss.item()
            for modality, loss in modality_losses.items():
                epoch_metrics[f"{modality}_loss"] += loss.item()
            
            # Accuracy calculation
            predictions = torch.argmax(outputs['answer_logits'], dim=-1)
            accuracy = (predictions == batch['answer_labels']).float().mean()
            epoch_metrics["total_accuracy"] += accuracy.item()
            
            num_batches += 1
        
        # Average over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        epoch_metrics["samples"] = len(self.train_loader.dataset)
        
        return dict(epoch_metrics)
    
    def _extract_multimodal_features(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract features based on device capabilities."""
        features = {}
        
        # Visual features
        if 'video_frames' in batch and self._can_process_modality('visual'):
            features['visual'] = self.feature_extractor.clip_extractor.extract_features(
                batch['video_frames']
            )
        else:
            features['visual'] = torch.zeros(1, self.feature_extractor.visual_dim)
        
        # Audio features
        if 'audio_data' in batch and self._can_process_modality('audio'):
            features['audio'] = self.feature_extractor.whisper_extractor.extract_features(
                batch['audio_data']
            )
        else:
            features['audio'] = torch.zeros(1, self.feature_extractor.audio_dim)
        
        # Text features
        if 'text_data' in batch and self._can_process_modality('text'):
            features['text'] = self.feature_extractor.bert_extractor.extract_features(
                batch['text_data']
            )
        else:
            features['text'] = torch.zeros(1, self.feature_extractor.text_dim)
        
        return features
    
    def _can_process_modality(self, modality: str) -> bool:
        """Check if device can process specific modality."""
        device_capabilities = self._get_device_capabilities()
        return device_capabilities.get(f"can_process_{modality}", True)
    
    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get device-specific capabilities."""
        if self.device_type == "mobile":
            return {
                "can_process_visual": True,
                "can_process_audio": True,
                "can_process_text": True,
                "max_video_resolution": "720p",
                "max_audio_duration": 30,  # seconds
                "memory_limit_mb": 2048
            }
        elif self.device_type == "desktop":
            return {
                "can_process_visual": True,
                "can_process_audio": True,
                "can_process_text": True,
                "max_video_resolution": "1080p",
                "max_audio_duration": 120,
                "memory_limit_mb": 8192
            }
        else:  # server
            return {
                "can_process_visual": True,
                "can_process_audio": True,
                "can_process_text": True,
                "max_video_resolution": "4K",
                "max_audio_duration": 600,
                "memory_limit_mb": 32768
            }
    
    def _compute_modality_losses(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses for each modality."""
        return {
            "visual": outputs.get('visual_loss', outputs['loss'] * 0.33),
            "audio": outputs.get('audio_loss', outputs['loss'] * 0.33),
            "text": outputs.get('text_loss', outputs['loss'] * 0.34)
        }
    
    def _compress_parameters(self, param_array: np.ndarray, compression_ratio: float) -> np.ndarray:
        """Simple parameter compression for bandwidth efficiency."""
        if compression_ratio >= 1.0:
            return param_array
        
        # Quantization-based compression
        param_flat = param_array.flatten()
        k = int(len(param_flat) * compression_ratio)
        
        # Keep top-k parameters by magnitude
        indices = np.argsort(np.abs(param_flat))[-k:]
        compressed = np.zeros_like(param_flat)
        compressed[indices] = param_flat[indices]
        
        return compressed.reshape(param_array.shape)
    
    def _decompress_parameters(self, param_array: np.ndarray) -> np.ndarray:
        """Decompress parameters (identity function for simple compression)."""
        return param_array
    
    def _estimate_communication_cost(self) -> float:
        """Estimate communication cost in MB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if self.device_type == "mobile":
            # Higher compression, lower cost
            return total_params * 2 / (1024 * 1024)  # 2 bytes per param (16-bit)
        else:
            # Standard precision
            return total_params * 4 / (1024 * 1024)  # 4 bytes per param (32-bit) 