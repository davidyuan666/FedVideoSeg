"""
Enhanced Federated Learning Client for FedVideoQA
Implements device-aware federated learning from Section 3.4
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import flwr as fl
import torch.nn.functional as F
import hashlib
import time

from ..models.fedvideo_qa_model import FedVideoQAModel, DeviceCapabilityConfig
from ..core.binary_search_localizer import BinarySearchLocalizer
from ..privacy.differential_privacy import DifferentialPrivacyManager

logger = logging.getLogger(__name__)

class FedVideoQAClient(fl.client.NumPyClient):
    """
    Enhanced federated learning client implementing the FedVideoQA methodology
    """
    
    def __init__(
        self,
        client_id: str,
        model: FedVideoQAModel,
        localizer: BinarySearchLocalizer,
        train_loader,
        test_loader,
        device_type: str = "desktop",
        local_epochs: int = 5,
        learning_rate: float = 1e-4,
        privacy_manager: Optional[DifferentialPrivacyManager] = None
    ):
        self.client_id = client_id
        self.model = model
        self.localizer = localizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device_type = device_type
        self.local_epochs = local_epochs
        self.privacy_manager = privacy_manager
        
        # Get device configuration
        if device_type == "mobile":
            self.device_config = DeviceCapabilityConfig.MOBILE
        elif device_type == "desktop":
            self.device_config = DeviceCapabilityConfig.DESKTOP
        else:
            self.device_config = DeviceCapabilityConfig.SERVER
        
        # Device-specific optimizer
        self.optimizer = self._create_device_optimizer(learning_rate)
        
        # Training metrics
        self.training_history = []
        self.communication_costs = []
        
        logger.info(f"Initialized FedVideoQA client {client_id} on {device_type}")
    
    def _create_device_optimizer(self, learning_rate: float):
        """Create device-specific optimizer"""
        if self.device_type == "mobile":
            # Lighter optimizer for mobile devices
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            # Full AdamW optimizer for desktop/server
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get model parameters with device-specific compression
        Implements gradient sparsification from Equation 9
        """
        parameters = []
        
        for param in self.model.parameters():
            param_array = param.detach().cpu().numpy()
            
            # Apply gradient sparsification (Equation 9)
            sparsification_ratio = self.device_config['sparsification_ratio']
            if sparsification_ratio < 1.0:
                param_array = self._apply_gradient_sparsification(
                    param_array, sparsification_ratio
                )
            
            # Apply device-specific compression
            compression_ratio = self.device_config['compression_ratio']
            if compression_ratio < 1.0:
                param_array = self._compress_parameters(param_array, compression_ratio)
            
            parameters.append(param_array)
        
        # Estimate communication cost
        comm_cost = self._estimate_communication_cost(parameters)
        self.communication_costs.append(comm_cost)
        
        logger.info(
            f"Client {self.client_id}: Sending {len(parameters)} parameters "
            f"({comm_cost:.2f} MB)"
        )
        
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from federated aggregation"""
        # Decompress parameters if needed
        decompressed_params = []
        compression_ratio = self.device_config['compression_ratio']
        
        for param_array in parameters:
            if compression_ratio < 1.0:
                param_array = self._decompress_parameters(param_array)
            decompressed_params.append(param_array)
        
        # Update model parameters
        params_dict = zip(self.model.parameters(), decompressed_params)
        for param, new_param in params_dict:
            param.data = torch.tensor(
                new_param, dtype=param.dtype, device=param.device
            )
        
        logger.info(f"Client {self.client_id}: Updated model parameters")
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Local training with adaptive multimodal processing
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Extract training configuration
        round_num = config.get("round", 0)
        local_epochs = config.get("local_epochs", self.local_epochs)
        
        logger.info(
            f"Client {self.client_id}: Starting local training for round {round_num}"
        )
        
        # Local training with device-aware processing
        train_metrics = self._local_training(local_epochs, round_num)
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        # Calculate dataset size
        dataset_size = len(self.train_loader.dataset)
        
        # Prepare metrics for server
        metrics = {
            "client_id": self.client_id,
            "device_type": self.device_type,
            "round": round_num,
            "dataset_size": dataset_size,
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics["train_accuracy"],
            "device_capability": self._calculate_device_capability(),
            "communication_cost_mb": self.communication_costs[-1] if self.communication_costs else 0,
            "processing_time": train_metrics["processing_time"]
        }
        
        if self.privacy_manager:
            metrics["privacy_budget_used"] = self.privacy_manager.get_privacy_budget_used()
        
        return updated_parameters, dataset_size, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model on local test set"""
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate with device-aware processing
        eval_metrics = self._local_evaluation()
        
        dataset_size = len(self.test_loader.dataset)
        
        metrics = {
            "client_id": self.client_id,
            "device_type": self.device_type,
            "dataset_size": dataset_size,
            "test_accuracy": eval_metrics["test_accuracy"],
            "test_loss": eval_metrics["test_loss"],
            "confidence_score": eval_metrics["confidence_score"],
            "modality_weights": eval_metrics["modality_weights"],
            "inference_time": eval_metrics["inference_time"]
        }
        
        return eval_metrics["test_loss"], dataset_size, metrics
    
    def _local_training(self, epochs: int, round_num: int) -> Dict[str, float]:
        """
        Perform local training with device-aware multimodal processing
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Process batch with device-aware feature extraction
                processed_batch = self._process_batch_device_aware(batch)
                
                # Forward pass
                outputs = self.model(
                    visual_features=processed_batch['visual_features'],
                    audio_features=processed_batch['audio_features'],
                    text_features=processed_batch['text_features'],
                    question_features=processed_batch['question_features'],
                    task_type=processed_batch.get('task_type', 'classification'),
                    answer_labels=processed_batch.get('answer_labels')
                )
                
                loss = outputs['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.privacy_manager:
                    self.privacy_manager.add_noise_to_gradients(self.model)
                
                # Device-specific gradient clipping
                max_norm = 0.5 if self.device_type == "mobile" else 1.0
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                
                # Calculate accuracy based on task type
                if 'answer_logits' in outputs:
                    predictions = torch.argmax(outputs['answer_logits'], dim=-1)
                    correct = (predictions == processed_batch['answer_labels']).sum().item()
                    epoch_correct += correct
                
                epoch_samples += processed_batch['answer_labels'].size(0)
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(
                        f"Client {self.client_id}, Epoch {epoch+1}/{epochs}, "
                        f"Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )
            
            # Epoch metrics
            epoch_loss /= len(self.train_loader)
            epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            logger.info(
                f"Client {self.client_id}, Epoch {epoch+1}/{epochs}: "
                f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
            )
        
        processing_time = time.time() - start_time
        
        # Average metrics over all epochs
        avg_loss = total_loss / epochs
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Store training history
        self.training_history.append({
            "round": round_num,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "processing_time": processing_time,
            "device_type": self.device_type
        })
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "processing_time": processing_time
        }
    
    def _local_evaluation(self) -> Dict[str, float]:
        """Evaluate model on local test set with device-aware processing"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        confidence_scores = []
        modality_weights_list = []
        start_time = time.time()
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Process batch with device-aware feature extraction
                processed_batch = self._process_batch_device_aware(batch)
                
                # Forward pass
                outputs = self.model(
                    visual_features=processed_batch['visual_features'],
                    audio_features=processed_batch['audio_features'],
                    text_features=processed_batch['text_features'],
                    question_features=processed_batch['question_features'],
                    task_type=processed_batch.get('task_type', 'classification'),
                    answer_labels=processed_batch.get('answer_labels')
                )
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                
                # Accuracy calculation
                if 'answer_logits' in outputs:
                    predictions = torch.argmax(outputs['answer_logits'], dim=-1)
                    correct = (predictions == processed_batch['answer_labels']).sum().item()
                    total_correct += correct
                
                total_samples += processed_batch['answer_labels'].size(0)
                
                # Collect confidence scores and modality weights
                confidence_scores.extend(outputs['confidence'].cpu().numpy().flatten())
                modality_weights_list.append(outputs['modality_weights'].cpu().numpy())
        
        inference_time = time.time() - start_time
        
        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        avg_modality_weights = np.mean(np.vstack(modality_weights_list), axis=0) if modality_weights_list else [0.33, 0.33, 0.34]
        
        return {
            "test_loss": avg_loss,
            "test_accuracy": avg_accuracy,
            "confidence_score": avg_confidence,
            "modality_weights": avg_modality_weights.tolist(),
            "inference_time": inference_time
        }
    
    def _process_batch_device_aware(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process batch with device-aware feature extraction and compression
        """
        processed_batch = {}
        
        # Apply device capability scaling to features
        device_caps = self.device_config
        
        if 'visual_features' in batch:
            visual_features = batch['visual_features']
            if device_caps['visual'] < 1.0:
                # Reduce visual feature complexity for limited devices
                visual_features = self._reduce_feature_complexity(
                    visual_features, device_caps['visual']
                )
            processed_batch['visual_features'] = visual_features
        
        if 'audio_features' in batch:
            audio_features = batch['audio_features']
            if device_caps['audio'] < 1.0:
                audio_features = self._reduce_feature_complexity(
                    audio_features, device_caps['audio']
                )
            processed_batch['audio_features'] = audio_features
        
        if 'text_features' in batch:
            text_features = batch['text_features']
            if device_caps['text'] < 1.0:
                text_features = self._reduce_feature_complexity(
                    text_features, device_caps['text']
                )
            processed_batch['text_features'] = text_features
        
        # Copy other batch elements
        for key in ['question_features', 'answer_labels', 'task_type']:
            if key in batch:
                processed_batch[key] = batch[key]
        
        return processed_batch
    
    def _reduce_feature_complexity(
        self, 
        features: torch.Tensor, 
        capability_ratio: float
    ) -> torch.Tensor:
        """Reduce feature complexity based on device capability"""
        if capability_ratio >= 1.0:
            return features
        
        # Simple feature reduction: subsample or compress
        if features.dim() >= 2:
            # Reduce sequence length for temporal features
            seq_len = features.size(1)
            new_seq_len = max(1, int(seq_len * capability_ratio))
            indices = torch.linspace(0, seq_len - 1, new_seq_len).long()
            features = features[:, indices]
        
        return features
    
    def _apply_gradient_sparsification(
        self, 
        param_array: np.ndarray, 
        sparsification_ratio: float
    ) -> np.ndarray:
        """
        Apply gradient sparsification (Equation 9)
        TopK selection of most important gradients
        """
        if sparsification_ratio >= 1.0:
            return param_array
        
        # Flatten array for TopK selection
        flat_array = param_array.flatten()
        k = max(1, int(len(flat_array) * sparsification_ratio))
        
        # Get TopK indices by magnitude
        indices = np.argpartition(np.abs(flat_array), -k)[-k:]
        
        # Create sparse array
        sparse_flat = np.zeros_like(flat_array)
        sparse_flat[indices] = flat_array[indices]
        
        return sparse_flat.reshape(param_array.shape)
    
    def _compress_parameters(
        self, 
        param_array: np.ndarray, 
        compression_ratio: float
    ) -> np.ndarray:
        """Apply parameter compression"""
        if compression_ratio >= 1.0:
            return param_array
        
        # Simple quantization-based compression
        if compression_ratio >= 0.5:
            # 16-bit quantization
            return param_array.astype(np.float16)
        else:
            # 8-bit quantization (more aggressive)
            min_val, max_val = param_array.min(), param_array.max()
            scale = (max_val - min_val) / 255.0
            quantized = np.round((param_array - min_val) / scale).astype(np.uint8)
            return quantized
    
    def _decompress_parameters(self, param_array: np.ndarray) -> np.ndarray:
        """Decompress parameters"""
        if param_array.dtype == np.float16:
            return param_array.astype(np.float32)
        elif param_array.dtype == np.uint8:
            # Need to store scale and min_val for proper decompression
            # For now, just convert back to float32
            return param_array.astype(np.float32) / 255.0
        else:
            return param_array
    
    def _estimate_communication_cost(self, parameters: List[np.ndarray]) -> float:
        """Estimate communication cost in MB"""
        total_bytes = sum(param.nbytes for param in parameters)
        return total_bytes / (1024 * 1024)
    
    def _calculate_device_capability(self) -> float:
        """
        Calculate overall device capability score
        Implements κ_i calculation from Equation 8
        """
        capabilities = [
            self.device_config['visual'],
            self.device_config['audio'],
            self.device_config['text']
        ]
        compression_ratio = self.device_config['compression_ratio']
        
        # Equation 8: κ_i = (1/3) * Σ(c_i^m * ρ_i)
        capability_score = (1/3) * sum(capabilities) * compression_ratio
        
        return capability_score
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get comprehensive client statistics"""
        return {
            "client_id": self.client_id,
            "device_type": self.device_type,
            "device_config": self.device_config,
            "training_history": self.training_history,
            "communication_costs": self.communication_costs,
            "total_parameters": self.model.get_model_size(),
            "trainable_parameters": self.model.get_trainable_parameters(),
            "device_capability_score": self._calculate_device_capability()
        } 