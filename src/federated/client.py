"""
Federated Learning Client for FedVideoQA
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import pickle
import gzip
from collections import OrderedDict
import flwr as fl

from ..models.multimodal_model import MultimodalVideoQAModel
from ..privacy.differential_privacy import DifferentialPrivacyManager
from ..utils.data_loader import VideoQADataLoader
from ..utils.metrics import compute_qa_metrics

logger = logging.getLogger(__name__)

class FedVideoQAClient(fl.client.NumPyClient):
    """
    Federated learning client for VideoQA training.
    Implements privacy-preserving local training with gradient sparsification.
    """
    
    def __init__(
        self,
        client_id: str,
        model: MultimodalVideoQAModel,
        data_loader: VideoQADataLoader,
        privacy_manager: DifferentialPrivacyManager,
        local_epochs: int = 5,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        gradient_sparsification_threshold: float = 0.01,
        max_communication_size_mb: int = 100
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.data_loader = data_loader
        self.privacy_manager = privacy_manager
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.gradient_sparsification_threshold = gradient_sparsification_threshold
        self.max_communication_size_mb = max_communication_size_mb
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=local_epochs
        )
        
        # Training metrics
        self.training_history = []
        
        logger.info(f"Initialized FedVideoQA client {client_id}")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters for federated aggregation."""
        parameters = []
        
        for param in self.model.parameters():
            param_array = param.detach().cpu().numpy()
            
            # Apply gradient sparsification
            if self.gradient_sparsification_threshold > 0:
                param_array = self._sparsify_gradients(param_array)
            
            parameters.append(param_array)
        
        # Compress parameters if needed
        compressed_params = self._compress_parameters(parameters)
        
        logger.info(f"Client {self.client_id}: Sending {len(compressed_params)} parameters")
        return compressed_params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from federated aggregation."""
        # Decompress parameters if needed
        decompressed_params = self._decompress_parameters(parameters)
        
        params_dict = zip(self.model.parameters(), decompressed_params)
        
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
        
        logger.info(f"Client {self.client_id}: Updated model parameters")
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train the model locally with differential privacy."""
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Extract training configuration
        round_num = config.get("round", 0)
        local_epochs = config.get("local_epochs", self.local_epochs)
        
        logger.info(f"Client {self.client_id}: Starting local training for round {round_num}")
        
        # Local training
        train_metrics = self._local_training(local_epochs, round_num)
        
        # Get updated parameters with privacy protection
        updated_parameters = self.get_parameters(config)
        
        # Calculate dataset size
        dataset_size = len(self.data_loader.train_dataset)
        
        # Prepare metrics for server
        metrics = {
            "client_id": self.client_id,
            "round": round_num,
            "dataset_size": dataset_size,
            "train_loss": train_metrics["train_loss"],
            "train_accuracy": train_metrics["train_accuracy"],
            "privacy_budget_used": self.privacy_manager.get_privacy_budget_used(),
            "communication_size_mb": self._estimate_communication_size(updated_parameters)
        }
        
        return updated_parameters, dataset_size, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate the model locally."""
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate on local test set
        eval_metrics = self._local_evaluation()
        
        dataset_size = len(self.data_loader.test_dataset)
        
        metrics = {
            "client_id": self.client_id,
            "dataset_size": dataset_size,
            "test_accuracy": eval_metrics["test_accuracy"],
            "test_loss": eval_metrics["test_loss"],
            "confidence_score": eval_metrics["confidence_score"],
            "modality_weights": eval_metrics["modality_weights"]
        }
        
        return eval_metrics["test_loss"], dataset_size, metrics
    
    def _local_training(self, epochs: int, round_num: int) -> Dict[str, float]:
        """Perform local training with differential privacy."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, batch in enumerate(self.data_loader.train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    visual_features=batch['visual_features'],
                    audio_features=batch['audio_features'],
                    text_features=batch['text_features'],
                    question_features=batch['question_features'],
                    answer_labels=batch['answer_labels'],
                    task_type="classification"
                )
                
                loss = outputs['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Apply differential privacy
                if self.privacy_manager:
                    self.privacy_manager.add_noise_to_gradients(self.model)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                predictions = torch.argmax(outputs['answer_logits'], dim=-1)
                epoch_correct += (predictions == batch['answer_labels']).sum().item()
                epoch_samples += batch['answer_labels'].size(0)
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(
                        f"Client {self.client_id}, Epoch {epoch+1}/{epochs}, "
                        f"Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )
            
            # Update learning rate
            self.scheduler.step()
            
            # Epoch metrics
            epoch_loss /= len(self.data_loader.train_loader)
            epoch_accuracy = epoch_correct / epoch_samples
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            logger.info(
                f"Client {self.client_id}, Epoch {epoch+1}/{epochs}: "
                f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
            )
        
        # Average metrics over all epochs
        avg_loss = total_loss / epochs
        avg_accuracy = total_correct / total_samples
        
        # Store training history
        self.training_history.append({
            "round": round_num,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "privacy_budget": self.privacy_manager.get_privacy_budget_used()
        })
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy
        }
    
    def _local_evaluation(self) -> Dict[str, float]:
        """Evaluate model on local test set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        confidence_scores = []
        modality_weights_list = []
        
        with torch.no_grad():
            for batch in self.data_loader.test_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    visual_features=batch['visual_features'],
                    audio_features=batch['audio_features'],
                    text_features=batch['text_features'],
                    question_features=batch['question_features'],
                    answer_labels=batch['answer_labels'],
                    task_type="classification"
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Predictions
                predictions = torch.argmax(outputs['answer_logits'], dim=-1)
                total_correct += (predictions == batch['answer_labels']).sum().item()
                total_samples += batch['answer_labels'].size(0)
                
                # Confidence scores
                confidence_scores.extend(outputs['confidence'].cpu().numpy().flatten())
                
                # Modality weights
                modality_weights_list.append(outputs['modality_weights'].cpu().numpy())
        
        avg_loss = total_loss / len(self.data_loader.test_loader)
        accuracy = total_correct / total_samples
        avg_confidence = np.mean(confidence_scores)
        avg_modality_weights = np.mean(np.vstack(modality_weights_list), axis=0)
        
        return {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "confidence_score": avg_confidence,
            "modality_weights": avg_modality_weights.tolist()
        }
    
    def _sparsify_gradients(self, param_array: np.ndarray) -> np.ndarray:
        """Apply gradient sparsification to reduce communication overhead."""
        if self.gradient_sparsification_threshold <= 0:
            return param_array
        
        # Calculate threshold based on gradient magnitude
        threshold = np.percentile(np.abs(param_array), 
                                 (1 - self.gradient_sparsification_threshold) * 100)
        
        # Zero out small gradients
        mask = np.abs(param_array) >= threshold
        sparsified = param_array * mask
        
        return sparsified
    
    def _compress_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Compress parameters to reduce communication overhead."""
        compressed = []
        
        for param in parameters:
            # Simple compression: quantization to 16-bit
            param_compressed = param.astype(np.float16)
            compressed.append(param_compressed)
        
        return compressed
    
    def _decompress_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Decompress parameters."""
        decompressed = []
        
        for param in parameters:
            # Convert back to 32-bit
            param_decompressed = param.astype(np.float32)
            decompressed.append(param_decompressed)
        
        return decompressed
    
    def _estimate_communication_size(self, parameters: List[np.ndarray]) -> float:
        """Estimate communication size in MB."""
        total_bytes = sum(param.nbytes for param in parameters)
        size_mb = total_bytes / (1024 * 1024)
        return size_mb
    
    def save_client_state(self, save_path: str) -> None:
        """Save client state for resuming training."""
        state = {
            "client_id": self.client_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "privacy_budget_used": self.privacy_manager.get_privacy_budget_used()
        }
        
        torch.save(state, save_path)
        logger.info(f"Client {self.client_id}: State saved to {save_path}")
    
    def load_client_state(self, load_path: str) -> None:
        """Load client state for resuming training."""
        state = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.training_history = state["training_history"]
        
        logger.info(f"Client {self.client_id}: State loaded from {load_path}") 