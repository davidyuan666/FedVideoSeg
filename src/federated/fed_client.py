"""
Enhanced Federated Learning Client
Supports Qwen2.5-VL fine-tuning with improved features for frame relevance classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from ..models.qwen_finetuner import SimpleQwenFrameClassifier

logger = logging.getLogger(__name__)

class FrameRelevanceDataset(Dataset):
    """Enhanced dataset for frame-question pairs with preprocessing"""
    
    def __init__(self, data: List[Tuple], transform=None):
        self.data = data  # List of (frame, question, label)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame, question, label = self.data[idx]
        
        # Apply transforms if specified
        if self.transform:
            frame = self.transform(frame)
            
        return frame, question, torch.tensor(label, dtype=torch.long)

class FedClient(fl.client.NumPyClient):
    """
    Enhanced federated client for Qwen2.5-VL fine-tuning
    Supports multiple training modes and device-specific optimizations
    """
    
    def __init__(self, 
                 client_id: str, 
                 local_data: List[Tuple], 
                 device_type: str = "desktop",
                 training_mode: str = "finetune",
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 tuner_configs: Optional[Dict] = None):
        
        self.client_id = client_id
        self.device_type = device_type
        self.training_mode = training_mode
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Device-specific configurations
        self.batch_size = self._get_batch_size()
        self.accumulation_steps = self._get_accumulation_steps()
        
        # Initialize model with training mode
        self.model = SimpleQwenFrameClassifier(
            model_name=model_name, 
            training_mode=training_mode
        )
        self.model.to(self.device)
        
        # Set up mixed precision training for better performance
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Prepare data
        self.dataset = FrameRelevanceDataset(local_data)
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 if device_type != "mobile" else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Enhanced optimizer with weight decay and scheduler
        self.optimizer = self._setup_optimizer(tuner_configs)
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training statistics
        self.training_stats = {
            "total_epochs": 0,
            "total_batches": 0,
            "avg_loss": 0.0,
            "best_accuracy": 0.0
        }
        
        logger.info(f"FedClient {client_id} initialized with {device_type} configuration")
        logger.info(f"Model: {model_name}, Training mode: {training_mode}")
        logger.info(f"Dataset size: {len(self.dataset)}, Batch size: {self.batch_size}")
    
    def _get_batch_size(self) -> int:
        """Get device-appropriate batch size"""
        if self.device_type == "mobile":
            return 2
        elif self.device_type == "edge":
            return 4
        else:  # desktop/server
            return 8 if torch.cuda.is_available() else 4
    
    def _get_accumulation_steps(self) -> int:
        """Get gradient accumulation steps for small batch training"""
        if self.device_type == "mobile":
            return 4
        elif self.device_type == "edge":
            return 2
        else:
            return 1
    
    def _setup_optimizer(self, tuner_configs: Optional[Dict]) -> optim.Optimizer:
        """Setup optimizer with proper parameter filtering"""
        trainable_params = []
        
        # Get trainable parameters based on training mode
        if hasattr(self.model, 'get_trainable_parameters'):
            params_dict = self.model.get_trainable_parameters(self.training_mode)
            trainable_params = list(params_dict.values())
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Different learning rates for different components
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'classifier' in n and p.requires_grad],
                'lr': 2e-4,
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'lora' in n and p.requires_grad],
                'lr': 1e-4,
                'weight_decay': 0.0
            }
        ]
        
        return optim.AdamW(
            param_groups if param_groups[0]['params'] or param_groups[1]['params'] else trainable_params,
            lr=1e-4,
            weight_decay=0.01,
            eps=1e-8
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = len(self.train_loader) * 5  # Assuming 5 epochs max
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=1e-6
        )
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters (only trainable parameters)"""
        params = []
        param_names = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param.detach().cpu().numpy())
                param_names.append(name)
        
        logger.debug(f"Client {self.client_id}: Sending {len(params)} parameter arrays")
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters with error handling"""
        param_idx = 0
        updated_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param_idx < len(parameters):
                    try:
                        new_param = torch.from_numpy(parameters[param_idx]).to(param.device)
                        if new_param.shape == param.shape:
                            param.data.copy_(new_param)
                            updated_params += 1
                        else:
                            logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {new_param.shape}")
                    except Exception as e:
                        logger.error(f"Error setting parameter {name}: {e}")
                    param_idx += 1
        
        logger.debug(f"Client {self.client_id}: Updated {updated_params} parameters")
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Enhanced local training with mixed precision and gradient accumulation"""
        self.set_parameters(parameters)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norm = 0.0
        
        # Training configuration
        epochs = config.get("local_epochs", 3)
        max_grad_norm = config.get("max_grad_norm", 1.0)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            
            for batch_idx, (frames, questions, labels) in enumerate(self.train_loader):
                labels = labels.to(self.device, non_blocking=True)
                
                # Mixed precision training
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(frames, questions, mode=self.training_mode)
                        loss = self.criterion(logits, labels) / self.accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    logits = self.model(frames, questions, mode=self.training_mode)
                    loss = self.criterion(logits, labels) / self.accumulation_steps
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    gradient_norm += grad_norm.item()
                
                # Statistics
                epoch_loss += loss.item() * self.accumulation_steps
                with torch.no_grad():
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Client {self.client_id}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() * self.accumulation_steps:.4f}")
            
            total_loss += epoch_loss
            self.training_stats["total_epochs"] += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / (len(self.train_loader) * epochs)
        avg_grad_norm = gradient_norm / (len(self.train_loader) * epochs / self.accumulation_steps)
        
        # Update statistics
        self.training_stats["avg_loss"] = avg_loss
        self.training_stats["best_accuracy"] = max(self.training_stats["best_accuracy"], accuracy)
        self.training_stats["total_batches"] += len(self.train_loader) * epochs
        
        # Return updated parameters and metrics
        updated_params = self.get_parameters({})
        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "gradient_norm": avg_grad_norm,
            "device_type": self.device_type,
            "dataset_size": len(self.dataset),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "epochs_completed": epochs,
            "training_mode": self.training_mode
        }
        
        logger.info(f"Client {self.client_id} training completed: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        return updated_params, len(self.dataset), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Enhanced local evaluation with comprehensive metrics"""
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []
        response_times = []
        confidence_scores = []
        
        with torch.no_grad():
            for frames, questions, labels in self.train_loader:
                labels = labels.to(self.device, non_blocking=True)
                
                # Measure response time
                start_time = time.time()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits = self.model(frames, questions, mode=self.training_mode)
                else:
                    logits = self.model(frames, questions, mode=self.training_mode)
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions and confidence scores
                probs = torch.softmax(logits, dim=-1)
                max_probs, predicted = torch.max(probs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                confidence_scores.extend(max_probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = sum(p == l for p, l in zip(predictions, true_labels)) / len(true_labels) * 100
        avg_loss = total_loss / len(self.train_loader)
        avg_response_time = sum(response_times) / len(response_times)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Calculate precision, recall, F1 for binary classification
        tp = sum(1 for p, l in zip(predictions, true_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, true_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, true_labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            "test_accuracy": accuracy,
            "test_loss": avg_loss,
            "avg_response_time_ms": avg_response_time,
            "avg_confidence": avg_confidence,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "device_type": self.device_type,
            "num_samples": len(true_labels),
            "training_stats": self.training_stats.copy()
        }
        
        logger.info(f"Client {self.client_id} evaluation: Acc={accuracy:.2f}%, F1={f1_score:.4f}")
        return avg_loss, len(self.dataset), metrics
    
    def save_local_model(self, round_num: int, save_dir: str = "./client_models"):
        """Save local model state for recovery/analysis"""
        save_path = Path(save_dir) / f"client_{self.client_id}_round_{round_num}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model_state_dict": {name: param.cpu() for name, param in self.model.named_parameters() 
                               if param.requires_grad},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_stats": self.training_stats,
            "client_config": {
                "client_id": self.client_id,
                "device_type": self.device_type,
                "training_mode": self.training_mode,
                "model_name": self.model_name
            }
        }
        
        torch.save(state, save_path)
        logger.debug(f"Client {self.client_id} model saved to {save_path}")
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get comprehensive client information"""
        return {
            "client_id": self.client_id,
            "device_type": self.device_type,
            "device": str(self.device),
            "training_mode": self.training_mode,
            "model_name": self.model_name,
            "dataset_size": len(self.dataset),
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "num_trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "training_stats": self.training_stats.copy()
        }