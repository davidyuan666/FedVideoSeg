"""
Simplified Federated Learning Client
Focuses on Qwen2.5-VL fine-tuning for frame relevance classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import time

from ..models.qwen_classifier import SimpleQwenFrameClassifier

logger = logging.getLogger(__name__)

class FrameRelevanceDataset(Dataset):
    """Simple dataset for frame-question pairs"""
    
    def __init__(self, data: List[Tuple]):
        self.data = data  # List of (frame, question, label)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame, question, label = self.data[idx]
        return frame, question, torch.tensor(label, dtype=torch.long)

class FedClient(fl.client.NumPyClient):
    """
    Simple federated client for Qwen2.5-VL fine-tuning
    """
    
    def __init__(self, client_id: str, local_data: List[Tuple], device_type: str = "desktop"):
        self.client_id = client_id
        self.device_type = device_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = SimpleQwenFrameClassifier()
        self.model.to(self.device)
        
        # Prepare data
        self.dataset = FrameRelevanceDataset(local_data)
        self.train_loader = DataLoader(
            self.dataset, 
            batch_size=4 if device_type == "mobile" else 8,
            shuffle=True
        )
        
        # Optimizer (only for trainable parameters)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters (only LoRA and classifier)"""
        params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # Only trainable parameters
                params.append(param.detach().cpu().numpy())
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        param_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = torch.from_numpy(parameters[param_idx]).to(param.device)
                param_idx += 1
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Local training"""
        self.set_parameters(parameters)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Local training epochs
        epochs = config.get("local_epochs", 3)
        for epoch in range(epochs):
            for batch_idx, (frames, questions, labels) in enumerate(self.train_loader):
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(frames, questions)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Client {self.client_id}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.train_loader) / epochs
        
        # Return updated parameters and metrics
        updated_params = self.get_parameters({})
        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "device_type": self.device_type,
            "dataset_size": len(self.dataset)
        }
        
        return updated_params, len(self.dataset), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Local evaluation with simple metrics"""
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []
        response_times = []
        
        with torch.no_grad():
            for frames, questions, labels in self.train_loader:
                labels = labels.to(self.device)
                
                # Measure response time
                start_time = time.time()
                logits = self.model(frames, questions)
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Collect predictions and labels
                _, predicted = torch.max(logits.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate simple metrics
        accuracy = sum(p == l for p, l in zip(predictions, true_labels)) / len(true_labels) * 100
        avg_loss = total_loss / len(self.train_loader)
        avg_response_time = sum(response_times) / len(response_times)
        
        metrics = {
            "test_accuracy": accuracy,
            "test_loss": avg_loss,
            "avg_response_time_ms": avg_response_time,
            "device_type": self.device_type,
            "num_samples": len(true_labels)
        }
        
        return avg_loss, len(self.dataset), metrics 