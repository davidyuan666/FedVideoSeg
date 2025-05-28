"""
Simplified Federated Learning Client
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import flwr as fl

from ..models.multimodal_model import MultimodalVideoQAModel

logger = logging.getLogger(__name__)

class SimpleFedClient(fl.client.NumPyClient):
    """Simplified federated learning client without complex privacy mechanisms."""
    
    def __init__(
        self,
        client_id: str,
        model: MultimodalVideoQAModel,
        train_loader,
        test_loader,
        local_epochs: int = 3,
        learning_rate: float = 1e-4
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters."""
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters."""
        params_dict = zip(self.model.parameters(), parameters)
        for param, new_param in params_dict:
            param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Simple local training."""
        
        self.set_parameters(parameters)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            for batch in self.train_loader:
                # Simple forward pass
                outputs = self.model(
                    visual_features=batch['visual_features'],
                    audio_features=batch['audio_features'],
                    text_features=batch['text_features'],
                    question_features=batch['question_features'],
                    answer_labels=batch['answer_labels']
                )
                
                loss = outputs['loss']
                
                # Simple backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"train_loss": avg_loss}
        )
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Simple evaluation."""
        
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.model(
                    visual_features=batch['visual_features'],
                    audio_features=batch['audio_features'],
                    text_features=batch['text_features'],
                    question_features=batch['question_features'],
                    answer_labels=batch['answer_labels']
                )
                
                total_loss += outputs['loss'].item()
                
                predictions = torch.argmax(outputs['answer_logits'], dim=-1)
                correct += (predictions == batch['answer_labels']).sum().item()
                total += batch['answer_labels'].size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.test_loader)
        
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy} 