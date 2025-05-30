"""
Simplified Federated Learning Server
Simple FedAvg for Qwen2.5-VL fine-tuning
"""

import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimpleFedStrategy(FedAvg):
    """
    Simple federated averaging strategy
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_metrics = []
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Tuple[Any, Exception]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results"""
        
        if not results:
            return None, {}
            
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Log round information
        num_clients = len(results)
        avg_loss = np.mean([res.metrics.get("train_loss", 0.0) for _, res in results])
        avg_accuracy = np.mean([res.metrics.get("train_accuracy", 0.0) for _, res in results])
        
        logger.info(f"Round {server_round}: {num_clients} clients, "
                   f"Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
        
        # Store metrics
        self.round_metrics.append({
            "round": server_round,
            "num_clients": num_clients,
            "avg_train_loss": avg_loss,
            "avg_train_accuracy": avg_accuracy
        })
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, EvaluateRes]],
        failures: List[Tuple[Any, Exception]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
            
        # Weighted average
        total_examples = sum([res.num_examples for _, res in results])
        weighted_loss = sum([res.loss * res.num_examples for _, res in results]) / total_examples
        
        # Average accuracy
        avg_accuracy = np.mean([res.metrics.get("test_accuracy", 0.0) for _, res in results])
        
        logger.info(f"Round {server_round} Evaluation: "
                   f"Loss: {weighted_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        metrics = {"test_accuracy": avg_accuracy}
        return weighted_loss, metrics 