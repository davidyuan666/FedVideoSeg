"""
Enhanced Federated Learning Server for FedVideoQA
Implements device-aware federated aggregation from Section 3.4
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from pathlib import Path
import json
import time
from collections import defaultdict
import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.strategy import Strategy

from .aggregation import DeviceAwareFedAvg
from ..models.fedvideo_qa_model import FedVideoQAModel

logger = logging.getLogger(__name__)

class FedVideoQAStrategy(Strategy):
    """
    Custom federated learning strategy for FedVideoQA
    Implements device-aware aggregation from Equations 6-8
    """
    
    def __init__(
        self,
        model: FedVideoQAModel,
        fraction_fit: float = 0.8,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        max_communication_size_mb: int = 100,
        convergence_threshold: float = 0.01,
        patience: int = 5
    ):
        super().__init__()
        
        self.model = model
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.max_communication_size_mb = max_communication_size_mb
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        
        # Device-aware aggregator
        self.aggregator = DeviceAwareFedAvg()
        
        # Training metrics
        self.round_metrics = []
        self.device_statistics = defaultdict(list)
        self.convergence_history = []
        
        # Early stopping
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
        logger.info("Initialized FedVideoQA strategy with device-aware aggregation")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        parameters = []
        for param in self.model.parameters():
            parameters.append(param.detach().cpu().numpy())
        
        return fl.common.ndarrays_to_parameters(parameters)
    
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure clients for training round"""
        
        # Sample clients with device diversity awareness
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_fit),
            self.min_fit_clients
        )
        
        # Smart client selection considering device diversity
        sampled_clients = self._smart_client_selection(
            client_manager, sample_size
        )
        
        # Create fit configuration
        config = {
            "round": server_round,
            "local_epochs": self._adaptive_local_epochs(server_round),
            "max_communication_size_mb": self.max_communication_size_mb
        }
        
        if self.on_fit_config_fn:
            config.update(self.on_fit_config_fn(server_round))
        
        # Return client configurations
        fit_configurations = []
        for client in sampled_clients:
            fit_configurations.append((client, config))
        
        logger.info(
            f"Round {server_round}: Configured {len(fit_configurations)} "
            f"clients for training"
        )
        
        return fit_configurations
    
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure clients for evaluation round"""
        
        if self.fraction_evaluate == 0.0:
            return []
        
        # Sample clients for evaluation
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_evaluate),
            self.min_evaluate_clients
        )
        sampled_clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )
        
        # Create evaluation configuration
        config = {"round": server_round}
        if self.on_evaluate_config_fn:
            config.update(self.on_evaluate_config_fn(server_round))
        
        # Return evaluation configurations
        evaluate_configurations = []
        for client in sampled_clients:
            evaluate_configurations.append((client, config))
        
        return evaluate_configurations
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Tuple[Any, Exception]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results using device-aware federated averaging
        Implements Equations 6-8 from the paper
        """
        if not results:
            return None, {}
        
        # Extract parameters and metrics
        parameters_list = []
        weights_list = []
        client_metrics = []
        
        for client, fit_res in results:
            parameters_list.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
            weights_list.append(fit_res.num_examples)
            client_metrics.append(fit_res.metrics)
        
        # Device-aware aggregation (Equations 6-8)
        aggregated_params = self.aggregator.aggregate_device_aware(
            parameters_list=parameters_list,
            weights=weights_list,
            client_metrics=client_metrics
        )
        
        # Convert back to Parameters
        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_params)
        
        # Aggregate training metrics
        aggregated_metrics = self._aggregate_fit_metrics(client_metrics, weights_list)
        aggregated_metrics.update({
            "round": server_round,
            "num_clients": len(results),
            "total_examples": sum(weights_list)
        })
        
        # Store round metrics
        self.round_metrics.append({
            "round": server_round,
            "aggregated_metrics": aggregated_metrics,
            "client_metrics": client_metrics,
            "device_distribution": self._analyze_device_distribution(client_metrics)
        })
        
        # Update device statistics
        self._update_device_statistics(client_metrics)
        
        logger.info(
            f"Round {server_round}: Aggregated {len(results)} client updates"
        )
        
        return parameters_aggregated, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, EvaluateRes]],
        failures: List[Tuple[Any, Exception]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        # Extract evaluation metrics
        losses = []
        weights = []
        client_metrics = []
        
        for client, evaluate_res in results:
            losses.append(evaluate_res.loss)
            weights.append(evaluate_res.num_examples)
            client_metrics.append(evaluate_res.metrics)
        
        # Weighted average loss
        total_examples = sum(weights)
        aggregated_loss = sum(loss * weight for loss, weight in zip(losses, weights)) / total_examples
        
        # Aggregate evaluation metrics
        aggregated_metrics = self._aggregate_evaluate_metrics(client_metrics, weights)
        aggregated_metrics.update({
            "round": server_round,
            "aggregated_loss": aggregated_loss,
            "num_clients": len(results)
        })
        
        # Check convergence
        current_accuracy = aggregated_metrics.get("test_accuracy", 0.0)
        self._check_convergence(server_round, current_accuracy)
        
        return aggregated_loss, aggregated_metrics
    
    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on server-side data"""
        
        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        return loss, metrics
    
    def _smart_client_selection(self, client_manager, sample_size: int):
        """
        Smart client selection considering device diversity
        """
        all_clients = client_manager.all()
        
        if len(all_clients) <= sample_size:
            return all_clients
        
        # Try to maintain device diversity in selection
        # For now, use random sampling - can be enhanced with actual device info
        return client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients
        )
    
    def _adaptive_local_epochs(self, server_round: int) -> int:
        """Adaptive local epochs based on convergence progress"""
        base_epochs = 5
        
        # Reduce epochs as training progresses
        if server_round > 20:
            return max(3, base_epochs - 1)
        elif server_round > 50:
            return max(2, base_epochs - 2)
        else:
            return base_epochs
    
    def _aggregate_fit_metrics(
        self, 
        client_metrics: List[Dict[str, Any]], 
        weights: List[int]
    ) -> Dict[str, Scalar]:
        """Aggregate training metrics from clients"""
        
        if not client_metrics:
            return {}
        
        total_weight = sum(weights)
        aggregated = {}
        
        # Numerical metrics to aggregate
        numerical_keys = [
            "train_loss", "train_accuracy", "processing_time",
            "communication_cost_mb", "device_capability"
        ]
        
        for key in numerical_keys:
            values = [metrics.get(key, 0.0) for metrics in client_metrics]
            if values and total_weight > 0:
                weighted_avg = sum(val * weight for val, weight in zip(values, weights)) / total_weight
                aggregated[key] = weighted_avg
        
        # Device type distribution
        device_types = [metrics.get("device_type", "unknown") for metrics in client_metrics]
        device_counts = {}
        for device_type in device_types:
            device_counts[device_type] = device_counts.get(device_type, 0) + 1
        
        for device_type, count in device_counts.items():
            aggregated[f"device_{device_type}_count"] = count
        
        # Privacy budget tracking
        privacy_budgets = [
            metrics.get("privacy_budget_used", 0.0) for metrics in client_metrics
        ]
        if privacy_budgets:
            aggregated["max_privacy_budget_used"] = max(privacy_budgets)
            aggregated["avg_privacy_budget_used"] = np.mean(privacy_budgets)
        
        return aggregated
    
    def _aggregate_evaluate_metrics(
        self, 
        client_metrics: List[Dict[str, Any]], 
        weights: List[int]
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from clients"""
        
        if not client_metrics:
            return {}
        
        total_weight = sum(weights)
        aggregated = {}
        
        # Numerical metrics
        numerical_keys = [
            "test_accuracy", "test_loss", "confidence_score", "inference_time"
        ]
        
        for key in numerical_keys:
            values = [metrics.get(key, 0.0) for metrics in client_metrics]
            if values and total_weight > 0:
                weighted_avg = sum(val * weight for val, weight in zip(values, weights)) / total_weight
                aggregated[key] = weighted_avg
        
        # Modality weights analysis
        modality_weights_list = []
        for metrics in client_metrics:
            if "modality_weights" in metrics:
                modality_weights_list.append(metrics["modality_weights"])
        
        if modality_weights_list:
            avg_modality_weights = np.mean(modality_weights_list, axis=0)
            aggregated["avg_visual_weight"] = avg_modality_weights[0]
            aggregated["avg_audio_weight"] = avg_modality_weights[1]
            aggregated["avg_text_weight"] = avg_modality_weights[2]
        
        return aggregated
    
    def _analyze_device_distribution(self, client_metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze device type distribution"""
        device_distribution = defaultdict(int)
        
        for metrics in client_metrics:
            device_type = metrics.get("device_type", "unknown")
            device_distribution[device_type] += 1
        
        return dict(device_distribution)
    
    def _update_device_statistics(self, client_metrics: List[Dict[str, Any]]) -> None:
        """Update device-specific statistics"""
        for metrics in client_metrics:
            device_type = metrics.get("device_type", "unknown")
            self.device_statistics[device_type].append({
                "train_accuracy": metrics.get("train_accuracy", 0.0),
                "processing_time": metrics.get("processing_time", 0.0),
                "communication_cost": metrics.get("communication_cost_mb", 0.0),
                "device_capability": metrics.get("device_capability", 0.0)
            })
    
    def _check_convergence(self, server_round: int, current_accuracy: float) -> bool:
        """Check for convergence and early stopping"""
        self.convergence_history.append(current_accuracy)
        
        if current_accuracy > self.best_accuracy + self.convergence_threshold:
            self.best_accuracy = current_accuracy
            self.patience_counter = 0
            logger.info(f"Round {server_round}: New best accuracy: {current_accuracy:.4f}")
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(f"Round {server_round}: Convergence detected (patience={self.patience})")
                return True
        
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            "total_rounds": len(self.round_metrics),
            "best_accuracy": self.best_accuracy,
            "convergence_history": self.convergence_history,
            "device_statistics": dict(self.device_statistics),
            "round_metrics": self.round_metrics[-5:] if self.round_metrics else [],  # Last 5 rounds
            "final_device_distribution": self._analyze_device_distribution(
                self.round_metrics[-1]["client_metrics"] if self.round_metrics else []
            )
        }

class DeviceAwareFedAvg:
    """
    Device-aware federated averaging implementation
    Implements Equations 6-8 from the paper
    """
    
    def aggregate_device_aware(
        self,
        parameters_list: List[List[np.ndarray]],
        weights: List[int],
        client_metrics: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Aggregate parameters using device-aware weighted averaging
        
        Implements:
        - Equation 6: θ_{t+1} = Σ w_i^eff * θ_i^t
        - Equation 7: w_i^eff = (n_i * κ_i) / Σ(n_j * κ_j)
        - Equation 8: κ_i = (1/3) * Σ(c_i^m * ρ_i)
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate")
        
        # Calculate effective weights (Equation 7)
        effective_weights = self._calculate_effective_weights(weights, client_metrics)
        
        # Normalize weights
        total_weight = sum(effective_weights)
        normalized_weights = [w / total_weight for w in effective_weights]
        
        # Aggregate parameters (Equation 6)
        aggregated_params = []
        num_layers = len(parameters_list[0])
        
        for layer_idx in range(num_layers):
            layer_params = [client_params[layer_idx] for client_params in parameters_list]
            
            # Weighted average
            weighted_sum = np.zeros_like(layer_params[0])
            for params, weight in zip(layer_params, normalized_weights):
                weighted_sum += params * weight
            
            aggregated_params.append(weighted_sum)
        
        logger.debug(f"Aggregated parameters from {len(parameters_list)} clients")
        return aggregated_params
    
    def _calculate_effective_weights(
        self,
        weights: List[int],
        client_metrics: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculate effective weights based on data size and device capability
        Implements Equation 7: w_i^eff = (n_i * κ_i) / Σ(n_j * κ_j)
        """
        effective_weights = []
        
        for weight, metrics in zip(weights, client_metrics):
            # Get device capability factor κ_i
            device_capability = metrics.get("device_capability", 1.0)
            
            # Calculate effective weight: n_i * κ_i
            effective_weight = weight * device_capability
            effective_weights.append(effective_weight)
        
        return effective_weights 