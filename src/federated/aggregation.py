"""
Federated Learning Aggregation Algorithms
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAggregator(ABC):
    """Base class for federated learning aggregators."""
    
    @abstractmethod
    def aggregate_parameters(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """Aggregate parameters from multiple clients."""
        pass

class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator."""
    
    def __init__(self):
        self.name = "FedAvg"
        logger.info("Initialized FedAvg aggregator")
    
    def aggregate_parameters(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """
        Aggregate parameters using weighted averaging.
        
        Args:
            parameters_list: List of parameter lists from each client
            weights: List of weights (typically dataset sizes) for each client
            
        Returns:
            Aggregated parameters
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize aggregated parameters
        aggregated_params = []
        num_layers = len(parameters_list[0])
        
        for layer_idx in range(num_layers):
            # Get parameters for this layer from all clients
            layer_params = [client_params[layer_idx] for client_params in parameters_list]
            
            # Weighted average
            weighted_sum = np.zeros_like(layer_params[0])
            for params, weight in zip(layer_params, normalized_weights):
                weighted_sum += params * weight
            
            aggregated_params.append(weighted_sum)
        
        logger.debug(f"Aggregated parameters from {len(parameters_list)} clients")
        return aggregated_params

class FedProxAggregator(BaseAggregator):
    """Federated Proximal (FedProx) aggregator with proximal term."""
    
    def __init__(self, mu: float = 0.01):
        self.mu = mu  # Proximal term coefficient
        self.name = "FedProx"
        self.global_params = None
        logger.info(f"Initialized FedProx aggregator with mu={mu}")
    
    def aggregate_parameters(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """
        Aggregate parameters using FedProx algorithm.
        
        Args:
            parameters_list: List of parameter lists from each client
            weights: List of weights for each client
            
        Returns:
            Aggregated parameters
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate")
        
        # First round: use FedAvg
        if self.global_params is None:
            aggregated = self._fedavg_aggregate(parameters_list, weights)
            self.global_params = [param.copy() for param in aggregated]
            return aggregated
        
        # Subsequent rounds: apply proximal term
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated_params = []
        num_layers = len(parameters_list[0])
        
        for layer_idx in range(num_layers):
            layer_params = [client_params[layer_idx] for client_params in parameters_list]
            
            # Weighted average
            weighted_sum = np.zeros_like(layer_params[0])
            for params, weight in zip(layer_params, normalized_weights):
                weighted_sum += params * weight
            
            # Apply proximal term
            global_param = self.global_params[layer_idx]
            proximal_param = (weighted_sum + self.mu * global_param) / (1 + self.mu)
            
            aggregated_params.append(proximal_param)
        
        # Update global parameters
        self.global_params = [param.copy() for param in aggregated_params]
        
        logger.debug(f"FedProx aggregated parameters from {len(parameters_list)} clients")
        return aggregated_params
    
    def _fedavg_aggregate(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """Standard FedAvg aggregation."""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated_params = []
        num_layers = len(parameters_list[0])
        
        for layer_idx in range(num_layers):
            layer_params = [client_params[layer_idx] for client_params in parameters_list]
            
            weighted_sum = np.zeros_like(layer_params[0])
            for params, weight in zip(layer_params, normalized_weights):
                weighted_sum += params * weight
            
            aggregated_params.append(weighted_sum)
        
        return aggregated_params

class FedAdamAggregator(BaseAggregator):
    """Federated Adam aggregator with adaptive learning rates."""
    
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, eta: float = 1e-3):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta  # Server learning rate
        self.name = "FedAdam"
        
        # Adam state
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        self.global_params = None
        
        logger.info(f"Initialized FedAdam aggregator")
    
    def aggregate_parameters(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """
        Aggregate parameters using FedAdam algorithm.
        
        Args:
            parameters_list: List of parameter lists from each client
            weights: List of weights for each client
            
        Returns:
            Aggregated parameters
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate")
        
        # First round: initialize
        if self.global_params is None:
            aggregated = self._fedavg_aggregate(parameters_list, weights)
            self.global_params = [param.copy() for param in aggregated]
            self.m = [np.zeros_like(param) for param in aggregated]
            self.v = [np.zeros_like(param) for param in aggregated]
            return aggregated
        
        # Compute pseudo-gradient (difference from global model)
        pseudo_gradients = self._compute_pseudo_gradients(parameters_list, weights)
        
        # Update time step
        self.t += 1
        
        # Apply Adam updates
        updated_params = []
        for i, (global_param, pseudo_grad, m_i, v_i) in enumerate(
            zip(self.global_params, pseudo_gradients, self.m, self.v)
        ):
            # Update biased first moment estimate
            m_i = self.beta1 * m_i + (1 - self.beta1) * pseudo_grad
            
            # Update biased second raw moment estimate
            v_i = self.beta2 * v_i + (1 - self.beta2) * (pseudo_grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m_i / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v_i / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_param = global_param + self.eta * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params.append(updated_param)
            
            # Update state
            self.m[i] = m_i
            self.v[i] = v_i
        
        # Update global parameters
        self.global_params = [param.copy() for param in updated_params]
        
        logger.debug(f"FedAdam aggregated parameters from {len(parameters_list)} clients")
        return updated_params
    
    def _compute_pseudo_gradients(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """Compute pseudo-gradients as weighted average of parameter differences."""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        pseudo_gradients = []
        num_layers = len(parameters_list[0])
        
        for layer_idx in range(num_layers):
            layer_params = [client_params[layer_idx] for client_params in parameters_list]
            global_param = self.global_params[layer_idx]
            
            # Compute weighted average of differences
            weighted_diff = np.zeros_like(global_param)
            for params, weight in zip(layer_params, normalized_weights):
                diff = params - global_param
                weighted_diff += diff * weight
            
            pseudo_gradients.append(weighted_diff)
        
        return pseudo_gradients
    
    def _fedavg_aggregate(
        self, 
        parameters_list: List[List[np.ndarray]], 
        weights: List[int]
    ) -> List[np.ndarray]:
        """Standard FedAvg aggregation for initialization."""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        aggregated_params = []
        num_layers = len(parameters_list[0])
        
        for layer_idx in range(num_layers):
            layer_params = [client_params[layer_idx] for client_params in parameters_list]
            
            weighted_sum = np.zeros_like(layer_params[0])
            for params, weight in zip(layer_params, normalized_weights):
                weighted_sum += params * weight
            
            aggregated_params.append(weighted_sum)
        
        return aggregated_params

def create_aggregator(aggregator_type: str, **kwargs) -> BaseAggregator:
    """Factory function to create aggregators."""
    
    if aggregator_type.lower() == "fedavg":
        return FedAvgAggregator()
    elif aggregator_type.lower() == "fedprox":
        mu = kwargs.get("mu", 0.01)
        return FedProxAggregator(mu=mu)
    elif aggregator_type.lower() == "fedadam":
        return FedAdamAggregator(**kwargs)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}") 