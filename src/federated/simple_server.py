"""
Simplified Federated Learning Server
"""

import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class SimpleFedStrategy(fl.server.strategy.FedAvg):
    """Simplified federated averaging strategy."""
    
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        
    def initialize_parameters(self, client_manager) -> Optional[fl.common.Parameters]:
        """Initialize with model parameters."""
        parameters = [param.detach().cpu().numpy() for param in self.model.parameters()]
        return fl.common.ndarrays_to_parameters(parameters)

def start_simple_server(
    model,
    num_rounds: int = 50,
    min_clients: int = 2,
    server_address: str = "[::]:8080"
):
    """Start simplified federated server."""
    
    strategy = SimpleFedStrategy(
        model=model,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    ) 