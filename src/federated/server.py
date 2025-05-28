"""
Federated Learning Server for FedVideoQA
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

from .aggregation import FedAvgAggregator, FedProxAggregator
from ..models.multimodal_model import MultimodalVideoQAModel
from ..utils.metrics import compute_qa_metrics

logger = logging.getLogger(__name__)

class FedVideoQAStrategy(Strategy):
    """Custom federated learning strategy for VideoQA."""
    
    def __init__(
        self,
        model: MultimodalVideoQAModel,
        aggregator_type: str = "fedavg",
        fraction_fit: float = 0.8,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        privacy_budget_per_round: float = 0.02,
        max_communication_size_mb: int = 100
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
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.privacy_budget_per_round = privacy_budget_per_round
        self.max_communication_size_mb = max_communication_size_mb
        
        # Initialize aggregator
        if aggregator_type == "fedavg":
            self.aggregator = FedAvgAggregator()
        elif aggregator_type == "fedprox":
            self.aggregator = FedProxAggregator(mu=0.01)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")
        
        # Training metrics
        self.round_metrics = []
        self.global_model_performance = []
        
        logger.info(f"Initialized FedVideoQA strategy with {aggregator_type} aggregator")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.initial_parameters is not None:
            return self.initial_parameters
        
        # Get initial parameters from model
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
        """Configure the next round of training."""
        
        # Sample clients
        sample_size = max(
            int(len(client_manager.all()) * self.fraction_fit),
            self.min_fit_clients
        )
        sampled_clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=self.min_available_clients
        )
        
        # Create fit configuration
        config = {
            "round": server_round,
            "local_epochs": 5,
            "privacy_budget_per_round": self.privacy_budget_per_round,
            "max_communication_size_mb": self.max_communication_size_mb
        }
        
        if self.on_fit_config_fn:
            config.update(self.on_fit_config_fn(server_round))
        
        # Return client configurations
        fit_configurations = []
        for client in sampled_clients:
            fit_configurations.append((client, config))
        
        logger.info(f"Round {server_round}: Configured {len(fit_configurations)} clients for training")
        return fit_configurations
    
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure the next round of evaluation."""
        
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
        
        # Return client configurations
        evaluate_configurations = []
        for client in sampled_clients:
            evaluate_configurations.append((client, config))
        
        logger.info(f"Round {server_round}: Configured {len(evaluate_configurations)} clients for evaluation")
        return evaluate_configurations
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, FitRes]],
        failures: List[Tuple[Any, FitRes]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        
        if not results:
            return None, {}
        
        # Handle failures
        if failures and not self.accept_failures:
            return None, {}
        
        # Extract parameters and metrics
        parameters_list = []
        weights_list = []
        client_metrics = []
        
        for client, fit_res in results:
            parameters_list.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
            weights_list.append(fit_res.num_examples)
            client_metrics.append(fit_res.metrics)
        
        # Aggregate parameters using the selected aggregator
        aggregated_parameters = self.aggregator.aggregate_parameters(
            parameters_list, weights_list
        )
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_fit_metrics(client_metrics, weights_list)
        
        # Store round metrics
        round_info = {
            "round": server_round,
            "num_clients": len(results),
            "num_failures": len(failures),
            "aggregated_metrics": aggregated_metrics,
            "timestamp": time.time()
        }
        self.round_metrics.append(round_info)
        
        # Log round summary
        logger.info(f"Round {server_round} aggregation complete:")
        logger.info(f"  Clients participated: {len(results)}")
        logger.info(f"  Failures: {len(failures)}")
        logger.info(f"  Avg train loss: {aggregated_metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"  Avg train accuracy: {aggregated_metrics.get('train_accuracy', 'N/A'):.4f}")
        
        return fl.common.ndarrays_to_parameters(aggregated_parameters), aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, EvaluateRes]],
        failures: List[Tuple[Any, EvaluateRes]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        
        if not results:
            return None, {}
        
        # Handle failures
        if failures and not self.accept_failures:
            return None, {}
        
        # Extract metrics
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
        
        # Aggregate other metrics
        aggregated_metrics = self._aggregate_evaluate_metrics(client_metrics, weights)
        
        # Store global model performance
        performance_info = {
            "round": server_round,
            "loss": aggregated_loss,
            "accuracy": aggregated_metrics.get("test_accuracy", 0.0),
            "confidence": aggregated_metrics.get("confidence_score", 0.0),
            "num_clients": len(results),
            "timestamp": time.time()
        }
        self.global_model_performance.append(performance_info)
        
        # Log evaluation summary
        logger.info(f"Round {server_round} evaluation complete:")
        logger.info(f"  Global loss: {aggregated_loss:.4f}")
        logger.info(f"  Global accuracy: {aggregated_metrics.get('test_accuracy', 'N/A'):.4f}")
        logger.info(f"  Avg confidence: {aggregated_metrics.get('confidence_score', 'N/A'):.4f}")
        
        return aggregated_loss, aggregated_metrics
    
    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the global model on server-side data."""
        
        if self.evaluate_fn is None:
            return None
        
        # Convert parameters to model weights
        parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        
        # Evaluate using the provided function
        loss, metrics = self.evaluate_fn(server_round, parameters_ndarrays, {})
        
        return loss, metrics
    
    def _aggregate_fit_metrics(
        self, 
        client_metrics: List[Dict[str, Any]], 
        weights: List[int]
    ) -> Dict[str, Scalar]:
        """Aggregate training metrics from clients."""
        
        if not client_metrics:
            return {}
        
        total_weight = sum(weights)
        aggregated = {}
        
        # Aggregate numerical metrics
        numerical_keys = ["train_loss", "train_accuracy", "privacy_budget_used", "communication_size_mb"]
        
        for key in numerical_keys:
            values = [metrics.get(key, 0.0) for metrics in client_metrics]
            if values:
                weighted_avg = sum(val * weight for val, weight in zip(values, weights)) / total_weight
                aggregated[key] = weighted_avg
        
        # Count total dataset size
        total_dataset_size = sum(metrics.get("dataset_size", 0) for metrics in client_metrics)
        aggregated["total_dataset_size"] = total_dataset_size
        
        # Privacy budget tracking
        max_privacy_budget = max(metrics.get("privacy_budget_used", 0.0) for metrics in client_metrics)
        aggregated["max_privacy_budget_used"] = max_privacy_budget
        
        return aggregated
    
    def _aggregate_evaluate_metrics(
        self, 
        client_metrics: List[Dict[str, Any]], 
        weights: List[int]
    ) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from clients."""
        
        if not client_metrics:
            return {}
        
        total_weight = sum(weights)
        aggregated = {}
        
        # Aggregate numerical metrics
        numerical_keys = ["test_accuracy", "test_loss", "confidence_score"]
        
        for key in numerical_keys:
            values = [metrics.get(key, 0.0) for metrics in client_metrics]
            if values:
                weighted_avg = sum(val * weight for val, weight in zip(values, weights)) / total_weight
                aggregated[key] = weighted_avg
        
        # Aggregate modality weights
        modality_weights_list = [metrics.get("modality_weights", [0.33, 0.33, 0.34]) for metrics in client_metrics]
        if modality_weights_list:
            avg_modality_weights = np.average(modality_weights_list, weights=weights, axis=0)
            aggregated["avg_visual_weight"] = avg_modality_weights[0]
            aggregated["avg_audio_weight"] = avg_modality_weights[1]
            aggregated["avg_text_weight"] = avg_modality_weights[2]
        
        return aggregated

class FedVideoQAServer:
    """Main federated learning server for VideoQA."""
    
    def __init__(
        self,
        model: MultimodalVideoQAModel,
        strategy_config: Dict[str, Any],
        server_config: Dict[str, Any],
        save_dir: str = "./federated_results"
    ):
        self.model = model
        self.strategy_config = strategy_config
        self.server_config = server_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize strategy
        self.strategy = FedVideoQAStrategy(
            model=model,
            **strategy_config
        )
        
        # Server metrics
        self.server_metrics = {
            "start_time": None,
            "end_time": None,
            "total_rounds": 0,
            "total_clients": 0,
            "convergence_round": None,
            "best_accuracy": 0.0,
            "privacy_budget_consumed": 0.0
        }
        
        logger.info("Initialized FedVideoQA server")
    
    def start_training(
        self,
        num_rounds: int = 100,
        server_address: str = "[::]:8080",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start federated training."""
        
        logger.info(f"Starting federated training for {num_rounds} rounds")
        self.server_metrics["start_time"] = time.time()
        self.server_metrics["total_rounds"] = num_rounds
        
        # Default server configuration
        default_config = fl.server.ServerConfig(num_rounds=num_rounds)
        if config:
            default_config = fl.server.ServerConfig(**config)
        
        try:
            # Start Flower server
            history = fl.server.start_server(
                server_address=server_address,
                config=default_config,
                strategy=self.strategy
            )
            
            self.server_metrics["end_time"] = time.time()
            
            # Analyze training results
            training_results = self._analyze_training_results(history)
            
            # Save results
            self._save_training_results(training_results)
            
            logger.info("Federated training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            self.server_metrics["end_time"] = time.time()
            raise
    
    def _analyze_training_results(self, history) -> Dict[str, Any]:
        """Analyze and summarize training results."""
        
        # Extract metrics from history
        losses_distributed = history.losses_distributed
        losses_centralized = history.losses_centralized
        metrics_distributed = history.metrics_distributed
        metrics_centralized = history.metrics_centralized
        
        # Find best performance
        best_round = 0
        best_accuracy = 0.0
        
        for round_num, metrics in metrics_distributed.items():
            accuracy = metrics.get("test_accuracy", 0.0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_round = round_num
        
        self.server_metrics["best_accuracy"] = best_accuracy
        self.server_metrics["convergence_round"] = best_round
        
        # Calculate training duration
        training_duration = self.server_metrics["end_time"] - self.server_metrics["start_time"]
        
        # Privacy budget analysis
        total_privacy_budget = 0.0
        for round_metrics in self.strategy.round_metrics:
            total_privacy_budget += round_metrics["aggregated_metrics"].get("max_privacy_budget_used", 0.0)
        
        self.server_metrics["privacy_budget_consumed"] = total_privacy_budget
        
        # Compile results
        results = {
            "server_metrics": self.server_metrics,
            "training_history": {
                "losses_distributed": dict(losses_distributed),
                "losses_centralized": dict(losses_centralized),
                "metrics_distributed": dict(metrics_distributed),
                "metrics_centralized": dict(metrics_centralized)
            },
            "round_metrics": self.strategy.round_metrics,
            "global_performance": self.strategy.global_model_performance,
            "summary": {
                "total_rounds": self.server_metrics["total_rounds"],
                "training_duration_hours": training_duration / 3600,
                "best_accuracy": best_accuracy,
                "convergence_round": best_round,
                "privacy_budget_consumed": total_privacy_budget,
                "privacy_compliant": total_privacy_budget <= 2.0,  # ε=2.0 threshold
                "communication_efficient": self._check_communication_efficiency()
            }
        }
        
        return results
    
    def _check_communication_efficiency(self) -> bool:
        """Check if communication was efficient (≤100MB per round)."""
        for round_metrics in self.strategy.round_metrics:
            avg_comm_size = round_metrics["aggregated_metrics"].get("communication_size_mb", 0.0)
            if avg_comm_size > self.strategy.max_communication_size_mb:
                return False
        return True
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to disk."""
        
        # Save main results
        results_file = self.save_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model checkpoint
        model_file = self.save_dir / "global_model.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "visual_dim": self.model.attention_fusion.visual_proj.in_features,
                "audio_dim": self.model.attention_fusion.audio_proj.in_features,
                "text_dim": self.model.attention_fusion.text_proj.in_features,
                "hidden_dim": self.model.hidden_dim,
                "num_classes": self.model.num_classes
            },
            "training_summary": results["summary"]
        }, model_file)
        
        # Save detailed metrics
        metrics_file = self.save_dir / "detailed_metrics.json"
        detailed_metrics = {
            "round_metrics": self.strategy.round_metrics,
            "global_performance": self.strategy.global_model_performance
        }
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {self.save_dir}")
    
    def load_model_checkpoint(self, checkpoint_path: str) -> None:
        """Load a saved model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model checkpoint loaded from {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process."""
        return {
            "server_metrics": self.server_metrics,
            "strategy_config": self.strategy_config,
            "num_rounds_completed": len(self.strategy.round_metrics),
            "best_global_accuracy": self.server_metrics.get("best_accuracy", 0.0),
            "privacy_budget_used": self.server_metrics.get("privacy_budget_consumed", 0.0),
            "training_duration": (
                self.server_metrics.get("end_time", 0) - 
                self.server_metrics.get("start_time", 0)
            ) if self.server_metrics.get("start_time") else 0
        }

def create_federated_server(
    model: MultimodalVideoQAModel,
    config: Dict[str, Any]
) -> FedVideoQAServer:
    """Factory function to create a federated server."""
    
    strategy_config = config.get("strategy", {})
    server_config = config.get("server", {})
    save_dir = config.get("save_dir", "./federated_results")
    
    return FedVideoQAServer(
        model=model,
        strategy_config=strategy_config,
        server_config=server_config,
        save_dir=save_dir
    ) 