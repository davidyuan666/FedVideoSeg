"""
Enhanced Federated Learning Server
Advanced FedAvg implementation for Qwen2.5-VL fine-tuning with comprehensive features
"""

import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import logging
import json
import time
from pathlib import Path
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

class FedServer(FedAvg):
    """
    Enhanced federated averaging strategy with advanced features:
    - Adaptive client selection
    - Performance monitoring
    - Model persistence
    - Convergence detection
    - Client contribution tracking
    """
    
    def __init__(self, 
                 training_mode: str = "finetune",
                 model_save_path: str = "./global_models",
                 convergence_threshold: float = 0.001,
                 convergence_patience: int = 5,
                 client_selection_strategy: str = "random",
                 adaptive_lr: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Training configuration
        self.training_mode = training_mode
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Convergence detection
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.converged = False
        
        # Client selection and adaptation
        self.client_selection_strategy = client_selection_strategy
        self.adaptive_lr = adaptive_lr
        self.client_performance_history = defaultdict(list)
        self.client_reliability_scores = defaultdict(float)
        
        # Monitoring and statistics
        self.round_metrics = []
        self.global_metrics = {
            "total_rounds": 0,
            "total_training_time": 0.0,
            "best_accuracy": 0.0,
            "convergence_round": None
        }
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        self.participation_history = []
        
        logger.info(f"FedServer initialized with training_mode: {training_mode}")
        logger.info(f"Model save path: {self.model_save_path}")
        logger.info(f"Convergence threshold: {convergence_threshold}, patience: {convergence_patience}")
        
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[Any, Dict[str, Any]]]:
        """Configure the next round of training with adaptive parameters"""
        
        # Get available clients
        available_clients = client_manager.all()
        
        # Adaptive client selection
        if self.client_selection_strategy == "performance_based" and server_round > 2:
            selected_clients = self._select_clients_by_performance(available_clients)
        else:
            # Standard random selection
            selected_clients = super().configure_fit(server_round, parameters, client_manager)
        
        # Adaptive learning rate
        config = {
            "server_round": server_round,
            "local_epochs": self._get_adaptive_epochs(server_round),
            "learning_rate": self._get_adaptive_lr(server_round),
            "training_mode": self.training_mode,
            "convergence_check": server_round % 5 == 0  # Check convergence every 5 rounds
        }
        
        # Add configuration to each client
        configured_clients = []
        for client, _ in selected_clients:
            configured_clients.append((client, config))
            
        logger.info(f"Round {server_round}: Configured {len(configured_clients)} clients")
        logger.info(f"Adaptive config: epochs={config['local_epochs']}, lr={config['learning_rate']:.6f}")
        
        return configured_clients
        
    def _select_clients_by_performance(self, available_clients: List) -> List:
        """Select clients based on historical performance"""
        if not self.client_performance_history:
            return available_clients[:self.min_fit_clients]
        
        # Calculate reliability scores
        client_scores = []
        for client in available_clients:
            client_id = getattr(client, 'cid', str(client))
            score = self.client_reliability_scores.get(client_id, 0.5)  # Default neutral score
            client_scores.append((client, score))
        
        # Sort by score and select top performers
        client_scores.sort(key=lambda x: x[1], reverse=True)
        num_select = max(self.min_fit_clients, int(len(available_clients) * self.fraction_fit))
        
        selected = [client for client, _ in client_scores[:num_select]]
        logger.debug(f"Selected {len(selected)} clients based on performance scores")
        
        return selected
        
    def _get_adaptive_epochs(self, server_round: int) -> int:
        """Get adaptive number of local epochs"""
        if server_round <= 5:
            return 3  # More epochs early on
        elif server_round <= 15:
            return 2  # Medium epochs in middle
        else:
            return 1  # Fewer epochs as training progresses
            
    def _get_adaptive_lr(self, server_round: int) -> float:
        """Get adaptive learning rate"""
        if not self.adaptive_lr:
            return 1e-4
            
        # Exponential decay
        initial_lr = 1e-4
        decay_rate = 0.95
        return initial_lr * (decay_rate ** (server_round // 5))
    
    def aggregate_fit(self,
                     server_round: int,
                     results: List[Tuple[Any, FitRes]],
                     failures: List[Tuple[Any, Exception]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Enhanced aggregation with comprehensive monitoring"""
        
        round_start_time = time.time()
        
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}
            
        # Log failures if any
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")
            for client, exception in failures:
                logger.error(f"Client {client} failed: {exception}")
        
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Calculate round statistics
        num_clients = len(results)
        total_samples = sum([res.num_examples for _, res in results])
        
        # Extract metrics
        train_losses = [res.metrics.get("train_loss", 0.0) for _, res in results if res.metrics]
        train_accuracies = [res.metrics.get("train_accuracy", 0.0) for _, res in results if res.metrics]
        gradient_norms = [res.metrics.get("gradient_norm", 0.0) for _, res in results if res.metrics]
        
        # Calculate weighted averages
        weights = [res.num_examples for _, res in results]
        weighted_avg_loss = np.average(train_losses, weights=weights) if train_losses else 0.0
        weighted_avg_accuracy = np.average(train_accuracies, weights=weights) if train_accuracies else 0.0
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        # Update client performance tracking
        self._update_client_performance(results)
        
        # Round timing
        round_duration = time.time() - round_start_time
        self.global_metrics["total_training_time"] += round_duration
        
        # Store comprehensive metrics
        round_metrics = {
            "round": server_round,
            "num_clients": num_clients,
            "total_samples": total_samples,
            "avg_train_loss": float(weighted_avg_loss),
            "avg_train_accuracy": float(weighted_avg_accuracy),
            "avg_gradient_norm": float(avg_gradient_norm),
            "round_duration": round_duration,
            "num_failures": len(failures),
            "participation_rate": num_clients / self.min_available_clients if self.min_available_clients > 0 else 1.0
        }
        
        self.round_metrics.append(round_metrics)
        self.loss_history.append(weighted_avg_loss)
        self.accuracy_history.append(weighted_avg_accuracy)
        self.participation_history.append(num_clients)
        
        # Update global metrics
        self.global_metrics["total_rounds"] = server_round
        if weighted_avg_accuracy > self.global_metrics["best_accuracy"]:
            self.global_metrics["best_accuracy"] = weighted_avg_accuracy
        
        # Check for convergence
        self._check_convergence(weighted_avg_loss, server_round)
        
        # Save model periodically
        if server_round % 10 == 0 or self.converged:
            self._save_global_model(aggregated_parameters, server_round, round_metrics)
        
        # Log comprehensive information
        logger.info(f"Round {server_round} Training Summary:")
        logger.info(f"  Clients: {num_clients}/{self.min_available_clients}, Samples: {total_samples}")
        logger.info(f"  Avg Loss: {weighted_avg_loss:.4f}, Avg Accuracy: {weighted_avg_accuracy:.4f}")
        logger.info(f"  Gradient Norm: {avg_gradient_norm:.4f}, Duration: {round_duration:.2f}s")
        
        if self.converged:
            logger.info(f"🎯 Training converged at round {server_round}!")
            
        return aggregated_parameters, {
            "avg_train_loss": weighted_avg_loss,
            "avg_train_accuracy": weighted_avg_accuracy,
            "round_duration": round_duration,
            "converged": self.converged
        }
        
    def _update_client_performance(self, results: List[Tuple[Any, FitRes]]) -> None:
        """Update client performance tracking and reliability scores"""
        for client, res in results:
            client_id = getattr(client, 'cid', str(client))
            
            if res.metrics:
                accuracy = res.metrics.get("train_accuracy", 0.0)
                loss = res.metrics.get("train_loss", float('inf'))
                response_time = res.metrics.get("round_duration", 0.0)
                
                # Store performance metrics
                self.client_performance_history[client_id].append({
                    "accuracy": accuracy,
                    "loss": loss,
                    "response_time": response_time,
                    "num_samples": res.num_examples
                })
                
                # Calculate reliability score (0-1, higher is better)
                # Based on accuracy, low loss, reasonable response time, and consistency
                recent_performance = self.client_performance_history[client_id][-5:]  # Last 5 rounds
                
                avg_accuracy = np.mean([p["accuracy"] for p in recent_performance])
                avg_loss = np.mean([p["loss"] for p in recent_performance])
                consistency = 1.0 - np.std([p["accuracy"] for p in recent_performance])
                
                # Normalize and combine metrics
                reliability = (avg_accuracy * 0.5 + 
                             (1.0 / (1.0 + avg_loss)) * 0.3 + 
                             max(0, consistency) * 0.2)
                
                self.client_reliability_scores[client_id] = min(1.0, max(0.0, reliability))
                
    def _check_convergence(self, current_loss: float, server_round: int) -> None:
        """Check if training has converged"""
        if self.converged:
            return
            
        if current_loss < self.best_loss - self.convergence_threshold:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.convergence_patience:
            self.converged = True
            self.global_metrics["convergence_round"] = server_round
            logger.info(f"Training converged at round {server_round} with loss {current_loss:.4f}")
    
    def aggregate_evaluate(self,
                          server_round: int,
                          results: List[Tuple[Any, EvaluateRes]],
                          failures: List[Tuple[Any, Exception]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Enhanced evaluation aggregation with detailed metrics"""
        
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results")
            return None, {}
            
        # Log evaluation failures
        if failures:
            logger.warning(f"Round {server_round} Evaluation: {len(failures)} client failures")
            
        # Calculate weighted metrics
        total_examples = sum([res.num_examples for _, res in results])
        if total_examples == 0:
            return None, {}
            
        weighted_loss = sum([res.loss * res.num_examples for _, res in results]) / total_examples
        
        # Extract various metrics
        accuracies = [res.metrics.get("test_accuracy", 0.0) for _, res in results if res.metrics]
        f1_scores = [res.metrics.get("f1_score", 0.0) for _, res in results if res.metrics]
        precisions = [res.metrics.get("precision", 0.0) for _, res in results if res.metrics]
        recalls = [res.metrics.get("recall", 0.0) for _, res in results if res.metrics]
        response_times = [res.metrics.get("avg_response_time_ms", 0.0) for _, res in results if res.metrics]
        
        # Calculate averages
        weights = [res.num_examples for _, res in results]
        avg_accuracy = np.average(accuracies, weights=weights) if accuracies else 0.0
        avg_f1 = np.average(f1_scores, weights=weights) if f1_scores else 0.0
        avg_precision = np.average(precisions, weights=weights) if precisions else 0.0
        avg_recall = np.average(recalls, weights=weights) if recalls else 0.0
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        # Update round metrics with evaluation results
        if self.round_metrics and self.round_metrics[-1]["round"] == server_round:
            self.round_metrics[-1].update({
                "eval_loss": float(weighted_loss),
                "eval_accuracy": float(avg_accuracy),
                "eval_f1_score": float(avg_f1),
                "eval_precision": float(avg_precision),
                "eval_recall": float(avg_recall),
                "avg_response_time": float(avg_response_time),
                "eval_clients": len(results)
            })
        
        logger.info(f"Round {server_round} Evaluation Summary:")
        logger.info(f"  Loss: {weighted_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
        logger.info(f"  Avg Response Time: {avg_response_time:.2f}ms")
        
        metrics = {
            "test_accuracy": avg_accuracy,
            "test_loss": weighted_loss,
            "f1_score": avg_f1,
            "precision": avg_precision,
            "recall": avg_recall,
            "avg_response_time": avg_response_time
        }
        
        return weighted_loss, metrics
    
    def _save_global_model(self, parameters: Parameters, server_round: int, metrics: Dict[str, Any]) -> None:
        """Save global model and training state"""
        try:
            model_path = self.model_save_path / f"global_model_round_{server_round}.pt"
            
            # Save model parameters
            if parameters:
                # Convert parameters to tensors and save
                param_dict = {}
                for i, param_array in enumerate(parameters.tensors):
                    param_dict[f"param_{i}"] = torch.from_numpy(param_array)
                
                save_dict = {
                    "parameters": param_dict,
                    "server_round": server_round,
                    "metrics": metrics,
                    "training_mode": self.training_mode,
                    "global_metrics": self.global_metrics.copy(),
                    "converged": self.converged
                }
                
                torch.save(save_dict, model_path)
                logger.info(f"Global model saved: {model_path}")
            
            # Save training history
            history_path = self.model_save_path / f"training_history_round_{server_round}.json"
            history_data = {
                "round_metrics": self.round_metrics,
                "global_metrics": self.global_metrics,
                "loss_history": self.loss_history,
                "accuracy_history": self.accuracy_history,
                "participation_history": self.participation_history,
                "client_reliability_scores": dict(self.client_reliability_scores)
            }
            
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save global model: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            "global_metrics": self.global_metrics.copy(),
            "total_rounds_completed": len(self.round_metrics),
            "convergence_status": {
                "converged": self.converged,
                "convergence_round": self.global_metrics.get("convergence_round"),
                "best_loss": self.best_loss,
                "patience_counter": self.patience_counter
            },
            "performance_trends": {
                "loss_trend": "decreasing" if len(self.loss_history) > 1 and self.loss_history[-1] < self.loss_history[0] else "stable",
                "accuracy_trend": "increasing" if len(self.accuracy_history) > 1 and self.accuracy_history[-1] > self.accuracy_history[0] else "stable",
                "avg_participation": np.mean(self.participation_history) if self.participation_history else 0
            },
            "client_statistics": {
                "num_unique_clients": len(self.client_reliability_scores),
                "top_performers": sorted(self.client_reliability_scores.items(), key=lambda x: x[1], reverse=True)[:5],
                "avg_reliability_score": np.mean(list(self.client_reliability_scores.values())) if self.client_reliability_scores else 0
            },
            "training_configuration": {
                "training_mode": self.training_mode,
                "convergence_threshold": self.convergence_threshold,
                "convergence_patience": self.convergence_patience,
                "client_selection_strategy": self.client_selection_strategy,
                "adaptive_lr": self.adaptive_lr
            }
        }
    
    def should_continue_training(self, server_round: int, max_rounds: int) -> bool:
        """Determine if training should continue"""
        if self.converged:
            logger.info("Training stopped due to convergence")
            return False
            
        if server_round >= max_rounds:
            logger.info(f"Training stopped: reached maximum rounds ({max_rounds})")
            return False
            
        return True

# Legacy alias for backwards compatibility
FedStrategy = FedServer 