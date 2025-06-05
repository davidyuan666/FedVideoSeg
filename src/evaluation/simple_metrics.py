"""
Simple Evaluation Metrics for FedVideoQA
Straightforward metrics without complex calculations
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleMetrics:
    """Simple evaluation metrics for FedVideoQA"""
    
    def __init__(self):
        self.results = {}
        
    def measure_processing_time(self, binary_search_func, exhaustive_search_func, video_path: str, question: str):
        """Measure processing time for binary vs exhaustive search"""
        
        # Time binary search
        start_time = time.time()
        binary_result = binary_search_func(video_path, question)
        binary_time = time.time() - start_time
        
        # Time exhaustive search  
        start_time = time.time()
        exhaustive_result = exhaustive_search_func(video_path, question)
        exhaustive_time = time.time() - start_time
        
        # Calculate metrics
        speedup = exhaustive_time / binary_time if binary_time > 0 else 0
        time_reduction = ((exhaustive_time - binary_time) / exhaustive_time * 100) if exhaustive_time > 0 else 0
        
        return {
            "binary_time": binary_time,
            "exhaustive_time": exhaustive_time,
            "speedup_ratio": speedup,
            "time_reduction_percent": time_reduction
        }
    
    def calculate_accuracy(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate basic accuracy metrics"""
        
        if len(predictions) != len(labels):
            logger.error("Predictions and labels must have same length")
            return {}
            
        # Convert to numpy arrays
        preds = np.array(predictions)
        true_labels = np.array(labels)
        
        # Calculate accuracy
        correct = np.sum(preds == true_labels)
        total = len(true_labels)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Calculate precision and recall for binary classification
        if len(np.unique(true_labels)) == 2:
            tp = np.sum((preds == 1) & (true_labels == 1))
            fp = np.sum((preds == 1) & (true_labels == 0))
            fn = np.sum((preds == 0) & (true_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        else:
            precision = accuracy / 100  # For multi-class, use accuracy
            recall = accuracy / 100
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "correct": correct,
            "total": total
        }
    
    def compare_models(self, baseline_results: Dict, improved_results: Dict) -> Dict[str, float]:
        """Compare baseline vs improved model performance"""
        
        baseline_acc = baseline_results.get("accuracy", 0)
        improved_acc = improved_results.get("accuracy", 0)
        
        improvement = improved_acc - baseline_acc
        improvement_percent = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        return {
            "baseline_accuracy": baseline_acc,
            "improved_accuracy": improved_acc,
            "absolute_improvement": improvement,
            "relative_improvement_percent": improvement_percent
        }
    
    def measure_federated_benefits(self, fed_accuracy: float, local_accuracies: List[float]) -> Dict[str, float]:
        """Measure benefits of federated learning vs local training"""
        
        local_avg = np.mean(local_accuracies) if local_accuracies else 0
        local_std = np.std(local_accuracies) if local_accuracies else 0
        
        collaboration_gain = fed_accuracy - local_avg
        consistency_improvement = max(0, min(local_accuracies) - (local_avg - local_std)) if local_accuracies else 0
        
        return {
            "federated_accuracy": fed_accuracy,
            "local_average_accuracy": local_avg,
            "local_std": local_std,
            "collaboration_gain": collaboration_gain,
            "consistency_improvement": consistency_improvement
        }
    
    def track_training_progress(self, round_metrics: List[Dict]) -> Dict[str, Any]:
        """Track training progress over rounds"""
        
        if not round_metrics:
            return {}
            
        rounds = [m.get("round", 0) for m in round_metrics]
        accuracies = [m.get("avg_train_accuracy", 0) for m in round_metrics]
        losses = [m.get("avg_train_loss", 0) for m in round_metrics]
        
        # Find convergence point (when accuracy > 85%)
        convergence_round = None
        for i, acc in enumerate(accuracies):
            if acc > 0.85:
                convergence_round = rounds[i]
                break
        
        return {
            "total_rounds": len(rounds),
            "final_accuracy": accuracies[-1] if accuracies else 0,
            "final_loss": losses[-1] if losses else 0,
            "convergence_round": convergence_round,
            "converged": convergence_round is not None,
            "accuracy_improvement": accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
        }
    
    def measure_system_performance(self, response_times: Dict[str, List[float]], 
                                 communication_costs: List[float]) -> Dict[str, Any]:
        """Measure practical system performance"""
        
        device_performance = {}
        for device_type, times in response_times.items():
            if times:
                device_performance[device_type] = {
                    "avg_response_time": np.mean(times),
                    "max_response_time": np.max(times),
                    "min_response_time": np.min(times)
                }
        
        comm_stats = {}
        if communication_costs:
            comm_stats = {
                "avg_communication_mb": np.mean(communication_costs),
                "total_communication_mb": np.sum(communication_costs),
                "max_round_communication_mb": np.max(communication_costs)
            }
        
        return {
            "device_performance": device_performance,
            "communication_stats": comm_stats
        }
    
    def generate_summary_report(self, all_results: Dict[str, Any]) -> str:
        """Generate a simple summary report"""
        
        report = "=== FedVideoQA Evaluation Summary ===\n\n"
        
        # Processing efficiency
        if "processing_efficiency" in all_results:
            pe = all_results["processing_efficiency"]
            report += f"Processing Efficiency:\n"
            report += f"  - Speedup: {pe.get('speedup_ratio', 0):.2f}x faster\n"
            report += f"  - Time Reduction: {pe.get('time_reduction_percent', 0):.1f}%\n\n"
        
        # QA Performance
        if "qa_performance" in all_results:
            qa = all_results["qa_performance"]
            report += f"Question Answering Performance:\n"
            report += f"  - Accuracy: {qa.get('accuracy', 0):.1f}%\n"
            report += f"  - Precision: {qa.get('precision', 0):.3f}\n"
            report += f"  - Recall: {qa.get('recall', 0):.3f}\n\n"
        
        # Federated benefits
        if "federated_benefits" in all_results:
            fb = all_results["federated_benefits"]
            report += f"Federated Learning Benefits:\n"
            report += f"  - Collaboration Gain: +{fb.get('collaboration_gain', 0):.1f}%\n"
            report += f"  - Final Accuracy: {fb.get('federated_accuracy', 0):.1f}%\n\n"
        
        # Training progress
        if "training_progress" in all_results:
            tp = all_results["training_progress"]
            report += f"Training Progress:\n"
            report += f"  - Total Rounds: {tp.get('total_rounds', 0)}\n"
            report += f"  - Converged: {'Yes' if tp.get('converged', False) else 'No'}\n"
            if tp.get('convergence_round'):
                report += f"  - Convergence Round: {tp.get('convergence_round')}\n"
            report += f"  - Accuracy Improvement: +{tp.get('accuracy_improvement', 0):.1f}%\n\n"
        
        return report 