"""
Main training script for FedVideoQA
Implements the complete training pipeline from the paper
"""

import torch
import torch.nn as nn
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import flwr as fl
import time

from src.models.fedvideo_qa_model import FedVideoQAModel
from src.federated.fedvideo_qa_client import FedVideoQAClient
from src.federated.fedvideo_qa_server import FedVideoQAStrategy
from src.core.binary_search_localizer import BinarySearchLocalizer
from src.core.deepseek_client import DeepSeekClient
from src.utils.data_loader import VideoQADataLoader
from src.privacy.differential_privacy import DifferentialPrivacyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FedVideoQA Training")
    
    # Model parameters
    parser.add_argument("--visual_dim", type=int, default=512, help="CLIP visual dimension")
    parser.add_argument("--audio_dim", type=int, default=768, help="Whisper audio dimension") 
    parser.add_argument("--text_dim", type=int, default=768, help="BERT text dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    
    # Training parameters
    parser.add_argument("--num_rounds", type=int, default=100, help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Federated learning parameters
    parser.add_argument("--min_fit_clients", type=int, default=3, help="Minimum clients for training")
    parser.add_argument("--min_evaluate_clients", type=int, default=3, help="Minimum clients for evaluation")
    parser.add_argument("--fraction_fit", type=float, default=0.8, help="Fraction of clients for training")
    parser.add_argument("--fraction_evaluate", type=float, default=0.5, help="Fraction of clients for evaluation")
    
    # Device configuration
    parser.add_argument("--device_type", type=str, default="desktop", 
                       choices=["mobile", "desktop", "server"], help="Device type")
    parser.add_argument("--enable_privacy", action="store_true", help="Enable differential privacy")
    parser.add_argument("--privacy_epsilon", type=float, default=2.0, help="Privacy epsilon")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    
    # DeepSeek API
    parser.add_argument("--deepseek_api_key", type=str, required=True, help="DeepSeek API key")
    parser.add_argument("--deepseek_api_url", type=str, default="https://api.deepseek.com/v1", 
                       help="DeepSeek API URL")
    
    # Binary search parameters
    parser.add_argument("--min_segment_length", type=float, default=5.0, help="Minimum segment length (seconds)")
    parser.add_argument("--relevance_threshold", type=float, default=0.8, help="Relevance threshold")
    
    return parser.parse_args()

class FedVideoQATrainer:
    """Main trainer class for FedVideoQA"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing FedVideoQA components...")
        
        # Initialize model
        self.model = FedVideoQAModel(
            visual_dim=self.args.visual_dim,
            audio_dim=self.args.audio_dim,
            text_dim=self.args.text_dim,
            hidden_dim=self.args.hidden_dim,
            num_heads=self.args.num_heads,
            num_layers=self.args.num_layers,
            device_type=self.args.device_type
        ).to(self.device)
        
        logger.info(f"Model initialized with {self.model.get_model_size():,} parameters")
        
        # Initialize DeepSeek client
        self.deepseek_client = DeepSeekClient(
            api_key=self.args.deepseek_api_key,
            api_url=self.args.deepseek_api_url
        )
        
        # Initialize binary search localizer
        self.localizer = BinarySearchLocalizer(
            deepseek_client=self.deepseek_client,
            min_segment_length=self.args.min_segment_length,
            relevance_threshold=self.args.relevance_threshold
        )
        
        # Initialize privacy manager
        self.privacy_manager = None
        if self.args.enable_privacy:
            self.privacy_manager = DifferentialPrivacyManager(
                epsilon=self.args.privacy_epsilon,
                delta=1e-5,
                max_grad_norm=1.0
            )
        
        # Initialize data loader
        self.data_loader = VideoQADataLoader(
            data_dir=self.args.data_dir,
            video_dir=self.args.video_dir,
            batch_size=self.args.batch_size,
            localizer=self.localizer
        )
        
        logger.info("All components initialized successfully")
    
    def create_client_fn(self):
        """Create client function for Flower"""
        def client_fn(cid: str) -> FedVideoQAClient:
            # Load client-specific data
            train_loader, test_loader = self.data_loader.get_client_dataloaders(cid)
            
            return FedVideoQAClient(
                client_id=cid,
                model=self.model,
                localizer=self.localizer,
                train_loader=train_loader,
                test_loader=test_loader,
                device_type=self.args.device_type,
                local_epochs=self.args.local_epochs,
                learning_rate=self.args.learning_rate,
                privacy_manager=self.privacy_manager
            )
        
        return client_fn
    
    def create_evaluate_fn(self):
        """Create evaluation function for server"""
        def evaluate_fn(server_round: int, parameters, config):
            # Set model parameters
            params_dict = zip(self.model.parameters(), parameters)
            for param, new_param in params_dict:
                param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device)
            
            # Evaluate on global test set
            test_loader = self.data_loader.get_global_test_loader()
            
            self.model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        visual_features=batch['visual_features'],
                        audio_features=batch['audio_features'],
                        text_features=batch['text_features'],
                        question_features=batch['question_features'],
                        task_type=batch.get('task_type', 'classification'),
                        answer_labels=batch.get('answer_labels')
                    )
                    
                    if 'loss' in outputs:
                        total_loss += outputs['loss'].item()
                    
                    # Calculate accuracy
                    if 'answer_logits' in outputs:
                        predictions = torch.argmax(outputs['answer_logits'], dim=-1)
                        correct = (predictions == batch['answer_labels']).sum().item()
                        total_correct += correct
                    
                    total_samples += batch['answer_labels'].size(0)
            
            avg_loss = total_loss / len(test_loader)
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            return avg_loss, {"test_accuracy": accuracy}
        
        return evaluate_fn
    
    def train(self):
        """Main training loop"""
        logger.info("Starting FedVideoQA training...")
        
        # Create strategy
        strategy = FedVideoQAStrategy(
            model=self.model,
            fraction_fit=self.args.fraction_fit,
            fraction_evaluate=self.args.fraction_evaluate,
            min_fit_clients=self.args.min_fit_clients,
            min_evaluate_clients=self.args.min_evaluate_clients,
            evaluate_fn=self.create_evaluate_fn()
        )
        
        # Configure client resources
        client_resources = {"num_cpus": 1, "num_gpus": 0.5}
        if self.args.device_type == "mobile":
            client_resources = {"num_cpus": 0.5, "num_gpus": 0.1}
        elif self.args.device_type == "server":
            client_resources = {"num_cpus": 2, "num_gpus": 1.0}
        
        # Start simulation
        start_time = time.time()
        
        history = fl.simulation.start_simulation(
            client_fn=self.create_client_fn(),
            num_clients=self.data_loader.get_num_clients(),
            config=fl.server.ServerConfig(num_rounds=self.args.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )
        
        training_time = time.time() - start_time
        
        # Save results
        self._save_results(history, strategy, training_time)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return history, strategy
    
    def _save_results(self, history, strategy, training_time):
        """Save training results and analysis"""
        
        # Training summary
        summary = {
            "training_time_seconds": training_time,
            "num_rounds": self.args.num_rounds,
            "device_type": self.args.device_type,
            "privacy_enabled": self.args.enable_privacy,
            "model_parameters": {
                "total_params": self.model.get_model_size(),
                "trainable_params": self.model.get_trainable_parameters(),
                "visual_dim": self.args.visual_dim,
                "audio_dim": self.args.audio_dim,
                "text_dim": self.args.text_dim,
                "hidden_dim": self.args.hidden_dim
            },
            "strategy_summary": strategy.get_training_summary()
        }
        
        # Save summary
        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save history
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump({
                "losses_distributed": dict(history.losses_distributed),
                "losses_centralized": dict(history.losses_centralized),
                "metrics_distributed": dict(history.metrics_distributed),
                "metrics_centralized": dict(history.metrics_centralized)
            }, f, indent=2, default=str)
        
        # Save final model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "args": vars(self.args),
            "summary": summary
        }, self.output_dir / "final_model.pth")
        
        logger.info(f"Results saved to {self.output_dir}")
        
        # Print final statistics
        self._print_final_statistics(summary, history)
    
    def _print_final_statistics(self, summary, history):
        """Print final training statistics"""
        print("\n" + "="*60)
        print("FEDVIDEOQA TRAINING SUMMARY")
        print("="*60)
        
        print(f"Training Time: {summary['training_time_seconds']:.2f} seconds")
        print(f"Total Rounds: {summary['num_rounds']}")
        print(f"Device Type: {summary['device_type']}")
        print(f"Privacy Enabled: {summary['privacy_enabled']}")
        
        print(f"\nModel Configuration:")
        print(f"  Total Parameters: {summary['model_parameters']['total_params']:,}")
        print(f"  Trainable Parameters: {summary['model_parameters']['trainable_params']:,}")
        print(f"  Hidden Dimension: {summary['model_parameters']['hidden_dim']}")
        
        # Final performance
        if history.losses_centralized:
            final_loss = list(history.losses_centralized.values())[-1]
            print(f"\nFinal Centralized Loss: {final_loss:.4f}")
        
        if history.metrics_centralized:
            final_metrics = list(history.metrics_centralized.values())[-1]
            if "test_accuracy" in final_metrics:
                print(f"Final Test Accuracy: {final_metrics['test_accuracy']:.4f}")
        
        # Strategy summary
        strategy_summary = summary["strategy_summary"]
        print(f"\nStrategy Summary:")
        print(f"  Best Accuracy: {strategy_summary.get('best_accuracy', 0.0):.4f}")
        print(f"  Device Distribution: {strategy_summary.get('final_device_distribution', {})}")
        
        print("="*60)

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create trainer
    trainer = FedVideoQATrainer(args)
    
    # Start training
    try:
        history, strategy = trainer.train()
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 