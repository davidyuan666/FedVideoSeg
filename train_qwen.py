"""
Training Script for FedVideoQA
binary search + Qwen2.5-VL fine-tuning + federated learning
Supports both fine-tuning and alignment tuning
"""

import argparse
import logging
import json
import torch
import flwr as fl
from pathlib import Path
from typing import List, Tuple, Dict, Any
from enum import Enum

from src.core.binary_search import BinarySearchLocalizer
from src.core.deepseek_client import DeepSeekClient
from src.federated.fed_client import FedClient
from src.federated.fed_server import FedServer
from src.models.qwen_finetuner import QwenFineTuner, QwenAlignmentTuner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMode(Enum):
    """Training modes supported"""
    FINETUNE = "finetune"
    ALIGNMENT = "alignment"
    BOTH = "both"

class FedVideoQATrainer:
    """Main trainer class for FedVideoQA with Qwen model support"""
    
    def __init__(self, args):
        self.args = args
        self.training_mode = TrainingMode(args.training_mode)
        self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.deepseek_client = DeepSeekClient()
        self.localizer = BinarySearchLocalizer(self.deepseek_client)
        
        # Initialize tuners based on mode
        if self.training_mode in [TrainingMode.FINETUNE, TrainingMode.BOTH]:
            self.finetuner = QwenFineTuner(
                model_name=self.model_name,
                lora_config={
                    "r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "target_modules": args.lora_targets,
                    "lora_dropout": args.lora_dropout
                }
            )
        
        if self.training_mode in [TrainingMode.ALIGNMENT, TrainingMode.BOTH]:
            self.alignment_tuner = QwenAlignmentTuner(
                model_name=self.model_name,
                preference_config={
                    "dpo_beta": args.dpo_beta,
                    "max_length": args.max_length,
                    "label_smoothing": args.label_smoothing
                }
            )

    def generate_training_data(self, video_paths: List[str], questions: List[str]) -> Tuple[List[Tuple], List[Tuple]]:
        """Generate training data using binary search for both fine-tuning and alignment"""
        
        all_finetune_data = []
        all_alignment_data = []
        
        for video_path, question in zip(video_paths, questions):
            logger.info(f"Processing video: {video_path}")
            
            # Generate frame-question pairs with relevance labels for fine-tuning
            if self.training_mode in [TrainingMode.FINETUNE, TrainingMode.BOTH]:
                finetune_data = self.localizer.generate_training_data(video_path, question)
                all_finetune_data.extend(finetune_data)
                logger.info(f"Generated {len(finetune_data)} fine-tuning samples")
            
            # Generate preference pairs for alignment tuning
            if self.training_mode in [TrainingMode.ALIGNMENT, TrainingMode.BOTH]:
                alignment_data = self.localizer.generate_preference_data(video_path, question)
                all_alignment_data.extend(alignment_data)
                logger.info(f"Generated {len(alignment_data)} alignment samples")
        
        return all_finetune_data, all_alignment_data

    def create_client_fn(self, client_data: Dict[str, Dict]):
        """Create client function for federated learning"""
        
        def client_fn(cid: str):
            # Get client-specific data
            local_data = client_data.get(cid, {})
            device_type = "mobile" if "mobile" in cid else "desktop"
            
            return FedClient(
                cid=cid, 
                local_data=local_data,
                device_type=device_type,
                training_mode=self.training_mode,
                model_name=self.model_name,
                tuner_configs={
                    "lora": {
                        "r": self.args.lora_r,
                        "lora_alpha": self.args.lora_alpha,
                        "target_modules": self.args.lora_targets,
                        "lora_dropout": self.args.lora_dropout
                    },
                    "alignment": {
                        "dpo_beta": self.args.dpo_beta,
                        "max_length": self.args.max_length,
                        "label_smoothing": self.args.label_smoothing
                    }
                }
            )
        
        return client_fn

    def distribute_data(self, finetune_data: List[Tuple], alignment_data: List[Tuple]) -> Dict[str, Dict]:
        """Distribute data to federated clients"""
        
        client_data = {}
        
        # Distribute fine-tuning data
        if finetune_data:
            samples_per_client = len(finetune_data) // self.args.num_clients
            for i in range(self.args.num_clients):
                client_id = f"client_{i}"
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                
                if client_id not in client_data:
                    client_data[client_id] = {}
                client_data[client_id]["finetune"] = finetune_data[start_idx:end_idx]
        
        # Distribute alignment data
        if alignment_data:
            samples_per_client = len(alignment_data) // self.args.num_clients
            for i in range(self.args.num_clients):
                client_id = f"client_{i}"
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                
                if client_id not in client_data:
                    client_data[client_id] = {}
                client_data[client_id]["alignment"] = alignment_data[start_idx:end_idx]
        
        # Log distribution info
        for client_id, data in client_data.items():
            finetune_count = len(data.get("finetune", []))
            alignment_count = len(data.get("alignment", []))
            logger.info(f"Client {client_id}: {finetune_count} finetune, {alignment_count} alignment samples")
        
        return client_data

    def run_federated_training(self):
        """Run the complete federated training pipeline"""
        
        # 1. Generate training data using binary search
        logger.info("Step 1: Generating training data with binary search...")
        
        # Example data (replace with your actual video paths and questions)
        video_paths = [f"{self.args.data_dir}/video_{i}.mp4" for i in range(self.args.num_videos)]
        questions = [f"What is discussed in this video segment {i}?" for i in range(self.args.num_videos)]
        
        finetune_data, alignment_data = self.generate_training_data(video_paths, questions)
        
        logger.info(f"Total fine-tuning samples: {len(finetune_data)}")
        logger.info(f"Total alignment samples: {len(alignment_data)}")
        
        # 2. Distribute data to clients
        logger.info("Step 2: Distributing data to federated clients...")
        client_data = self.distribute_data(finetune_data, alignment_data)
        
        # 3. Start federated learning
        logger.info("Step 3: Starting federated learning...")
        
        strategy = FedServer(
            fraction_fit=self.args.fraction_fit,
            fraction_evaluate=self.args.fraction_evaluate,
            min_fit_clients=self.args.min_fit_clients,
            min_evaluate_clients=self.args.min_evaluate_clients,
            training_mode=self.training_mode
        )
        
        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=self.create_client_fn(client_data),
            num_clients=self.args.num_clients,
            config=fl.server.ServerConfig(num_rounds=self.args.num_rounds),
            strategy=strategy,
            client_resources={
                "num_cpus": self.args.client_cpus, 
                "num_gpus": self.args.client_gpus
            }
        )
        
        logger.info("Training completed!")
        
        # 4. Save results
        self.save_results(history, strategy)

    def save_results(self, history, strategy):
        """Save training results and model checkpoints"""
        
        logger.info("Step 4: Saving results...")
        
        results = {
            "training_config": {
                "training_mode": self.training_mode.value,
                "model_name": self.model_name,
                "num_rounds": self.args.num_rounds,
                "num_clients": self.args.num_clients,
                "lora_config": {
                    "r": self.args.lora_r,
                    "lora_alpha": self.args.lora_alpha,
                    "target_modules": self.args.lora_targets,
                    "lora_dropout": self.args.lora_dropout
                }
            },
            "training_history": {
                "losses": dict(history.losses_distributed),
                "metrics": dict(history.metrics_distributed)
            },
            "round_metrics": getattr(strategy, 'round_metrics', {})
        }
        
        # Save training results
        results_path = f"training_results_{self.training_mode.value}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save model checkpoints if available
        if hasattr(strategy, 'get_global_model'):
            model_path = f"global_model_{self.training_mode.value}.pth"
            torch.save(strategy.get_global_model(), model_path)
            logger.info(f"Global model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="FedVideoQA Training with Qwen Fine-tuning")
    
    # Basic training arguments
    parser.add_argument("--training_mode", type=str, default="finetune", 
                       choices=["finetune", "alignment", "both"],
                       help="Training mode: finetune, alignment, or both")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Qwen model name")
    parser.add_argument("--num_rounds", type=int, default=50,
                       help="Number of federated learning rounds")
    parser.add_argument("--num_clients", type=int, default=5,
                       help="Number of federated clients")
    parser.add_argument("--data_dir", type=str, default="data/",
                       help="Data directory")
    parser.add_argument("--num_videos", type=int, default=10,
                       help="Number of videos to process")
    
    # Federated learning arguments
    parser.add_argument("--fraction_fit", type=float, default=0.8,
                       help="Fraction of clients for training")
    parser.add_argument("--fraction_evaluate", type=float, default=0.5,
                       help="Fraction of clients for evaluation")
    parser.add_argument("--min_fit_clients", type=int, default=3,
                       help="Minimum clients for training")
    parser.add_argument("--min_evaluate_clients", type=int, default=3,
                       help="Minimum clients for evaluation")
    
    # Resource allocation
    parser.add_argument("--client_cpus", type=int, default=1,
                       help="CPU cores per client")
    parser.add_argument("--client_gpus", type=float, default=0.5,
                       help="GPU allocation per client")
    
    # LoRA fine-tuning arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_targets", type=str, nargs="+", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj"],
                       help="LoRA target modules")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Alignment tuning arguments
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                       help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                       help="Label smoothing for alignment")
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = FedVideoQATrainer(args)
    trainer.run_federated_training()

if __name__ == "__main__":
    main()