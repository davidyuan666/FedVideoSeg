"""
Simplified Training Script for FedVideoQA
Focus on binary search + Qwen2.5-VL fine-tuning + federated learning
"""

import argparse
import logging
import flwr as fl
from pathlib import Path
from typing import List, Tuple

from src.core.simple_binary_search import SimpleBinarySearchLocalizer
from src.core.deepseek_client import DeepSeekClient
from src.federated.simple_fed_client import SimpleFedClient
from src.federated.simple_fed_server import SimpleFedStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_training_data(video_paths: List[str], questions: List[str]) -> List[Tuple]:
    """Generate training data using binary search"""
    
    # Initialize DeepSeek client and localizer
    deepseek_client = DeepSeekClient()
    localizer = SimpleBinarySearchLocalizer(deepseek_client)
    
    all_training_data = []
    
    for video_path, question in zip(video_paths, questions):
        logger.info(f"Processing video: {video_path}")
        
        # Generate frame-question pairs with relevance labels
        training_data = localizer.generate_training_data(video_path, question)
        all_training_data.extend(training_data)
        
        logger.info(f"Generated {len(training_data)} training samples")
    
    return all_training_data

def create_client_fn(client_data: dict):
    """Create client function for federated learning"""
    
    def client_fn(cid: str):
        # Get client-specific data
        local_data = client_data.get(cid, [])
        device_type = "mobile" if "mobile" in cid else "desktop"
        
        return SimpleFedClient(cid, local_data, device_type)
    
    return client_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/")
    args = parser.parse_args()
    
    # 1. Generate training data using binary search
    logger.info("Step 1: Generating training data with binary search...")
    
    # Example data (replace with your actual video paths and questions)
    video_paths = [f"{args.data_dir}/video_{i}.mp4" for i in range(10)]
    questions = [f"What is discussed in this video segment {i}?" for i in range(10)]
    
    training_data = generate_training_data(video_paths, questions)
    logger.info(f"Total training samples: {len(training_data)}")
    
    # 2. Distribute data to clients
    logger.info("Step 2: Distributing data to federated clients...")
    
    client_data = {}
    samples_per_client = len(training_data) // args.num_clients
    
    for i in range(args.num_clients):
        client_id = f"client_{i}"
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data[client_id] = training_data[start_idx:end_idx]
        
        logger.info(f"Client {client_id}: {len(client_data[client_id])} samples")
    
    # 3. Start federated learning
    logger.info("Step 3: Starting federated learning...")
    
    strategy = SimpleFedStrategy(
        fraction_fit=0.8,
        fraction_evaluate=0.5,
        min_fit_clients=3,
        min_evaluate_clients=3,
    )
    
    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn(client_data),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5}
    )
    
    logger.info("Training completed!")
    
    # 4. Save results
    logger.info("Step 4: Saving results...")
    
    results = {
        "training_history": {
            "losses": dict(history.losses_distributed),
            "metrics": dict(history.metrics_distributed)
        },
        "round_metrics": strategy.round_metrics
    }
    
    import json
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Results saved to training_results.json")

if __name__ == "__main__":
    main() 