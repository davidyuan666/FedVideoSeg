"""
Simple Federated Training Example
"""

import torch
import asyncio
from pathlib import Path

from src.models.multimodal_model import MultimodalVideoQAModel
from src.core.simple_video_processor import SimpleVideoProcessor
from src.core.deepseek_client import DeepSeekClient
from src.federated.simple_client import SimpleFedClient
from src.federated.simple_server import start_simple_server
from src.privacy.simple_privacy import SimplePrivacyManager

async def main():
    """Simple federated training example."""
    
    # Initialize components
    model = MultimodalVideoQAModel(
        visual_dim=512,
        audio_dim=768,
        text_dim=768,
        hidden_dim=256,  # Reduced complexity
        num_classes=1000
    )
    
    # Simple privacy manager
    privacy_manager = SimplePrivacyManager(enable_anonymization=True)
    
    # DeepSeek client (optional)
    deepseek_client = DeepSeekClient(api_key="your-api-key")
    
    # Video processor
    video_processor = SimpleVideoProcessor(deepseek_client)
    
    # Process a sample video
    video_path = "sample_video.mp4"
    question = "What is the main topic of this video?"
    
    if Path(video_path).exists():
        result = await video_processor.process_video_simple(video_path, question)
        print(f"Processed {result['total_segments']} segments")
    
    print("Simple FedVideoQA framework initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 