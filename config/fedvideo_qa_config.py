"""
Configuration file for FedVideoQA training
"""

import torch

class FedVideoQAConfig:
    """Main configuration class for FedVideoQA"""
    
    # Model configuration
    MODEL = {
        "visual_dim": 512,      # CLIP ViT-B/32 dimension
        "audio_dim": 768,       # Whisper-base dimension
        "text_dim": 768,        # BERT-base dimension
        "hidden_dim": 512,      # Hidden dimension for fusion
        "num_heads": 8,         # Number of attention heads
        "num_layers": 6,        # Number of transformer layers
        "dropout": 0.1,         # Dropout rate
        "vocab_size": 30522,    # BERT vocabulary size
        "max_answer_length": 50,
        "num_answer_choices": 4
    }
    
    # Training configuration
    TRAINING = {
        "num_rounds": 100,      # Number of federated rounds
        "local_epochs": 5,      # Local epochs per client
        "learning_rate": 1e-4,  # Learning rate
        "batch_size": 8,        # Batch size
        "weight_decay": 1e-5,   # Weight decay
        "max_grad_norm": 1.0,   # Gradient clipping norm
        "convergence_threshold": 0.01,
        "patience": 5           # Early stopping patience
    }
    
    # Federated learning configuration
    FEDERATED = {
        "fraction_fit": 0.8,           # Fraction of clients for training
        "fraction_evaluate": 0.5,      # Fraction of clients for evaluation
        "min_fit_clients": 3,          # Minimum clients for training
        "min_evaluate_clients": 3,     # Minimum clients for evaluation
        "min_available_clients": 3,    # Minimum available clients
        "max_communication_size_mb": 100
    }
    
    # Device configuration
    DEVICE_CONFIG = {
        "mobile": {
            "visual_capability": 0.5,
            "audio_capability": 0.8,
            "text_capability": 0.9,
            "compression_ratio": 0.5,
            "sparsification_ratio": 0.1,
            "max_grad_norm": 0.5
        },
        "desktop": {
            "visual_capability": 0.8,
            "audio_capability": 0.9,
            "text_capability": 0.9,
            "compression_ratio": 0.8,
            "sparsification_ratio": 0.3,
            "max_grad_norm": 1.0
        },
        "server": {
            "visual_capability": 1.0,
            "audio_capability": 1.0,
            "text_capability": 1.0,
            "compression_ratio": 1.0,
            "sparsification_ratio": 1.0,
            "max_grad_norm": 1.0
        }
    }
    
    # Binary search configuration
    BINARY_SEARCH = {
        "min_segment_length": 5.0,    # Minimum segment length in seconds
        "max_iterations": 10,         # Maximum search iterations
        "relevance_threshold": 0.8,   # Relevance threshold for segment selection
        "max_segments": 3             # Maximum segments per video
    }
    
    # Privacy configuration
    PRIVACY = {
        "enable_differential_privacy": False,
        "epsilon": 2.0,               # Privacy epsilon
        "delta": 1e-5,                # Privacy delta
        "noise_multiplier": 0.1,      # Noise multiplier
        "max_grad_norm": 1.0          # Gradient norm for clipping
    }
    
    # Data configuration
    DATA = {
        "train_split": 0.8,           # Training data split
        "max_videos_per_client": 100, # Maximum videos per client for training
        "max_test_videos": 50,        # Maximum videos for testing
        "cache_features": True,       # Enable feature caching
        "num_workers": 2              # Number of data loading workers
    }
    
    # DeepSeek API configuration
    DEEPSEEK = {
        "api_url": "https://api.deepseek.com/v1",
        "max_concurrent_requests": 5,
        "timeout_seconds": 30,
        "retry_attempts": 3
    }
    
    # Logging configuration
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "save_logs": True,
        "log_file": "fedvideo_qa.log"
    }
    
    @classmethod
    def get_device_config(cls, device_type: str) -> dict:
        """Get configuration for specific device type"""
        if device_type not in cls.DEVICE_CONFIG:
            raise ValueError(f"Unknown device type: {device_type}")
        return cls.DEVICE_CONFIG[device_type]
    
    @classmethod
    def update_config(cls, updates: dict) -> None:
        """Update configuration with custom values"""
        for section, values in updates.items():
            if hasattr(cls, section):
                section_config = getattr(cls, section)
                section_config.update(values)
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters"""
        # Check model dimensions
        assert cls.MODEL["hidden_dim"] > 0, "Hidden dimension must be positive"
        assert cls.MODEL["num_heads"] > 0, "Number of heads must be positive"
        
        # Check training parameters
        assert 0 < cls.TRAINING["learning_rate"] < 1, "Learning rate must be in (0, 1)"
        assert cls.TRAINING["batch_size"] > 0, "Batch size must be positive"
        
        # Check federated parameters
        assert 0 < cls.FEDERATED["fraction_fit"] <= 1, "Fraction fit must be in (0, 1]"
        assert cls.FEDERATED["min_fit_clients"] > 0, "Minimum fit clients must be positive"
        
        # Check device configurations
        for device_type, config in cls.DEVICE_CONFIG.items():
            assert 0 < config["compression_ratio"] <= 1, f"Invalid compression ratio for {device_type}"
            assert 0 < config["sparsification_ratio"] <= 1, f"Invalid sparsification ratio for {device_type}"
        
        return True 