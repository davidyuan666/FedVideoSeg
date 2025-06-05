"""
Qwen2.5-VL Fine-tuning and Alignment Tuning Components
Supports both supervised fine-tuning and preference-based alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class QwenFineTuner:
    """
    Qwen2.5-VL Fine-tuner for frame relevance classification
    Supports LoRA-based efficient fine-tuning
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", lora_config: Dict = None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load base model
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Configure LoRA
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1
            }
        
        self.lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, self.lora_config)
        
        # Add classification head
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification
        ).to(self.device)
        
        logger.info(f"QwenFineTuner initialized with LoRA config: {lora_config}")
    
    def encode_batch(self, images: List, questions: List[str]) -> torch.Tensor:
        """Encode a batch of image-question pairs"""
        # Prepare inputs
        inputs = self.processor(
            text=questions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Get features from model
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Pool features (mean pooling over sequence length)
        pooled_features = hidden_states.mean(dim=1)
        return pooled_features
    
    def forward(self, images: List, questions: List[str]) -> torch.Tensor:
        """Forward pass for classification"""
        features = self.encode_batch(images, questions)
        logits = self.classifier(features)
        return logits
    
    def compute_loss(self, images: List, questions: List[str], labels: torch.Tensor) -> torch.Tensor:
        """Compute classification loss"""
        logits = self.forward(images, questions)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def predict(self, images: List, questions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions"""
        self.model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            logits = self.forward(images, questions)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions, probs
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Get trainable parameters (LoRA + classifier)"""
        params = {}
        
        # LoRA parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[f"model.{name}"] = param
        
        # Classifier parameters
        for name, param in self.classifier.named_parameters():
            params[f"classifier.{name}"] = param
        
        return params
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load model state dict"""
        model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        classifier_state = {k[11:]: v for k, v in state_dict.items() if k.startswith("classifier.")}
        
        self.model.load_state_dict(model_state, strict=False)
        self.classifier.load_state_dict(classifier_state, strict=False)
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict"""
        state = {}
        
        # LoRA parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state[f"model.{name}"] = param.data
        
        # Classifier parameters
        for name, param in self.classifier.named_parameters():
            state[f"classifier.{name}"] = param.data
        
        return state

class QwenAlignmentTuner:
    """
    Qwen2.5-VL Alignment Tuner using Direct Preference Optimization (DPO)
    For aligning model preferences with human feedback
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", preference_config: Dict = None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load models (reference and policy)
        self.reference_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.policy_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Configure preference learning
        if preference_config is None:
            preference_config = {
                "dpo_beta": 0.1,
                "max_length": 512,
                "label_smoothing": 0.0
            }
        
        self.preference_config = preference_config
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        logger.info(f"QwenAlignmentTuner initialized with config: {preference_config}")
    
    def compute_dpo_loss(self, 
                        preferred_images: List, 
                        preferred_questions: List[str],
                        rejected_images: List,
                        rejected_questions: List[str]) -> torch.Tensor:
        """Compute DPO loss for preference optimization"""
        
        beta = self.preference_config["dpo_beta"]
        
        # Get policy model outputs
        policy_preferred_logprobs = self._get_log_probs(
            self.policy_model, preferred_images, preferred_questions
        )
        policy_rejected_logprobs = self._get_log_probs(
            self.policy_model, rejected_images, rejected_questions
        )
        
        # Get reference model outputs
        with torch.no_grad():
            ref_preferred_logprobs = self._get_log_probs(
                self.reference_model, preferred_images, preferred_questions
            )
            ref_rejected_logprobs = self._get_log_probs(
                self.reference_model, rejected_images, rejected_questions
            )
        
        # Compute DPO loss
        policy_ratio_preferred = policy_preferred_logprobs - ref_preferred_logprobs
        policy_ratio_rejected = policy_rejected_logprobs - ref_rejected_logprobs
        
        logits = beta * (policy_ratio_preferred - policy_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()
        
        return loss
    
    def _get_log_probs(self, model, images: List, questions: List[str]) -> torch.Tensor:
        """Get log probabilities from model"""
        # Prepare inputs
        inputs = self.processor(
            text=questions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.preference_config["max_length"]
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Get model outputs
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get sequence-level log probabilities (sum over tokens)
        if "labels" in inputs:
            labels = inputs["labels"]
            # Mask padding tokens
            mask = (labels != self.processor.tokenizer.pad_token_id)
            log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            log_probs = (log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
        else:
            # If no labels, use mean log prob over sequence
            log_probs = log_probs.mean(dim=1).mean(dim=-1)
        
        return log_probs
    
    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Get trainable parameters (only policy model)"""
        params = {}
        for name, param in self.policy_model.named_parameters():
            if param.requires_grad:
                params[f"policy.{name}"] = param
        return params
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load policy model state dict"""
        policy_state = {k[7:]: v for k, v in state_dict.items() if k.startswith("policy.")}
        self.policy_model.load_state_dict(policy_state, strict=False)
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get policy model state dict"""
        state = {}
        for name, param in self.policy_model.named_parameters():
            if param.requires_grad:
                state[f"policy.{name}"] = param.data
        return state

class SimpleQwenFrameClassifier(nn.Module):
    """
    Simplified Qwen Frame Classifier for federated learning
    Combines fine-tuning and alignment capabilities
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", training_mode: str = "finetune"):
        super().__init__()
        
        self.training_mode = training_mode
        self.model_name = model_name
        
        if training_mode in ["finetune", "both"]:
            self.finetuner = QwenFineTuner(model_name)
        
        if training_mode in ["alignment", "both"]:
            self.alignment_tuner = QwenAlignmentTuner(model_name)
    
    def forward(self, images: List, questions: List[str], mode: str = "finetune"):
        """Forward pass based on training mode"""
        if mode == "finetune" and hasattr(self, 'finetuner'):
            return self.finetuner.forward(images, questions)
        elif mode == "alignment" and hasattr(self, 'alignment_tuner'):
            # For alignment, return policy model outputs
            return self.alignment_tuner.policy_model(
                **self.alignment_tuner.processor(
                    text=questions, images=images, return_tensors="pt", padding=True
                )
            ).logits
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def compute_loss(self, data: Dict[str, Any], mode: str = "finetune"):
        """Compute loss based on training mode"""
        if mode == "finetune" and hasattr(self, 'finetuner'):
            return self.finetuner.compute_loss(
                data["images"], data["questions"], data["labels"]
            )
        elif mode == "alignment" and hasattr(self, 'alignment_tuner'):
            return self.alignment_tuner.compute_dpo_loss(
                data["preferred_images"], data["preferred_questions"],
                data["rejected_images"], data["rejected_questions"]
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def get_trainable_parameters(self, mode: str = "finetune"):
        """Get trainable parameters based on mode"""
        if mode == "finetune" and hasattr(self, 'finetuner'):
            return self.finetuner.get_trainable_parameters()
        elif mode == "alignment" and hasattr(self, 'alignment_tuner'):
            return self.alignment_tuner.get_trainable_parameters()
        else:
            return {}
    
    def predict(self, images: List, questions: List[str], mode: str = "finetune"):
        """Make predictions based on mode"""
        if mode == "finetune" and hasattr(self, 'finetuner'):
            return self.finetuner.predict(images, questions)
        else:
            # Fallback to simple prediction
            logits = self.forward(images, questions, mode)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            return predictions, probs 