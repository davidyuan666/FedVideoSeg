"""
Simplified Qwen2.5-VL Frame Relevance Classifier
Fine-tunes Qwen2.5-VL for binary frame-question relevance classification
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import logging

logger = logging.getLogger(__name__)

class SimpleQwenFrameClassifier(nn.Module):
    """
    Simple frame relevance classifier using Qwen2.5-VL
    Binary classification: can this frame answer the question?
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__()
        
        # Load base model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Add LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        
        # Add classification head
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification
        )
        
    def forward(self, images, questions):
        """
        Forward pass for frame relevance classification
        Args:
            images: List of PIL images
            questions: List of question strings
        Returns:
            logits: (batch_size, 2) - [not_relevant, relevant]
        """
        # Process inputs
        inputs = self.processor(
            text=questions,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Get features from base model
        with torch.no_grad():
            outputs = self.base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
        # Pool features (simple mean pooling)
        pooled_features = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        return logits
    
    def predict_relevance(self, image, question):
        """
        Predict if a single frame can answer the question
        Returns: (is_relevant, confidence)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward([image], [question])
            probs = torch.softmax(logits, dim=-1)
            is_relevant = torch.argmax(logits, dim=-1).item()
            confidence = probs.max().item()
            
        return is_relevant, confidence 