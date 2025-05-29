"""
Enhanced Data Loader for FedVideoQA
Implements video processing with binary search localization
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..models.feature_extractors import MultimodalFeatureExtractor
from ..core.binary_search_localizer import BinarySearchLocalizer

logger = logging.getLogger(__name__)

class VideoQADataset(Dataset):
    """
    Enhanced VideoQA dataset with binary search localization
    """
    
    def __init__(
        self,
        data_file: str,
        video_dir: str,
        feature_extractor: MultimodalFeatureExtractor,
        localizer: BinarySearchLocalizer,
        max_videos: Optional[int] = None,
        cache_features: bool = True
    ):
        self.data_file = data_file
        self.video_dir = Path(video_dir)
        self.feature_extractor = feature_extractor
        self.localizer = localizer
        self.cache_features = cache_features
        
        # Load data
        self.data = self._load_data(max_videos)
        
        # Feature cache
        self.feature_cache = {} if cache_features else None
        
        logger.info(f"Loaded {len(self.data)} video QA samples")
    
    def _load_data(self, max_videos: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load video QA data from file"""
        if self.data_file.endswith('.json'):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        elif self.data_file.endswith('.csv'):
            df = pd.read_csv(self.data_file)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported data file format: {self.data_file}")
        
        # Limit number of videos if specified
        if max_videos is not None:
            data = data[:max_videos]
        
        # Validate data format
        required_fields = ['video_id', 'question', 'answer']
        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing field '{field}' in data item {i}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single video QA sample with binary search localization
        """
        item = self.data[idx]
        
        # Check cache first
        cache_key = f"{item['video_id']}_{item['question']}"
        if self.feature_cache and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Process video with binary search localization
        processed_item = asyncio.run(self._process_video_qa_item(item))
        
        # Cache if enabled
        if self.feature_cache:
            self.feature_cache[cache_key] = processed_item
        
        return processed_item
    
    async def _process_video_qa_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a single video QA item with binary search localization
        """
        video_path = self.video_dir / f"{item['video_id']}.mp4"
        question = item['question']
        answer = item['answer']
        
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            # Return dummy features
            return self._create_dummy_features(question, answer)
        
        try:
            # Step 1: Binary search localization
            relevant_segments = await self.localizer.localize_segments(
                video_path=str(video_path),
                question=question,
                max_segments=3
            )
            
            if not relevant_segments:
                logger.warning(f"No relevant segments found for video {item['video_id']}")
                return self._create_dummy_features(question, answer)
            
            # Step 2: Extract features from relevant segments
            visual_features_list = []
            audio_features_list = []
            
            for segment in relevant_segments:
                if segment.frames:
                    # Extract visual features using CLIP
                    visual_feat = self.feature_extractor.clip_extractor.extract_features(
                        segment.frames
                    )
                    visual_features_list.append(visual_feat)
                    
                    # Extract audio features (placeholder - would need actual audio extraction)
                    # For now, create dummy audio features
                    audio_feat = torch.randn(1, self.feature_extractor.audio_dim)
                    audio_features_list.append(audio_feat)
            
            # Combine features from all segments
            if visual_features_list:
                visual_features = torch.cat(visual_features_list, dim=0)
                audio_features = torch.cat(audio_features_list, dim=0)
            else:
                visual_features = torch.randn(1, self.feature_extractor.visual_dim)
                audio_features = torch.randn(1, self.feature_extractor.audio_dim)
            
            # Step 3: Extract text features
            text_features = self.feature_extractor.bert_extractor.extract_features([question])
            question_features = text_features.clone()
            
            # Step 4: Process answer
            if isinstance(answer, str):
                # For generative answers
                answer_features = self.feature_extractor.bert_extractor.extract_features([answer])
                answer_labels = torch.tensor([0])  # Placeholder for generation task
                task_type = "generation"
            else:
                # For multiple choice answers
                answer_labels = torch.tensor([int(answer)])
                answer_features = torch.randn(1, self.feature_extractor.text_dim)
                task_type = "classification"
            
            return {
                'visual_features': visual_features,
                'audio_features': audio_features,
                'text_features': text_features,
                'question_features': question_features,
                'answer_features': answer_features,
                'answer_labels': answer_labels,
                'task_type': task_type,
                'video_id': item['video_id'],
                'question': question,
                'relevance_scores': torch.tensor([s.relevance_score for s in relevant_segments])
            }
            
        except Exception as e:
            logger.error(f"Error processing video {item['video_id']}: {e}")
            return self._create_dummy_features(question, answer)
    
    def _create_dummy_features(self, question: str, answer: Any) -> Dict[str, torch.Tensor]:
        """Create dummy features for failed video processing"""
        return {
            'visual_features': torch.randn(1, self.feature_extractor.visual_dim),
            'audio_features': torch.randn(1, self.feature_extractor.audio_dim),
            'text_features': torch.randn(1, self.feature_extractor.text_dim),
            'question_features': torch.randn(1, self.feature_extractor.text_dim),
            'answer_features': torch.randn(1, self.feature_extractor.text_dim),
            'answer_labels': torch.tensor([0]),
            'task_type': "classification",
            'video_id': "dummy",
            'question': question,
            'relevance_scores': torch.tensor([0.0])
        }

class VideoQADataLoader:
    """
    Enhanced data loader for federated VideoQA training
    """
    
    def __init__(
        self,
        data_dir: str,
        video_dir: str,
        batch_size: int = 8,
        localizer: Optional[BinarySearchLocalizer] = None,
        num_workers: int = 2,
        train_split: float = 0.8
    ):
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir)
        self.batch_size = batch_size
        self.localizer = localizer
        self.num_workers = num_workers
        self.train_split = train_split
        
        # Initialize feature extractor
        self.feature_extractor = MultimodalFeatureExtractor()
        
        # Load client data distribution
        self.client_data = self._load_client_data_distribution()
        
    def _load_client_data_distribution(self) -> Dict[str, Dict[str, str]]:
        """Load client data distribution configuration"""
        client_config_file = self.data_dir / "client_distribution.json"
        
        if client_config_file.exists():
            with open(client_config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default distribution
            data_files = list(self.data_dir.glob("*.json")) + list(self.data_dir.glob("*.csv"))
            client_data = {}
            
            for i, data_file in enumerate(data_files):
                client_id = f"client_{i}"
                client_data[client_id] = {
                    "train_file": str(data_file),
                    "test_file": str(data_file),  # Same file for now
                    "device_type": ["mobile", "desktop", "server"][i % 3]
                }
            
            # Save default configuration
            with open(client_config_file, 'w') as f:
                json.dump(client_data, f, indent=2)
            
            return client_data
    
    def get_client_dataloaders(self, client_id: str) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders for a specific client"""
        
        if client_id not in self.client_data:
            raise ValueError(f"Client {client_id} not found in data distribution")
        
        client_config = self.client_data[client_id]
        
        # Create datasets
        train_dataset = VideoQADataset(
            data_file=client_config["train_file"],
            video_dir=self.video_dir,
            feature_extractor=self.feature_extractor,
            localizer=self.localizer,
            max_videos=100  # Limit for faster training
        )
        
        test_dataset = VideoQADataset(
            data_file=client_config["test_file"],
            video_dir=self.video_dir,
            feature_extractor=self.feature_extractor,
            localizer=self.localizer,
            max_videos=50  # Smaller test set
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
        
        return train_loader, test_loader
    
    def get_global_test_loader(self) -> DataLoader:
        """Get global test dataloader for server evaluation"""
        
        # Combine all test data
        all_test_files = []
        for client_config in self.client_data.values():
            if client_config["test_file"] not in all_test_files:
                all_test_files.append(client_config["test_file"])
        
        # Create combined dataset (use first file for simplicity)
        if all_test_files:
            global_test_dataset = VideoQADataset(
                data_file=all_test_files[0],
                video_dir=self.video_dir,
                feature_extractor=self.feature_extractor,
                localizer=self.localizer,
                max_videos=100
            )
            
            return DataLoader(
                global_test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn
            )
        else:
            raise ValueError("No test data available")
    
    def get_num_clients(self) -> int:
        """Get number of clients"""
        return len(self.client_data)
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batch processing"""
        
        # Initialize batch dictionary
        batch_dict = {}
        
        # Get all keys from first item
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['video_id', 'question', 'task_type']:
                # Keep as list for string values
                batch_dict[key] = [item[key] for item in batch]
            else:
                # Stack tensors
                tensors = [item[key] for item in batch]
                
                # Handle different tensor shapes
                if key in ['visual_features', 'audio_features']:
                    # Pad sequences to same length
                    max_seq_len = max(t.size(0) for t in tensors)
                    padded_tensors = []
                    
                    for tensor in tensors:
                        if tensor.size(0) < max_seq_len:
                            padding = torch.zeros(
                                max_seq_len - tensor.size(0), 
                                tensor.size(1)
                            )
                            padded_tensor = torch.cat([tensor, padding], dim=0)
                        else:
                            padded_tensor = tensor[:max_seq_len]
                        padded_tensors.append(padded_tensor)
                    
                    batch_dict[key] = torch.stack(padded_tensors)
                else:
                    # Direct stacking for fixed-size tensors
                    batch_dict[key] = torch.stack(tensors)
        
        return batch_dict 