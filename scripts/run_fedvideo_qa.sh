#!/bin/bash

# FedVideoQA Training Script
# This script runs the complete FedVideoQA training pipeline

set -e

echo "Starting FedVideoQA Training..."

# Configuration
DATA_DIR="./data/video_qa"
VIDEO_DIR="./data/videos"
OUTPUT_DIR="./outputs/$(date +%Y%m%d_%H%M%S)"
DEEPSEEK_API_KEY="your_deepseek_api_key_here"

# Create output directory
mkdir -p $OUTPUT_DIR

# Device configuration (change as needed)
DEVICE_TYPE="desktop"  # Options: mobile, desktop, server

# Training parameters
NUM_ROUNDS=50
LOCAL_EPOCHS=5
BATCH_SIZE=4
LEARNING_RATE=1e-4

# Federated learning parameters
MIN_FIT_CLIENTS=2
MIN_EVALUATE_CLIENTS=2
FRACTION_FIT=0.8

# Privacy settings
ENABLE_PRIVACY=false
PRIVACY_EPSILON=2.0

echo "Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Video Directory: $VIDEO_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Device Type: $DEVICE_TYPE"
echo "  Number of Rounds: $NUM_ROUNDS"
echo "  Privacy Enabled: $ENABLE_PRIVACY"

# Run training
python train_fedvideo_qa.py \
    --data_dir $DATA_DIR \
    --video_dir $VIDEO_DIR \
    --output_dir $OUTPUT_DIR \
    --deepseek_api_key $DEEPSEEK_API_KEY \
    --device_type $DEVICE_TYPE \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --min_fit_clients $MIN_FIT_CLIENTS \
    --min_evaluate_clients $MIN_EVALUATE_CLIENTS \
    --fraction_fit $FRACTION_FIT \
    $([ "$ENABLE_PRIVACY" = true ] && echo "--enable_privacy --privacy_epsilon $PRIVACY_EPSILON")

echo "Training completed! Results saved to: $OUTPUT_DIR"

# Optional: Run evaluation
echo "Running evaluation..."
python scripts/evaluate_model.py \
    --model_path $OUTPUT_DIR/final_model.pth \
    --data_dir $DATA_DIR \
    --video_dir $VIDEO_DIR \
    --output_dir $OUTPUT_DIR

echo "Evaluation completed!" 