#!/bin/bash
# =============================================================================
# Point-MAE Custom Dataset Finetuning Script
# =============================================================================
# This script finetunes a pretrained Point-MAE model on your custom dataset
# for classification.
#
# Prerequisites:
#   1. Preprocess your data using tools/preprocess_custom_dataset.py
#   2. Update cfgs/dataset_configs/CustomDataset.yaml with your data path
#   3. Update cfgs/finetune_custom.yaml with correct cls_dim (number of classes)
#   4. Have a pretrained checkpoint ready
#
# Usage:
#   ./scripts/run_finetune_custom.sh
# =============================================================================

# Configuration - MODIFY THESE PATHS
PRETRAINED_CKPT="experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth"
# Alternative: Use official pretrained model
# PRETRAINED_CKPT="pretrain.pth"  # Download from Point-MAE releases

CONFIG="cfgs/finetune_custom.yaml"
EXP_NAME="finetune_custom"
NUM_GPUS=1  # Change to number of available GPUs

# Create logs directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/finetune_custom_${TIMESTAMP}.log"

echo "=============================================="
echo "Point-MAE Custom Dataset Finetuning"
echo "=============================================="
echo "Config:          $CONFIG"
echo "Pretrained:      $PRETRAINED_CKPT"
echo "Experiment:      $EXP_NAME"
echo "GPUs:            $NUM_GPUS"
echo "Log file:        $LOG_FILE"
echo "=============================================="

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    echo "Please download or specify correct path."
    exit 1
fi

# Run training
if [ $NUM_GPUS -gt 1 ]; then
    # Multi-GPU training with distributed data parallel
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29501 \
        main.py \
        --config $CONFIG \
        --finetune_model \
        --ckpts $PRETRAINED_CKPT \
        --launcher pytorch \
        --exp_name $EXP_NAME \
        2>&1 | tee $LOG_FILE
else
    # Single GPU training
    echo "Starting single-GPU training..."
    
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config $CONFIG \
        --finetune_model \
        --ckpts $PRETRAINED_CKPT \
        --exp_name $EXP_NAME \
        2>&1 | tee $LOG_FILE
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "Log file: $LOG_FILE"
echo "Checkpoints: experiments/finetune_custom/cfgs/$EXP_NAME/"
echo "=============================================="
