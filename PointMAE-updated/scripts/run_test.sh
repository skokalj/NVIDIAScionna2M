#!/bin/bash
# =============================================================================
# Point-MAE Test Script
# =============================================================================
# This script evaluates a trained Point-MAE model on the test set.
#
# Usage:
#   ./scripts/run_test.sh
# =============================================================================

# Configuration - MODIFY THESE PATHS
CONFIG="cfgs/finetune_custom.yaml"
CHECKPOINT="experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth"
EXP_NAME="test_custom"

echo "=============================================="
echo "Point-MAE Test Evaluation"
echo "=============================================="
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please train a model first or specify correct path."
    exit 1
fi

# Run test
CUDA_VISIBLE_DEVICES=0 python main.py \
    --test \
    --config $CONFIG \
    --ckpts $CHECKPOINT \
    --exp_name $EXP_NAME

echo ""
echo "Test evaluation complete!"
