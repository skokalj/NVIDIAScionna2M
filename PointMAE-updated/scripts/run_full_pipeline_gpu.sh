#!/bin/bash
# =============================================================================
# Point-MAE Full Pipeline Script (GPU Version - For NVIDIA Server)
# =============================================================================
# This script runs the complete pipeline on GPU:
#   1. Preprocess PLY meshes to Point-MAE format
#   2. Finetune pretrained Point-MAE model
#   3. Evaluate and classify
#
# Requirements:
#   - CUDA-capable GPU
#   - pointnet2_ops installed
#   - knn_cuda installed
#   - Pretrained checkpoint
#
# Usage:
#   ./scripts/run_full_pipeline_gpu.sh
# =============================================================================

set -e

# Configuration
INPUT_DIR="${INPUT_DIR:-/path/to/meshes}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/processed_data}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-pretrain.pth}"
EXP_NAME="${EXP_NAME:-finetune_custom}"
NUM_GPUS="${NUM_GPUS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Point-MAE Full Pipeline (GPU Version)"
echo "============================================================"
echo "Input meshes:    $INPUT_DIR"
echo "Output data:     $OUTPUT_DIR"
echo "Pretrained:      $PRETRAINED_CKPT"
echo "Experiment:      $EXP_NAME"
echo "GPUs:            $NUM_GPUS"
echo "============================================================"

# Step 1: Preprocessing
echo ""
echo "STEP 1: Preprocessing PLY meshes"
echo "============================================================"
python tools/preprocess_custom_dataset.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --n_points 8192 \
    --dataset_name custom \
    --n_workers 32

# Update config with data path
echo ""
echo "Updating configuration files..."
# Note: User should manually update cfgs/dataset_configs/CustomDataset.yaml
# with DATA_PATH and NUM_CATEGORY

# Step 2: Finetuning
echo ""
echo "STEP 2: Finetuning pretrained model"
echo "============================================================"

if [ $NUM_GPUS -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29501 \
        main.py \
        --config cfgs/finetune_custom.yaml \
        --finetune_model \
        --ckpts "$PRETRAINED_CKPT" \
        --launcher pytorch \
        --exp_name "$EXP_NAME"
else
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config cfgs/finetune_custom.yaml \
        --finetune_model \
        --ckpts "$PRETRAINED_CKPT" \
        --exp_name "$EXP_NAME"
fi

# Step 3: Test evaluation
echo ""
echo "STEP 3: Test evaluation"
echo "============================================================"
CKPT_PATH="experiments/finetune_custom/cfgs/$EXP_NAME/ckpt-best.pth"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --test \
    --config cfgs/finetune_custom.yaml \
    --ckpts "$CKPT_PATH" \
    --exp_name "test_$EXP_NAME"

# Step 4: Classification
echo ""
echo "STEP 4: Classifying meshes"
echo "============================================================"
python tools/classify_single.py \
    --config cfgs/finetune_custom.yaml \
    --ckpts "$CKPT_PATH" \
    --input "$INPUT_DIR" \
    --output "experiments/finetune_custom/cfgs/$EXP_NAME/classification_results.csv"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
