#!/bin/bash
# =============================================================================
# Point-MAE Classification Script
# =============================================================================
# This script classifies PLY files using a trained Point-MAE model.
#
# Usage:
#   # Single file
#   ./scripts/run_classify.sh /path/to/mesh.ply
#
#   # Directory of files
#   ./scripts/run_classify.sh /path/to/meshes/
# =============================================================================

# Configuration - MODIFY THESE PATHS
CONFIG="cfgs/finetune_custom.yaml"
CHECKPOINT="experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth"
OUTPUT_CSV="predictions.csv"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_ply_or_directory> [output_csv]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/mesh.ply"
    echo "  $0 /path/to/meshes/"
    echo "  $0 /path/to/meshes/ results.csv"
    exit 1
fi

INPUT=$1
if [ $# -ge 2 ]; then
    OUTPUT_CSV=$2
fi

echo "=============================================="
echo "Point-MAE Classification"
echo "=============================================="
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Input:      $INPUT"
echo "Output:     $OUTPUT_CSV"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please train a model first or specify correct path."
    exit 1
fi

# Run classification
CUDA_VISIBLE_DEVICES=0 python tools/classify_single.py \
    --config $CONFIG \
    --ckpts $CHECKPOINT \
    --input $INPUT \
    --output $OUTPUT_CSV \
    --top_k 3

echo ""
echo "Classification complete!"
echo "Results saved to: $OUTPUT_CSV"
