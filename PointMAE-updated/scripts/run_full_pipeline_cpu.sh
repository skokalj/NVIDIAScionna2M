#!/bin/bash
# =============================================================================
# Point-MAE Full Pipeline Script (CPU Version)
# =============================================================================
# This script runs the complete pipeline on CPU:
#   1. Preprocess PLY meshes to Point-MAE format
#   2. Train a simplified model from scratch
#   3. Classify all meshes using the trained model
#
# Usage:
#   ./scripts/run_full_pipeline_cpu.sh
#
# Or with custom paths:
#   INPUT_DIR=/path/to/meshes OUTPUT_DIR=/path/to/output ./scripts/run_full_pipeline_cpu.sh
# =============================================================================

set -e  # Exit on error

# Configuration - Modify these as needed
INPUT_DIR="${INPUT_DIR:-/app/sid_gigs/nvidia_brev_pont_mae_class/sean_updated_blender_labels/meshes}"
OUTPUT_DIR="${OUTPUT_DIR:-data/custom_processed}"
EXP_NAME="${EXP_NAME:-cpu_test}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
N_POINTS="${N_POINTS:-512}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Point-MAE Full Pipeline (CPU Version)"
echo "============================================================"
echo "Input meshes:    $INPUT_DIR"
echo "Output data:     $OUTPUT_DIR"
echo "Experiment:      $EXP_NAME"
echo "Epochs:          $EPOCHS"
echo "Batch size:      $BATCH_SIZE"
echo "Points:          $N_POINTS"
echo "============================================================"
echo ""

# Step 1: Preprocessing
echo "============================================================"
echo "STEP 1: Preprocessing PLY meshes"
echo "============================================================"
python tools/preprocess_custom_dataset.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --n_points 8192 \
    --dataset_name custom \
    --n_workers 4

echo ""
echo "Preprocessing complete!"
echo ""

# Step 2: Training
echo "============================================================"
echo "STEP 2: Training model from scratch (CPU)"
echo "============================================================"
python tools/train_cpu_simple.py \
    --data_path "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --npoints "$N_POINTS"

echo ""
echo "Training complete!"
echo ""

# Step 3: Classification
echo "============================================================"
echo "STEP 3: Classifying all meshes"
echo "============================================================"
CKPT_PATH="experiments/cpu_training/$EXP_NAME/ckpt-best.pth"
OUTPUT_JSON="experiments/cpu_training/$EXP_NAME/classification_results.json"

python tools/classify_cpu_simple.py \
    --ckpts "$CKPT_PATH" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_JSON" \
    --n_points "$N_POINTS"

echo ""
echo "Classification complete!"
echo ""

# Summary
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Generated files:"
echo "  - Processed data:       $OUTPUT_DIR/"
echo "  - Model checkpoint:     experiments/cpu_training/$EXP_NAME/ckpt-best.pth"
echo "  - Training curves:      experiments/cpu_training/$EXP_NAME/training_curves.png"
echo "  - Training history:     experiments/cpu_training/$EXP_NAME/training_history.json"
echo "  - Classification:       $OUTPUT_JSON"
echo ""
echo "To view training curves:"
echo "  open experiments/cpu_training/$EXP_NAME/training_curves.png"
echo ""
echo "============================================================"
