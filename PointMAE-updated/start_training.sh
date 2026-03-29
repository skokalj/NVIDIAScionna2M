#!/bin/bash

# Point-MAE 6-Channel Training Script
# This script starts pretraining with 8 GPUs using nohup for background execution

# Set environment
source /home/joshi/experiments/.pointMAEenv/bin/activate

# Create logs directory
mkdir -p /home/joshi/experiments/Point-MAE/logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/home/joshi/experiments/Point-MAE/logs/pretrain_6channel_${TIMESTAMP}.log"

echo "Starting Point-MAE 6-channel pretraining..."
echo "Log file: $LOG_FILE"
echo "Config: cfgs/pretrain_modelnet_normals.yaml"
echo "GPUs: 0,1,2,3,4,5,6,7"
echo "Experiment name: pretrain_6channel"
echo ""

# Start training with nohup
cd /home/joshi/experiments/Point-MAE

nohup bash -c "
    source /home/joshi/experiments/.pointMAEenv/bin/activate && \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=29600 \
        --use_env \
        main.py \
        --config cfgs/pretrain_modelnet_normals.yaml \
        --launcher pytorch \
        --num_workers 0 \
        --exp_name pretrain_6channel
" > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if training is running:"
echo "  ps aux | grep $PID"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo ""

# Save PID for easy reference
echo $PID > /home/joshi/experiments/Point-MAE/logs/pretrain_6channel.pid
echo "PID saved to: /home/joshi/experiments/Point-MAE/logs/pretrain_6channel.pid"


# #!/bin/bash

# # Point-MAE 6-Channel Training Script
# # This script starts pretraining with 8 GPUs using nohup for background execution

# # Set environment
# source /home/joshi/experiments/.pointMAEenv/bin/activate

# # Create logs directory
# mkdir -p /home/joshi/experiments/Point-MAE/logs

# # Get timestamp for log files
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# LOG_FILE="/home/joshi/experiments/Point-MAE/logs/pretrain_6channel_${TIMESTAMP}.log"

# echo "Starting Point-MAE 6-channel pretraining..."
# echo "Log file: $LOG_FILE"
# echo "Config: cfgs/pretrain_modelnet_normals.yaml"
# echo "GPUs: 0,1,2,3,4,5,6,7"
# echo "Experiment name: pretrain_6channel"
# echo ""

# # Start training with nohup
# cd /home/joshi/experiments/Point-MAE

# nohup bash -c "
#     source /home/joshi/experiments/.pointMAEenv/bin/activate && \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
#         --nproc_per_node=8 \
#         main.py \
#         --config cfgs/pretrain_modelnet_normals.yaml \
#         --launcher pytorch \
#         --exp_name pretrain_6channel
# " > "$LOG_FILE" 2>&1 &

# # Get the process ID
# PID=$!
# echo "Training started with PID: $PID"
# echo "Log file: $LOG_FILE"
# echo ""
# echo "To monitor progress:"
# echo "  tail -f $LOG_FILE"
# echo ""
# echo "To check if training is running:"
# echo "  ps aux | grep $PID"
# echo ""
# echo "To stop training:"
# echo "  kill $PID"
# echo ""

# # Save PID for easy reference
# echo $PID > /home/joshi/experiments/Point-MAE/logs/pretrain_6channel.pid
# echo "PID saved to: /home/joshi/experiments/Point-MAE/logs/pretrain_6channel.pid"
