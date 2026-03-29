"""
09_main.py - Main Entry Point for Point-MAE Training

This is the main script to run Point-MAE training:
- Pretraining: Self-supervised learning on ShapeNet55
- Finetuning: Classification on ModelNet40/ScanObjectNN
- Testing: Evaluate trained models

Usage:
    # Pretraining (single GPU)
    python 09_main.py --config cfgs/pretrain.yaml --exp_name pretrain_test
    
    # Pretraining (8 GPU distributed)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \\
        --nproc_per_node=8 \\
        09_main.py \\
        --config cfgs/pretrain.yaml \\
        --launcher pytorch \\
        --exp_name pretrain_8gpu
    
    # Finetuning
    python 09_main.py \\
        --config cfgs/finetune_modelnet.yaml \\
        --finetune_model \\
        --ckpts experiments/pretrain/ckpt-last.pth \\
        --exp_name finetune_modelnet
    
    # Testing
    python 09_main.py \\
        --config cfgs/finetune_modelnet.yaml \\
        --test \\
        --ckpts experiments/finetune/ckpt-best.pth \\
        --exp_name test_modelnet
"""

import os
import sys
import time
import random
import logging
import numpy as np
import torch

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logger(log_file, name='Point-MAE'):
    """Setup logging to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_dist(launcher):
    """Initialize distributed training."""
    if launcher == 'pytorch':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        return local_rank
    else:
        raise NotImplementedError(f"Launcher {launcher} not implemented")


def get_dist_info():
    """Get distributed training info."""
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def main():
    """Main entry point."""
    # Import config utilities
    from config import get_args, get_config, log_args_to_file, log_config_to_file
    
    # Parse arguments
    args = get_args()
    
    # CUDA setup
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    
    # Distributed training setup
    if args.launcher == 'none':
        args.distributed = False
        args.local_rank = 0
        args.world_size = 1
    else:
        args.distributed = True
        args.local_rank = init_dist(args.launcher)
        _, world_size = get_dist_info()
        args.world_size = world_size
    
    # Setup logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = setup_logger(log_file, name=args.log_name)
    
    # TensorBoard writers
    if not args.test:
        if args.local_rank == 0:
            from tensorboardX import SummaryWriter
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    else:
        train_writer = None
        val_writer = None
    
    # Load config
    config = get_config(args, logger)
    
    # Adjust batch size for distributed training
    if args.distributed:
        assert config.total_bs % args.world_size == 0
        config.dataset.train.others.bs = config.total_bs // args.world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // args.world_size * 2
        config.dataset.val.others.bs = config.total_bs // args.world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // args.world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
    
    # Log configuration
    log_args_to_file(args, 'args', logger)
    log_config_to_file(config, 'config', logger)
    
    logger.info(f'Distributed training: {args.distributed}')
    
    # Set random seed
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)
    
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()
    
    # Few-shot settings
    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold
    
    # Run training or testing
    if args.test:
        from train_finetune import test_net
        test_net(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            from train_finetune import run_net as finetune_run_net
            finetune_run_net(args, config, train_writer, val_writer)
        else:
            from train_pretrain import run_net as pretrain_run_net
            pretrain_run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
