"""
01_config.py - Configuration Management for Point-MAE

This module handles:
- YAML configuration file parsing
- EasyDict-based config objects
- Argument parsing for training
- Experiment directory creation
"""

import os
import yaml
import argparse
from pathlib import Path
from easydict import EasyDict


def merge_new_config(config, new_config):
    """
    Recursively merge new_config into config.
    Handles nested dictionaries and _base_ references.
    """
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file."""
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(args, logger=None):
    """Get configuration, handling resume case."""
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            if logger:
                logger.info("Failed to resume - config not found")
            raise FileNotFoundError(f"Config not found at {cfg_path}")
        if logger:
            logger.info(f'Resume yaml from {cfg_path}')
        args.config = cfg_path
    
    config = cfg_from_yaml_file(args.config)
    
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    
    return config


def save_experiment_config(args, config, logger=None):
    """Save config to experiment directory."""
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    if logger:
        logger.info(f'Copy the Config file from {args.config} to {config_path}')


def log_args_to_file(args, pre='args', logger=None):
    """Log all arguments."""
    for key, val in args.__dict__.items():
        if logger:
            logger.info(f'{pre}.{key} : {val}')


def log_config_to_file(cfg, pre='cfg', logger=None):
    """Log all config values recursively."""
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            if logger:
                logger.info(f'{pre}.{key} = edict()')
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        if logger:
            logger.info(f'{pre}.{key} : {val}')


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Point-MAE Training')
    
    parser.add_argument('--config', type=str, required=True,
                        help='yaml config file')
    parser.add_argument('--launcher', choices=['none', 'pytorch'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend')
    
    # Batch Norm
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='whether to use sync bn')
    
    # Experiment
    parser.add_argument('--exp_name', type=str, default='default',
                        help='experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type=str, default=None,
                        help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None,
                        help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    
    parser.add_argument('--vote', action='store_true', default=False,
                        help='vote acc')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='autoresume training (interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test mode for certain ckpt')
    parser.add_argument('--finetune_model', action='store_true', default=False,
                        help='finetune modelnet with pretrained weight')
    parser.add_argument('--scratch_model', action='store_true', default=False,
                        help='training modelnet from scratch')
    
    # Few-shot
    parser.add_argument('--mode', choices=['easy', 'median', 'hard', None],
                        default=None, help='difficulty mode for shapenet')
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Validation
    if args.test and args.resume:
        raise ValueError('--test and --resume cannot be both activate')
    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both activate')
    if args.test and args.ckpts is None:
        raise ValueError('ckpts shouldnt be None while test mode')
    if args.finetune_model and args.ckpts is None:
        print('training from scratch')
    
    # Set LOCAL_RANK env
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    # Experiment paths
    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode
    
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem,
                                         Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem,
                                      Path(args.config).parent.stem, 'TFBoard', args.exp_name)
    args.log_name = Path(args.config).stem
    
    create_experiment_dir(args)
    
    return args


def create_experiment_dir(args):
    """Create experiment and tensorboard directories."""
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)


if __name__ == '__main__':
    # Test configuration loading
    import sys
    if len(sys.argv) > 1:
        args = get_args()
        config = get_config(args)
        print("Config loaded successfully!")
        print(f"Model: {config.model.NAME}")
        print(f"Batch size: {config.total_bs}")
