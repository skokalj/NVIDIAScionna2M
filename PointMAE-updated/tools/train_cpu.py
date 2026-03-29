#!/usr/bin/env python
"""
train_cpu.py - CPU-compatible training script for Point-MAE classification

This script trains a PointTransformer model from scratch on CPU for testing purposes.
It uses a simplified model architecture and fewer points for faster execution.

Usage:
    python tools/train_cpu.py \
        --config cfgs/finetune_custom_cpu.yaml \
        --exp_name cpu_test \
        --epochs 2
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import cfg_from_yaml_file
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import print_log, get_root_logger
from utils.AverageMeter import AverageMeter

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def create_dataloader(config, split='train', batch_size=2, num_workers=0):
    """Create dataloader for the specified split."""
    if split == 'train':
        dataset_config = config.dataset.train
    elif split == 'val':
        dataset_config = config.dataset.val
    else:
        dataset_config = config.dataset.test
    
    dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        drop_last=(split == 'train')
    )
    
    return dataset, dataloader


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter(['loss', 'acc'])
    
    for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
        points = data[0].to(device)
        label = data[1].to(device)
        
        # Forward pass
        logits = model(points)
        loss, acc = model.get_loss_acc(logits, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update([loss.item(), acc.item()])
        
        if idx % 5 == 0:
            print_log(f'  [Epoch {epoch}][Batch {idx+1}/{len(dataloader)}] '
                     f'Loss: {loss.item():.4f}, Acc: {acc.item():.2f}%', 
                     logger=logger)
    
    return losses.avg(0), losses.avg(1)


def validate(model, dataloader, device, epoch, logger):
    """Validate the model."""
    model.eval()
    
    test_pred = []
    test_label = []
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            points = data[0].to(device)
            label = data[1].to(device)
            
            logits = model(points)
            pred = logits.argmax(-1)
            
            test_pred.append(pred)
            test_label.append(label)
    
    test_pred = torch.cat(test_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    
    acc = (test_pred == test_label).sum().float() / float(test_label.size(0)) * 100.
    
    print_log(f'[Validation] Epoch {epoch}: Accuracy = {acc:.2f}%', logger=logger)
    
    return acc.item()


def plot_training_curves(history, output_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description='CPU Training for Point-MAE')
    parser.add_argument('--config', type=str, default='cfgs/finetune_custom_cpu.yaml',
                        help='Path to config file')
    parser.add_argument('--exp_name', type=str, default='cpu_test',
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = os.path.join('experiments', 'cpu_training', args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(output_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logger = get_root_logger(log_file=log_file, name='CPU_Train')
    
    print_log("=" * 60, logger=logger)
    print_log("Point-MAE CPU Training (From Scratch)", logger=logger)
    print_log("=" * 60, logger=logger)
    print_log(f"Config: {args.config}", logger=logger)
    print_log(f"Experiment: {args.exp_name}", logger=logger)
    print_log(f"Epochs: {args.epochs}", logger=logger)
    print_log(f"Output: {output_dir}", logger=logger)
    print_log("=" * 60, logger=logger)
    
    # Device
    device = torch.device('cpu')
    print_log(f"Using device: {device}", logger=logger)
    
    # Load config
    config = cfg_from_yaml_file(args.config)
    
    # Override epochs
    config.max_epoch = args.epochs
    
    # Create datasets and dataloaders
    print_log("\nLoading datasets...", logger=logger)
    train_dataset, train_loader = create_dataloader(
        config, 'train', args.batch_size, args.num_workers
    )
    val_dataset, val_loader = create_dataloader(
        config, 'val', args.batch_size, args.num_workers
    )
    
    print_log(f"Training samples: {len(train_dataset)}", logger=logger)
    print_log(f"Validation samples: {len(val_dataset)}", logger=logger)
    
    # Get class names
    if hasattr(train_dataset, 'get_class_names'):
        class_names = train_dataset.get_class_names()
    else:
        class_names = [f'class_{i}' for i in range(config.model.cls_dim)]
    print_log(f"Classes ({len(class_names)}): {class_names}", logger=logger)
    
    # Build model
    print_log("\nBuilding model...", logger=logger)
    model = build_model_from_cfg(config.model)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"Total parameters: {total_params:,}", logger=logger)
    print_log(f"Trainable parameters: {trainable_params:,}", logger=logger)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.05
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    
    # Training loop
    print_log("\n" + "=" * 60, logger=logger)
    print_log("Starting training...", logger=logger)
    print_log("=" * 60, logger=logger)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print_log(f"\n[Epoch {epoch}/{args.epochs}]", logger=logger)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )
        
        # Validate
        val_acc = validate(model, val_loader, device, epoch, logger)
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        print_log(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
                 f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                 f"Time: {epoch_time:.1f}s", logger=logger)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(output_dir, 'ckpt-best.pth')
            torch.save({
                'epoch': epoch,
                'base_model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': args.config
            }, checkpoint_path)
            print_log(f"  -> New best model saved! (Acc: {best_acc:.2f}%)", logger=logger)
    
    # Save last checkpoint
    checkpoint_path = os.path.join(output_dir, 'ckpt-last.pth')
    torch.save({
        'epoch': args.epochs,
        'base_model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': args.config
    }, checkpoint_path)
    print_log(f"\nLast checkpoint saved to: {checkpoint_path}", logger=logger)
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print_log(f"Training history saved to: {history_path}", logger=logger)
    
    # Plot training curves
    plot_path = plot_training_curves(history, output_dir)
    
    # Summary
    print_log("\n" + "=" * 60, logger=logger)
    print_log("Training Complete!", logger=logger)
    print_log("=" * 60, logger=logger)
    print_log(f"Best Validation Accuracy: {best_acc:.2f}%", logger=logger)
    print_log(f"Total Training Time: {sum(history['epoch_time']):.1f}s", logger=logger)
    print_log(f"Checkpoints: {output_dir}", logger=logger)
    print_log(f"Training curves: {plot_path}", logger=logger)
    print_log(f"Log file: {log_file}", logger=logger)
    print_log("=" * 60, logger=logger)
    
    return output_dir, best_acc


if __name__ == '__main__':
    main()
