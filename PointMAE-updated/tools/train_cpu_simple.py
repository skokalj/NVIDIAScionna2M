#!/usr/bin/env python
"""
train_cpu_simple.py - Simplified CPU-only training script for Point-MAE classification

This script trains a simplified PointTransformer model from scratch on CPU.
It does NOT require CUDA extensions (pointnet2_ops, knn_cuda).

Usage:
    python tools/train_cpu_simple.py \
        --data_path data/custom_processed \
        --exp_name cpu_test \
        --epochs 2
"""

import os
import sys
import argparse
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Simple Point Cloud Dataset (no CUDA dependencies)
# ============================================================================

def pc_normalize(pc):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def farthest_point_sample_numpy(points, npoint):
    """FPS in numpy (CPU-only)."""
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return points[centroids]


class SimplePointDataset(Dataset):
    """Simple dataset loader without CUDA dependencies."""
    
    def __init__(self, data_path, split='train', npoints=1024, use_normals=True, dataset_name='custom'):
        self.data_path = data_path
        self.npoints = npoints
        self.use_normals = use_normals
        self.split = split
        
        # Load class names
        cat_file = os.path.join(data_path, f'{dataset_name}_shape_names.txt')
        self.classes = [line.strip() for line in open(cat_file)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Load cache file
        cache_file = os.path.join(data_path, f'{dataset_name}_{split}_8192pts_fps.dat')
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.points_list, self.labels_list = pickle.load(f)
        else:
            # Load from txt files
            split_file = os.path.join(data_path, f'{dataset_name}_{split}.txt')
            sample_ids = [line.strip() for line in open(split_file)]
            
            self.points_list = []
            self.labels_list = []
            
            for sample_id in sample_ids:
                parts = sample_id.rsplit('_', 1)
                class_name = parts[0]
                txt_path = os.path.join(data_path, class_name, f'{sample_id}.txt')
                
                points = np.loadtxt(txt_path, delimiter=',').astype(np.float32)
                label = np.array([self.class_to_idx[class_name]], dtype=np.int32)
                
                self.points_list.append(points)
                self.labels_list.append(label)
        
        print(f"Loaded {len(self.points_list)} samples for {split}")
    
    def __len__(self):
        return len(self.points_list)
    
    def __getitem__(self, idx):
        points = self.points_list[idx].copy()
        label = self.labels_list[idx][0]
        
        # Subsample to npoints
        if len(points) > self.npoints:
            points = farthest_point_sample_numpy(points, self.npoints)
        elif len(points) < self.npoints:
            indices = np.random.choice(len(points), self.npoints, replace=True)
            points = points[indices]
        
        # Normalize
        points[:, :3] = pc_normalize(points[:, :3])
        
        # Use normals or not
        if not self.use_normals:
            points = points[:, :3]
        
        # Shuffle for training
        if self.split == 'train':
            np.random.shuffle(points)
        
        return torch.from_numpy(points).float(), label
    
    def get_class_names(self):
        return self.classes


# ============================================================================
# Simple PointNet-style Model (no CUDA dependencies)
# ============================================================================

class SimplePointEncoder(nn.Module):
    """Simple point cloud encoder using 1D convolutions."""
    
    def __init__(self, input_channel=6, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        # x: (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, hidden_dim)
        return x


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block."""
    
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SimplePointTransformer(nn.Module):
    """Simplified Point Transformer for CPU training."""
    
    def __init__(self, input_channel=6, num_classes=22, hidden_dim=128, 
                 num_groups=32, group_size=32, depth=2, num_heads=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        
        # Point encoder
        self.encoder = SimplePointEncoder(input_channel, hidden_dim)
        
        # Simple grouping: just split points into groups
        self.group_encoder = nn.Sequential(
            nn.Conv1d(input_channel, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        """
        x: (B, N, C) point cloud
        """
        B, N, C = x.shape
        
        # Global features
        global_feat = self.encoder(x)  # (B, hidden_dim)
        
        # Group features: split into groups and encode
        # Reshape to groups
        num_groups = min(self.num_groups, N // 4)
        group_size = N // num_groups
        
        x_groups = x[:, :num_groups * group_size, :].reshape(B, num_groups, group_size, C)
        x_groups = x_groups.reshape(B * num_groups, group_size, C)
        
        # Encode groups
        group_feat = self.group_encoder(x_groups.transpose(1, 2))  # (B*G, hidden, group_size)
        group_feat = torch.max(group_feat, dim=2)[0]  # (B*G, hidden)
        group_feat = group_feat.reshape(B, num_groups, self.hidden_dim)  # (B, G, hidden)
        
        # Transformer
        for block in self.blocks:
            group_feat = block(group_feat)
        
        group_feat = self.norm(group_feat)
        
        # Aggregate: concat global max and mean
        feat_max = torch.max(group_feat, dim=1)[0]  # (B, hidden)
        feat_mean = torch.mean(group_feat, dim=1)   # (B, hidden)
        feat = torch.cat([feat_max, global_feat], dim=1)  # (B, hidden*2)
        
        # Classification
        logits = self.cls_head(feat)
        return logits
    
    def get_loss_acc(self, logits, labels):
        loss = self.loss_fn(logits, labels.long())
        pred = logits.argmax(-1)
        acc = (pred == labels).float().mean() * 100
        return loss, acc


# ============================================================================
# Training Functions
# ============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for idx, (points, labels) in enumerate(dataloader):
        points = points.to(device)
        labels = labels.to(device)
        
        logits = model(points)
        loss, acc = model.get_loss_acc(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), points.size(0))
        acc_meter.update(acc.item(), points.size(0))
        
        if idx % 5 == 0:
            print(f"  [Epoch {epoch}][Batch {idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}, Acc: {acc.item():.2f}%")
    
    return loss_meter.avg, acc_meter.avg


def validate(model, dataloader, device, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)
            labels = labels.to(device)
            
            logits = model(points)
            preds = logits.argmax(-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    acc = (all_preds == all_labels).float().mean() * 100
    print(f"[Validation] Epoch {epoch}: Accuracy = {acc:.2f}%")
    
    return acc.item()


def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
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
    parser = argparse.ArgumentParser(description='Simple CPU Training')
    parser.add_argument('--data_path', type=str, 
                        default='data/custom_processed',
                        help='Path to processed data')
    parser.add_argument('--exp_name', type=str, default='cpu_test',
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--npoints', type=int, default=1024,
                        help='Number of points per sample')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=2,
                        help='Transformer depth')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Output directory
    output_dir = os.path.join('experiments', 'cpu_training', args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(output_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    print("=" * 60)
    print("Point-MAE Simple CPU Training (From Scratch)")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Experiment: {args.exp_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Points: {args.npoints}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SimplePointDataset(
        args.data_path, 'train', args.npoints, use_normals=True
    )
    val_dataset = SimplePointDataset(
        args.data_path, 'test', args.npoints, use_normals=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    num_classes = len(train_dataset.get_class_names())
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes ({num_classes}): {train_dataset.get_class_names()}")
    
    # Build model
    print("\nBuilding model...")
    model = SimplePointTransformer(
        input_channel=6,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_groups=32,
        group_size=32,
        depth=args.depth,
        num_heads=4
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_acc = 0.0
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_acc = validate(model, val_loader, device, epoch)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(output_dir, 'ckpt-best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'class_names': train_dataset.get_class_names(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"  -> New best model saved! (Acc: {best_acc:.2f}%)")
    
    # Save last checkpoint
    checkpoint_path = os.path.join(output_dir, 'ckpt-last.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'num_classes': num_classes,
        'class_names': train_dataset.get_class_names(),
        'args': vars(args)
    }, checkpoint_path)
    print(f"\nLast checkpoint saved to: {checkpoint_path}")
    
    # Save history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Plot curves
    plot_path = plot_training_curves(history, output_dir)
    
    # Save log
    log_content = {
        'args': vars(args),
        'num_classes': num_classes,
        'class_names': train_dataset.get_class_names(),
        'best_acc': best_acc,
        'total_time': sum(history['epoch_time']),
        'history': history
    }
    with open(log_file, 'w') as f:
        json.dump(log_content, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Total Training Time: {sum(history['epoch_time']):.1f}s")
    print(f"Checkpoints: {output_dir}")
    print(f"Training curves: {plot_path}")
    print(f"Log file: {log_file}")
    print("=" * 60)
    
    return output_dir, best_acc


if __name__ == '__main__':
    main()
