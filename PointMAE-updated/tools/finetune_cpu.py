#!/usr/bin/env python
"""
finetune_cpu.py - CPU-compatible finetuning script for Point-MAE

This script finetunes a pretrained Point-MAE model on CPU by:
1. Loading pretrained encoder weights from checkpoint
2. Building a PointTransformer-like model without CUDA dependencies
3. Training the classification head on custom data

Usage:
    python tools/finetune_cpu.py \
        --pretrained /path/to/ckpt-last.pth \
        --data_path data/custom_processed \
        --exp_name finetune_cpu \
        --epochs 10
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
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from timm.models.layers import DropPath, trunc_normal_


# ============================================================================
# Dataset (same as train_cpu_simple.py)
# ============================================================================

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def farthest_point_sample_numpy(points, npoint):
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
    def __init__(self, data_path, split='train', npoints=1024, use_normals=True, dataset_name='custom'):
        self.data_path = data_path
        self.npoints = npoints
        self.use_normals = use_normals
        self.split = split
        
        cat_file = os.path.join(data_path, f'{dataset_name}_shape_names.txt')
        self.classes = [line.strip() for line in open(cat_file)]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        cache_file = os.path.join(data_path, f'{dataset_name}_{split}_8192pts_fps.dat')
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.points_list, self.labels_list = pickle.load(f)
        else:
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
        
        if len(points) > self.npoints:
            points = farthest_point_sample_numpy(points, self.npoints)
        elif len(points) < self.npoints:
            indices = np.random.choice(len(points), self.npoints, replace=True)
            points = points[indices]
        
        points[:, :3] = pc_normalize(points[:, :3])
        
        if not self.use_normals:
            points = points[:, :3]
        
        if self.split == 'train':
            np.random.shuffle(points)
        
        return torch.from_numpy(points).float(), label
    
    def get_class_names(self):
        return self.classes


# ============================================================================
# CPU-Compatible PointTransformer Model
# ============================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        if isinstance(drop_path_rate, list):
            dpr = drop_path_rate
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_channel, input_channel=6):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.input_channel = input_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
        point_groups : B G N C
        '''
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


def fps_numpy(xyz, npoint):
    """Farthest Point Sampling in numpy."""
    N = xyz.shape[0]
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
    
    return centroids


def knn_numpy(x, k):
    """K-Nearest Neighbors in numpy."""
    # x: (B, N, 3)
    B, N, _ = x.shape
    
    # Compute pairwise distances
    inner = -2 * np.matmul(x, x.transpose(0, 2, 1))
    xx = np.sum(x ** 2, axis=2, keepdims=True)
    pairwise_distance = -xx - inner - xx.transpose(0, 2, 1)
    
    # Get k nearest neighbors
    idx = np.argsort(pairwise_distance, axis=-1)[:, :, :k]
    return idx


class GroupCPU(nn.Module):
    """CPU-compatible grouping module."""
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, features=None):
        '''
        xyz: B N C (C >= 3, first 3 are xyz)
        '''
        B, N, C = xyz.shape
        
        # Use numpy for FPS and KNN (CPU-compatible)
        xyz_np = xyz[:, :, :3].detach().cpu().numpy()
        
        batch_centers = []
        batch_neighborhoods = []
        
        for b in range(B):
            # FPS to get center indices
            center_idx = fps_numpy(xyz_np[b], self.num_group)
            centers = xyz[b, center_idx, :3]  # (G, 3)
            
            # KNN to get neighborhood
            # Compute distances from centers to all points
            centers_np = xyz_np[b, center_idx]  # (G, 3)
            
            # (G, N) distances
            dists = np.sum((centers_np[:, None, :] - xyz_np[b][None, :, :]) ** 2, axis=2)
            
            # Get k nearest for each center
            knn_idx = np.argsort(dists, axis=1)[:, :self.group_size]  # (G, K)
            
            # Gather neighborhoods
            neighborhood = xyz[b, knn_idx.flatten(), :].reshape(self.num_group, self.group_size, C)
            
            # Normalize to local coordinates
            neighborhood[:, :, :3] = neighborhood[:, :, :3] - centers.unsqueeze(1)
            
            batch_centers.append(centers)
            batch_neighborhoods.append(neighborhood)
        
        centers = torch.stack(batch_centers, dim=0)  # (B, G, 3)
        neighborhoods = torch.stack(batch_neighborhoods, dim=0)  # (B, G, K, C)
        
        return neighborhoods, centers


class PointTransformerCPU(nn.Module):
    """CPU-compatible PointTransformer for finetuning."""
    
    def __init__(self, config):
        super().__init__()
        
        self.trans_dim = config.get('trans_dim', 384)
        self.depth = config.get('depth', 12)
        self.drop_path_rate = config.get('drop_path_rate', 0.1)
        self.cls_dim = config.get('cls_dim', 22)
        self.num_heads = config.get('num_heads', 6)
        self.group_size = config.get('group_size', 32)
        self.num_group = config.get('num_group', 64)
        self.encoder_dims = config.get('encoder_dims', 384)
        self.input_channel = config.get('input_channel', 6)

        self.group_divider = GroupCPU(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, input_channel=self.input_channel)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.loss_ce = nn.CrossEntropyLoss()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_pretrained(self, ckpt_path):
        """Load pretrained weights from Point-MAE checkpoint."""
        print(f"Loading pretrained weights from: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if 'base_model' in ckpt:
            base_ckpt = ckpt['base_model']
        elif 'model' in ckpt:
            base_ckpt = ckpt['model']
        else:
            base_ckpt = ckpt
        
        # Clean up keys
        new_ckpt = {}
        for k, v in base_ckpt.items():
            k = k.replace("module.", "")
            
            # Map MAE_encoder keys to our encoder
            if k.startswith('MAE_encoder.'):
                new_k = k.replace('MAE_encoder.', '')
                new_ckpt[new_k] = v
            elif k.startswith('group_divider.'):
                new_ckpt[k] = v
            elif k.startswith('encoder.'):
                new_ckpt[k] = v
            elif k.startswith('pos_embed.'):
                new_ckpt[k] = v
            elif k.startswith('blocks.'):
                new_ckpt[k] = v
            elif k.startswith('norm.'):
                new_ckpt[k] = v
            elif k.startswith('cls_token'):
                new_ckpt[k] = v
            elif k.startswith('cls_pos'):
                new_ckpt[k] = v
        
        # Load with strict=False to allow missing classification head
        incompatible = self.load_state_dict(new_ckpt, strict=False)
        
        print(f"Loaded {len(new_ckpt)} pretrained parameters")
        if incompatible.missing_keys:
            print(f"Missing keys (expected for classification head): {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys[:5]}...")
        
        return self

    def forward(self, pts):
        # pts: (B, N, C)
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        x = self.blocks(x, pos)
        x = self.norm(x)
        
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        
        return ret


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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
    parser = argparse.ArgumentParser(description='CPU Finetuning for Point-MAE')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--data_path', type=str, default='data/custom_processed',
                        help='Path to processed data')
    parser.add_argument('--exp_name', type=str, default='finetune_cpu',
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--npoints', type=int, default=1024,
                        help='Number of points per sample')
    parser.add_argument('--num_group', type=int, default=32,
                        help='Number of groups')
    parser.add_argument('--group_size', type=int, default=32,
                        help='Group size')
    parser.add_argument('--trans_dim', type=int, default=384,
                        help='Transformer dimension')
    parser.add_argument('--depth', type=int, default=12,
                        help='Transformer depth')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = os.path.join('experiments', 'finetune_cpu', args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Point-MAE CPU Finetuning")
    print("=" * 60)
    print(f"Pretrained: {args.pretrained}")
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
    train_dataset = SimplePointDataset(args.data_path, 'train', args.npoints, use_normals=True)
    val_dataset = SimplePointDataset(args.data_path, 'test', args.npoints, use_normals=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(train_dataset.get_class_names())
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes ({num_classes}): {train_dataset.get_class_names()}")
    
    # Build model
    print("\nBuilding model...")
    config = {
        'trans_dim': args.trans_dim,
        'depth': args.depth,
        'drop_path_rate': 0.1,
        'cls_dim': num_classes,
        'num_heads': 6,
        'group_size': args.group_size,
        'num_group': args.num_group,
        'encoder_dims': args.trans_dim,
        'input_channel': 6
    }
    
    model = PointTransformerCPU(config)
    
    # Load pretrained weights
    if os.path.exists(args.pretrained):
        model.load_pretrained(args.pretrained)
    else:
        print(f"Warning: Pretrained checkpoint not found: {args.pretrained}")
        print("Training from scratch...")
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epoch_time': [],
        'lr': []
    }
    
    best_acc = 0.0
    
    print("\n" + "=" * 60)
    print("Starting finetuning...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        print(f"\n[Epoch {epoch}/{args.epochs}] LR: {scheduler.get_last_lr()[0]:.6f}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_acc = validate(model, val_loader, device, epoch)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
              f"Time: {epoch_time:.1f}s")
        
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
                'config': config,
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
        'config': config,
        'args': vars(args)
    }, checkpoint_path)
    print(f"\nLast checkpoint saved to: {checkpoint_path}")
    
    # Save history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_path = plot_training_curves(history, output_dir)
    
    print("\n" + "=" * 60)
    print("Finetuning Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Total Training Time: {sum(history['epoch_time']):.1f}s")
    print(f"Checkpoints: {output_dir}")
    print(f"Training curves: {plot_path}")
    print("=" * 60)
    
    return output_dir, best_acc


if __name__ == '__main__':
    main()
