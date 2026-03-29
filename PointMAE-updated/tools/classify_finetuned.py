#!/usr/bin/env python
"""
classify_finetuned.py - Classify PLY files using finetuned PointTransformer

Usage:
    python tools/classify_finetuned.py \
        --ckpts experiments/finetune_cpu/finetune_pretrained/ckpt-best.pth \
        --input /path/to/meshes \
        --output predictions.json
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

try:
    import trimesh
except ImportError:
    os.system("pip install trimesh -q")
    import trimesh


# ============================================================================
# Model Definition (same as finetune_cpu.py)
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
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


def fps_numpy(xyz, npoint):
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


class GroupCPU(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, features=None):
        B, N, C = xyz.shape
        xyz_np = xyz[:, :, :3].detach().cpu().numpy()
        
        batch_centers = []
        batch_neighborhoods = []
        
        for b in range(B):
            center_idx = fps_numpy(xyz_np[b], self.num_group)
            centers = xyz[b, center_idx, :3]
            centers_np = xyz_np[b, center_idx]
            dists = np.sum((centers_np[:, None, :] - xyz_np[b][None, :, :]) ** 2, axis=2)
            knn_idx = np.argsort(dists, axis=1)[:, :self.group_size]
            neighborhood = xyz[b, knn_idx.flatten(), :].reshape(self.num_group, self.group_size, C)
            neighborhood[:, :, :3] = neighborhood[:, :, :3] - centers.unsqueeze(1)
            batch_centers.append(centers)
            batch_neighborhoods.append(neighborhood)
        
        centers = torch.stack(batch_centers, dim=0)
        neighborhoods = torch.stack(batch_neighborhoods, dim=0)
        return neighborhoods, centers


class PointTransformerCPU(nn.Module):
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
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def forward(self, pts):
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
# Utility Functions
# ============================================================================

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def load_ply_as_points(ply_path, n_points=1024, use_normals=True):
    mesh = trimesh.load(ply_path)
    
    if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
        points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
        normals = mesh.face_normals[face_indices]
    elif hasattr(mesh, 'vertices'):
        vertices = np.array(mesh.vertices)
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normals = np.array(mesh.vertex_normals)
        else:
            centroid = vertices.mean(axis=0)
            normals = vertices - centroid
            normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        if len(vertices) >= n_points:
            indices = np.random.choice(len(vertices), n_points, replace=False)
        else:
            indices = np.random.choice(len(vertices), n_points, replace=True)
        
        points = vertices[indices]
        normals = normals[indices]
    else:
        raise ValueError(f"Cannot load {ply_path}")
    
    points = pc_normalize(points)
    
    if use_normals:
        return np.hstack([points, normals]).astype(np.float32)
    else:
        return points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Classify with Finetuned Model')
    parser.add_argument('--ckpts', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to PLY file or directory')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--n_points', type=int, default=1024, help='Number of points')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k predictions')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpts}")
    checkpoint = torch.load(args.ckpts, map_location='cpu')
    
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(22)])
    config = checkpoint.get('config', {
        'trans_dim': 384, 'depth': 12, 'drop_path_rate': 0.1,
        'cls_dim': len(class_names), 'num_heads': 6,
        'group_size': 32, 'num_group': 32, 'encoder_dims': 384, 'input_channel': 6
    })
    
    print(f"Classes ({len(class_names)}): {class_names}")
    
    # Build model
    model = PointTransformerCPU(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Get input files
    if os.path.isfile(args.input):
        ply_files = [args.input]
    elif os.path.isdir(args.input):
        ply_files = sorted([
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if f.endswith('.ply')
        ])
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")
    
    print(f"\nClassifying {len(ply_files)} files...")
    
    results = []
    
    for ply_path in ply_files:
        try:
            points = load_ply_as_points(ply_path, args.n_points, use_normals=True)
            points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(points_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            pred_class = np.argmax(probs)
            pred_name = class_names[pred_class]
            confidence = probs[pred_class]
            
            top_k_indices = np.argsort(probs)[::-1][:args.top_k]
            top_k_probs = probs[top_k_indices]
            top_k_names = [class_names[i] for i in top_k_indices]
            
            result = {
                'file': os.path.basename(ply_path),
                'prediction': pred_name,
                'confidence': float(confidence),
                'top_k': [{'class': n, 'prob': float(p)} for n, p in zip(top_k_names, top_k_probs)]
            }
            results.append(result)
            
            print(f"\n{os.path.basename(ply_path)}:")
            print(f"  Prediction: {pred_name} ({confidence*100:.1f}%)")
            print(f"  Top-{args.top_k}:")
            for name, prob in zip(top_k_names, top_k_probs):
                print(f"    - {name}: {prob*100:.1f}%")
                
        except Exception as e:
            print(f"\nError processing {ply_path}: {e}")
            results.append({
                'file': os.path.basename(ply_path),
                'prediction': 'ERROR',
                'error': str(e)
            })
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*60}")
    print(f"Classification complete: {len(results)} files processed")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    main()
