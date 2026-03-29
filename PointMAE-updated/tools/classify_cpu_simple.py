#!/usr/bin/env python
"""
classify_cpu_simple.py - Simple CPU-only classification script

This script classifies PLY files using the trained SimplePointTransformer model.
No CUDA dependencies required.

Usage:
    python tools/classify_cpu_simple.py \
        --ckpts experiments/cpu_training/cpu_test/ckpt-best.pth \
        --input /path/to/mesh.ply
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

try:
    import trimesh
except ImportError:
    print("Installing trimesh...")
    os.system("pip install trimesh -q")
    import trimesh


# ============================================================================
# Model Definition (same as training)
# ============================================================================

class SimplePointEncoder(nn.Module):
    def __init__(self, input_channel=6, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]
        return x


class SimpleTransformerBlock(nn.Module):
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
    def __init__(self, input_channel=6, num_classes=22, hidden_dim=128, 
                 num_groups=32, group_size=32, depth=2, num_heads=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        
        self.encoder = SimplePointEncoder(input_channel, hidden_dim)
        
        self.group_encoder = nn.Sequential(
            nn.Conv1d(input_channel, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(hidden_dim, num_heads)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        B, N, C = x.shape
        
        global_feat = self.encoder(x)
        
        num_groups = min(self.num_groups, N // 4)
        group_size = N // num_groups
        
        x_groups = x[:, :num_groups * group_size, :].reshape(B, num_groups, group_size, C)
        x_groups = x_groups.reshape(B * num_groups, group_size, C)
        
        group_feat = self.group_encoder(x_groups.transpose(1, 2))
        group_feat = torch.max(group_feat, dim=2)[0]
        group_feat = group_feat.reshape(B, num_groups, self.hidden_dim)
        
        for block in self.blocks:
            group_feat = block(group_feat)
        
        group_feat = self.norm(group_feat)
        
        feat_max = torch.max(group_feat, dim=1)[0]
        feat = torch.cat([feat_max, global_feat], dim=1)
        
        logits = self.cls_head(feat)
        return logits


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


def load_ply_as_points(ply_path, n_points=512, use_normals=True):
    """Load PLY file and convert to point cloud."""
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
    parser = argparse.ArgumentParser(description='Simple CPU Classification')
    parser.add_argument('--ckpts', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to PLY file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--n_points', type=int, default=512,
                        help='Number of points')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k predictions')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpts}")
    checkpoint = torch.load(args.ckpts, map_location='cpu')
    
    class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(22)])
    num_classes = checkpoint.get('num_classes', len(class_names))
    saved_args = checkpoint.get('args', {})
    
    print(f"Classes ({num_classes}): {class_names}")
    
    # Build model
    model = SimplePointTransformer(
        input_channel=6,
        num_classes=num_classes,
        hidden_dim=saved_args.get('hidden_dim', 128),
        num_groups=32,
        group_size=32,
        depth=saved_args.get('depth', 2),
        num_heads=4
    )
    
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
            
            # Top-k
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
    
    # Save results
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
