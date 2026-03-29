#!/usr/bin/env python
"""
classify_cpu.py - CPU-compatible classification script for Point-MAE

This script classifies PLY files using a trained Point-MAE model on CPU.

Usage:
    python tools/classify_cpu.py \
        --config cfgs/finetune_custom_cpu.yaml \
        --ckpts experiments/cpu_training/cpu_test/ckpt-best.pth \
        --input /path/to/mesh.ply
"""

import os
import sys
import argparse
import numpy as np
import torch
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import cfg_from_yaml_file
from models import build_model_from_cfg

try:
    import trimesh
except ImportError:
    print("Installing trimesh...")
    os.system("pip install trimesh -q")
    import trimesh


def pc_normalize(pc):
    """Normalize point cloud to unit sphere centered at origin."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def load_ply_as_points(ply_path, n_points=1024, use_normals=True):
    """Load a PLY file and convert to point cloud with normals."""
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
        raise ValueError(f"Cannot load {ply_path} as mesh or point cloud")
    
    points = pc_normalize(points)
    
    if use_normals:
        return np.hstack([points, normals]).astype(np.float32)
    else:
        return points.astype(np.float32)


def load_class_names(data_path, dataset_name='custom'):
    """Load class names from dataset."""
    cat_file = os.path.join(data_path, f'{dataset_name}_shape_names.txt')
    if os.path.exists(cat_file):
        return [line.strip() for line in open(cat_file)]
    return None


def classify(model, points, device):
    """Classify a point cloud."""
    model.eval()
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(points_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = probs.max(dim=-1)
    
    return pred_class.item(), confidence.item(), probs.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='CPU Classification for Point-MAE')
    parser.add_argument('--config', type=str, default='cfgs/finetune_custom_cpu.yaml',
                        help='Path to config file')
    parser.add_argument('--ckpts', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to PLY file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for predictions')
    parser.add_argument('--n_points', type=int, default=1024,
                        help='Number of points to sample')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Show top-k predictions')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = cfg_from_yaml_file(args.config)
    
    # Build model
    print("Loading model...")
    model = build_model_from_cfg(config.model)
    
    # Load checkpoint
    if not os.path.exists(args.ckpts):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpts}")
    
    state_dict = torch.load(args.ckpts, map_location='cpu')
    if 'base_model' in state_dict:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        base_ckpt = state_dict
    
    model.load_state_dict(base_ckpt, strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpts}")
    
    # Load class names
    data_path = config.dataset.train._base_.DATA_PATH
    dataset_name = getattr(config.dataset.train._base_, 'DATASET_NAME', 'custom')
    class_names = load_class_names(data_path, dataset_name)
    
    if class_names is None:
        class_names = [f'class_{i}' for i in range(config.model.cls_dim)]
    
    print(f"Classes ({len(class_names)}): {class_names}")
    
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
    
    # Classify each file
    results = []
    use_normals = getattr(config.model, 'input_channel', 6) == 6
    
    for ply_path in ply_files:
        try:
            points = load_ply_as_points(ply_path, args.n_points, use_normals)
            pred_class, confidence, probs = classify(model, points, device)
            pred_name = class_names[pred_class]
            
            # Get top-k predictions
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
                'confidence': 0.0,
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
