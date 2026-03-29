#!/usr/bin/env python
"""
classify_single.py - Classify a single PLY file using a trained Point-MAE model

This script loads a finetuned Point-MAE model and classifies a single PLY file
or a directory of PLY files.

Usage:
    # Single file
    python tools/classify_single.py \
        --config cfgs/finetune_custom.yaml \
        --ckpts experiments/finetune_custom/cfgs/ckpt-best.pth \
        --input /path/to/mesh.ply
    
    # Directory of files
    python tools/classify_single.py \
        --config cfgs/finetune_custom.yaml \
        --ckpts experiments/finetune_custom/cfgs/ckpt-best.pth \
        --input /path/to/meshes/ \
        --output predictions.csv
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import cfg_from_yaml_file, EasyConfig
from models import build_model_from_cfg
from utils.logger import print_log

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


def load_ply_as_points(ply_path, n_points=8192, use_normals=True):
    """
    Load a PLY file and convert to point cloud with normals.
    
    Args:
        ply_path: Path to PLY file
        n_points: Number of points to sample
        use_normals: Whether to include normals (6 channels) or just xyz (3 channels)
        
    Returns:
        points: (n_points, 6) or (n_points, 3) numpy array
    """
    mesh = trimesh.load(ply_path)
    
    if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
        # Sample points from mesh surface
        points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
        normals = mesh.face_normals[face_indices]
    elif hasattr(mesh, 'vertices'):
        vertices = np.array(mesh.vertices)
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normals = np.array(mesh.vertex_normals)
        else:
            # Estimate normals
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
    
    # Normalize xyz
    points = pc_normalize(points)
    
    if use_normals:
        return np.hstack([points, normals]).astype(np.float32)
    else:
        return points.astype(np.float32)


def load_class_names(config):
    """Load class names from dataset config."""
    data_path = config.dataset.train._base_.DATA_PATH
    dataset_name = getattr(config.dataset.train._base_, 'DATASET_NAME', 'custom')
    
    cat_file = os.path.join(data_path, f'{dataset_name}_shape_names.txt')
    if os.path.exists(cat_file):
        return [line.strip() for line in open(cat_file)]
    else:
        # Return generic class names
        num_classes = config.model.cls_dim
        return [f'class_{i}' for i in range(num_classes)]


def classify(model, points, device):
    """
    Classify a point cloud.
    
    Args:
        model: Trained model
        points: (N, C) numpy array
        device: torch device
        
    Returns:
        pred_class: Predicted class index
        confidence: Confidence score (softmax probability)
        logits: Raw logits
    """
    model.eval()
    
    # Add batch dimension
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(points_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = probs.max(dim=-1)
    
    return pred_class.item(), confidence.item(), logits.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description='Classify PLY files with Point-MAE')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., cfgs/finetune_custom.yaml)')
    parser.add_argument('--ckpts', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to PLY file or directory of PLY files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for predictions (optional)')
    parser.add_argument('--n_points', type=int, default=8192,
                        help='Number of points to sample')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Show top-k predictions')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
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
    elif 'model' in state_dict:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    else:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(base_ckpt, strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpts}")
    
    # Load class names
    class_names = load_class_names(config)
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
            # Load and preprocess
            points = load_ply_as_points(ply_path, args.n_points, use_normals)
            
            # Classify
            pred_class, confidence, logits = classify(model, points, device)
            pred_name = class_names[pred_class]
            
            # Get top-k predictions
            top_k_indices = np.argsort(logits)[::-1][:args.top_k]
            top_k_probs = np.exp(logits[top_k_indices]) / np.exp(logits).sum()
            top_k_names = [class_names[i] for i in top_k_indices]
            
            results.append({
                'file': os.path.basename(ply_path),
                'prediction': pred_name,
                'confidence': confidence,
                'top_k': list(zip(top_k_names, top_k_probs.tolist()))
            })
            
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
                'top_k': []
            })
    
    # Save results to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'prediction', 'confidence'])
            for r in results:
                writer.writerow([r['file'], r['prediction'], f"{r['confidence']:.4f}"])
        print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*60}")
    print(f"Classification complete: {len(results)} files processed")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
