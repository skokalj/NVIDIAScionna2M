"""
Extract embeddings from a pretrained Point-MAE model with 6-channel input.

This script extracts embeddings from various layers of the pretrained model:
1. Encoder output (local patch features)
2. Transformer encoder output (contextualized features)
3. Pooled global features (for downstream tasks)

Usage:
    python tools/extract_embeddings.py \
        --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
        --config cfgs/pretrain_modelnet_normals.yaml \
        --data_path /data/joshi/modelnet40_pointmae \
        --output_dir embeddings/ \
        --split test
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Point_MAE import Point_MAE, Group, Encoder, MaskTransformer
from models.build import MODELS
from utils.config import cfg_from_yaml_file
from easydict import EasyDict


class EmbeddingExtractor(nn.Module):
    """
    Wrapper around Point_MAE to extract intermediate embeddings.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.group_divider = model.group_divider
        self.MAE_encoder = model.MAE_encoder
        
    def extract_encoder_features(self, pts):
        """
        Extract features after the encoder (before transformer).
        
        Args:
            pts: (B, N, 6) input point cloud with normals
            
        Returns:
            encoder_features: (B, G, C) features for each group
            center: (B, G, 3) group centers
        """
        neighborhood, center = self.group_divider(pts)
        # Get encoder features (before transformer)
        encoder_features = self.MAE_encoder.encoder(neighborhood)
        return encoder_features, center
    
    def extract_transformer_features(self, pts, return_all_layers=False):
        """
        Extract features after the transformer encoder.
        
        Args:
            pts: (B, N, 6) input point cloud with normals
            return_all_layers: if True, return features from all transformer layers
            
        Returns:
            transformer_features: (B, G, C) or list of (B, G, C) if return_all_layers
            center: (B, G, 3) group centers
            mask: (B, G) boolean mask indicating which groups were masked
        """
        neighborhood, center = self.group_divider(pts)
        
        # Get features through the full MAE encoder (including transformer)
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        
        return x_vis, center, mask
    
    def extract_global_features(self, pts, pooling='max'):
        """
        Extract global features by pooling over all tokens.
        
        Args:
            pts: (B, N, 6) input point cloud with normals
            pooling: 'max' or 'mean'
            
        Returns:
            global_features: (B, C) global feature vector
        """
        x_vis, center, mask = self.extract_transformer_features(pts)
        
        if pooling == 'max':
            global_features = x_vis.max(dim=1)[0]
        elif pooling == 'mean':
            global_features = x_vis.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
            
        return global_features
    
    def extract_all_features(self, pts):
        """
        Extract all types of features at once.
        
        Args:
            pts: (B, N, 6) input point cloud with normals
            
        Returns:
            dict with:
                - encoder_features: (B, G, C) local patch features
                - transformer_features: (B, G_vis, C) contextualized features
                - global_max: (B, C) max-pooled global features
                - global_mean: (B, C) mean-pooled global features
                - center: (B, G, 3) group centers
                - mask: (B, G) mask indicating which groups were masked
        """
        neighborhood, center = self.group_divider(pts)
        
        # Encoder features (before transformer)
        encoder_features = self.MAE_encoder.encoder(neighborhood)
        
        # Transformer features
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        
        # Global features
        global_max = x_vis.max(dim=1)[0]
        global_mean = x_vis.mean(dim=1)
        
        return {
            'encoder_features': encoder_features,
            'transformer_features': x_vis,
            'global_max': global_max,
            'global_mean': global_mean,
            'center': center,
            'mask': mask
        }


def load_model(ckpt_path, config_path, device='cuda'):
    """Load pretrained model from checkpoint."""
    # Load config
    config = cfg_from_yaml_file(config_path)
    
    # Build model
    model = MODELS.build(config.model).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, config


def load_data(data_path, split='test', npoints=8192):
    """Load point cloud data from txt files or dat cache."""
    dat_file = os.path.join(data_path, f'modelnet40_{split}_{npoints}pts_fps.dat')
    txt_file = os.path.join(data_path, f'modelnet40_{split}.txt')
    
    # Try loading from dat cache first
    if os.path.exists(dat_file):
        print(f"Loading from {dat_file}")
        with open(dat_file, 'rb') as f:
            data = pickle.load(f)
        # Check if cache has enough valid samples
        valid_count = sum(1 for d in data if isinstance(d, (tuple, list)) and len(d) >= 2 
                        and hasattr(d[0], 'shape') and len(d[0].shape) == 2 and d[0].shape[1] >= 6)
        if valid_count > 10:  # Cache seems valid
            return data
        print(f"  Cache has only {valid_count} valid samples, loading from txt files...")
    
    # Load from txt file listing
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Neither dat cache nor txt file found: {dat_file}, {txt_file}")
    
    print(f"Loading from txt file: {txt_file}")
    
    # Read class names
    shape_names_file = os.path.join(data_path, 'modelnet40_shape_names.txt')
    with open(shape_names_file, 'r') as f:
        shape_names = [line.strip() for line in f.readlines()]
    class_to_idx = {name: i for i, name in enumerate(shape_names)}
    
    # Read sample list
    with open(txt_file, 'r') as f:
        sample_names = [line.strip() for line in f.readlines()]
    
    print(f"  Found {len(sample_names)} samples in {split} split")
    
    data = []
    for sample_name in tqdm(sample_names, desc=f"Loading {split} data"):
        # Parse class name from sample name (e.g., "airplane_0001" -> "airplane")
        class_name = '_'.join(sample_name.split('_')[:-1])
        class_idx = class_to_idx.get(class_name, -1)
        
        # Try to find the point cloud file
        txt_path = os.path.join(data_path, class_name, f'{sample_name}.txt')
        
        if os.path.exists(txt_path):
            # Load point cloud from txt file
            points = np.loadtxt(txt_path, delimiter=',')
            
            # Ensure we have 6 channels (xyz + normals)
            if points.shape[1] < 6:
                # Pad with zeros for normals if missing
                normals = np.zeros((points.shape[0], 6 - points.shape[1]))
                points = np.hstack([points, normals])
            
            # Subsample to npoints if needed
            if points.shape[0] > npoints:
                idx = np.random.choice(points.shape[0], npoints, replace=False)
                points = points[idx]
            elif points.shape[0] < npoints:
                # Repeat points to reach npoints
                repeat_times = (npoints // points.shape[0]) + 1
                points = np.tile(points, (repeat_times, 1))[:npoints]
            
            data.append((points.astype(np.float32), class_idx))
        else:
            print(f"  Warning: File not found: {txt_path}")
    
    print(f"  Loaded {len(data)} samples")
    return data


def extract_embeddings(model, data, device='cuda', batch_size=16, npoints=8192):
    """
    Extract embeddings for all samples in the dataset.
    
    Args:
        model: EmbeddingExtractor wrapper
        data: list of (points, label) tuples
        device: cuda or cpu
        batch_size: batch size for processing
        npoints: expected number of points per sample
        
    Returns:
        dict with embeddings and labels
    """
    model.eval()
    
    # Filter valid samples
    valid_data = []
    for i, sample in enumerate(data):
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            points = sample[0]
            if hasattr(points, 'shape') and len(points.shape) == 2 and points.shape[1] >= 6:
                valid_data.append(sample)
            else:
                print(f"  Skipping sample {i}: invalid shape {getattr(points, 'shape', type(points))}")
        else:
            print(f"  Skipping sample {i}: invalid format")
    
    print(f"  Valid samples: {len(valid_data)}/{len(data)}")
    
    if len(valid_data) == 0:
        raise ValueError("No valid samples found in data!")
    
    all_encoder_features = []
    all_transformer_features = []
    all_global_max = []
    all_global_mean = []
    all_labels = []
    all_centers = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_data), batch_size), desc="Extracting embeddings"):
            batch_data = valid_data[i:i+batch_size]
            
            # Stack batch - handle variable point counts
            batch_points = []
            labels = []
            for d in batch_data:
                pts = torch.from_numpy(d[0]).float()
                # Ensure we have exactly npoints
                if pts.shape[0] > npoints:
                    # Random subsample
                    idx = torch.randperm(pts.shape[0])[:npoints]
                    pts = pts[idx]
                elif pts.shape[0] < npoints:
                    # Pad by repeating
                    repeat_times = (npoints // pts.shape[0]) + 1
                    pts = pts.repeat(repeat_times, 1)[:npoints]
                batch_points.append(pts)
                labels.append(d[1])
            
            points = torch.stack(batch_points).to(device)
            
            # Extract features
            features = model.extract_all_features(points)
            
            all_encoder_features.append(features['encoder_features'].cpu().numpy())
            all_transformer_features.append(features['transformer_features'].cpu().numpy())
            all_global_max.append(features['global_max'].cpu().numpy())
            all_global_mean.append(features['global_mean'].cpu().numpy())
            all_centers.append(features['center'].cpu().numpy())
            all_labels.extend(labels)
    
    return {
        'encoder_features': np.concatenate(all_encoder_features, axis=0),
        'transformer_features': np.concatenate(all_transformer_features, axis=0),
        'global_max': np.concatenate(all_global_max, axis=0),
        'global_mean': np.concatenate(all_global_mean, axis=0),
        'centers': np.concatenate(all_centers, axis=0),
        'labels': np.array(all_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from Point-MAE')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='embeddings/', help='Output directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Data split')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.ckpt}")
    model, config = load_model(args.ckpt, args.config, args.device)
    extractor = EmbeddingExtractor(model).to(args.device)
    
    # Load data
    print(f"Loading {args.split} data from {args.data_path}")
    data = load_data(args.data_path, args.split)
    print(f"Loaded {len(data)} samples")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = extract_embeddings(extractor, data, args.device, args.batch_size)
    
    # Save embeddings
    output_file = os.path.join(args.output_dir, f'embeddings_{args.split}.npz')
    print(f"Saving embeddings to {output_file}")
    np.savez(output_file, **embeddings)
    
    # Print summary
    print("\nEmbedding shapes:")
    print(f"  encoder_features: {embeddings['encoder_features'].shape}")
    print(f"  transformer_features: {embeddings['transformer_features'].shape}")
    print(f"  global_max: {embeddings['global_max'].shape}")
    print(f"  global_mean: {embeddings['global_mean'].shape}")
    print(f"  labels: {embeddings['labels'].shape}")
    
    print(f"\nDone! Embeddings saved to {output_file}")


if __name__ == '__main__':
    main()
