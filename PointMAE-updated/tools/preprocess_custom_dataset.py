#!/usr/bin/env python
"""
preprocess_custom_dataset.py - Convert custom PLY meshes to Point-MAE format

This script converts PLY mesh files to the format expected by Point-MAE for classification.
It automatically extracts class labels from filenames and creates train/test splits.

Features:
- Extracts vertices and vertex normals from PLY files
- Samples points from mesh surface with normals
- Saves in Point-MAE compatible format (txt files with x,y,z,nx,ny,nz)
- Creates train/test splits
- Generates category list file and .dat cache files
- Uses multiprocessing for faster processing

Usage:
    python tools/preprocess_custom_dataset.py \
        --input /path/to/meshes \
        --output /path/to/output \
        --n_points 8192 \
        --train_ratio 0.8

Output structure:
    output_dir/
    ├── custom_shape_names.txt      # List of class names
    ├── custom_train.txt            # Training sample list
    ├── custom_test.txt             # Test sample list
    ├── custom_train_8192pts_fps.dat  # Training cache
    ├── custom_test_8192pts_fps.dat   # Test cache
    ├── book/
    │   ├── book_0001.txt
    │   └── ...
    └── ...
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm
import re

# Try to import trimesh for PLY loading
try:
    import trimesh
except ImportError:
    print("Installing trimesh...")
    os.system("pip install trimesh -q")
    import trimesh


def extract_class_from_filename(filename):
    """
    Extract class name from filename.
    
    Examples:
        Table_NW.ply -> table
        Monitor_Cart.ply -> monitor
        Book2_NE.ply -> book
        CabinetLock_S.ply -> cabinetlock
        Handle_Cart_N.ply -> handle
    
    Args:
        filename: PLY filename
        
    Returns:
        class_name: Lowercase class name
    """
    name = os.path.basename(filename).replace('.ply', '')
    
    # Remove location suffixes like _N, _S, _NE, _NW, _SE, _SW, _C, _NC, _Cart
    # Pattern: remove _[NSEW]{1,2} or _C or _NC or _Cart at the end
    name = re.sub(r'_(?:Cart|[NSEWC]{1,2}|NC)$', '', name, flags=re.IGNORECASE)
    
    # Remove trailing numbers (e.g., Book2 -> Book, Container8 -> Container)
    name = re.sub(r'\d+$', '', name)
    
    # Remove any remaining trailing underscores
    name = name.rstrip('_')
    
    return name.lower()


def sample_points_from_mesh(mesh, n_points=8192):
    """
    Sample points uniformly from mesh surface with normals.
    
    Args:
        mesh: trimesh.Trimesh object
        n_points: Number of points to sample
        
    Returns:
        points: (n_points, 6) array with xyz and normals
    """
    # Sample points from surface
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    # Get face normals for sampled points
    normals = mesh.face_normals[face_indices]
    
    # Combine xyz and normals
    points_with_normals = np.hstack([points, normals]).astype(np.float32)
    
    return points_with_normals


def pc_normalize(pc):
    """Normalize point cloud to unit sphere centered at origin."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc


def process_single_ply(args):
    """
    Process a single PLY file.
    
    Args:
        args: tuple of (ply_path, output_dir, n_points, class_name, sample_idx)
        
    Returns:
        tuple of (class_name, sample_name, success, error_msg)
    """
    ply_path, output_dir, n_points, class_name, sample_idx = args
    
    try:
        # Load mesh
        mesh = trimesh.load(ply_path)
        
        if isinstance(mesh, trimesh.Trimesh):
            # Sample points from mesh surface
            if len(mesh.faces) == 0:
                # No faces, use vertices directly
                vertices = np.array(mesh.vertices)
                if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                    normals = np.array(mesh.vertex_normals)
                else:
                    # Estimate normals (pointing outward from centroid)
                    centroid = vertices.mean(axis=0)
                    normals = vertices - centroid
                    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
                
                if len(vertices) >= n_points:
                    indices = np.random.choice(len(vertices), n_points, replace=False)
                else:
                    indices = np.random.choice(len(vertices), n_points, replace=True)
                
                points_with_normals = np.hstack([
                    vertices[indices],
                    normals[indices]
                ]).astype(np.float32)
            else:
                points_with_normals = sample_points_from_mesh(mesh, n_points)
        elif hasattr(mesh, 'vertices'):
            # Handle point clouds or scenes
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
            
            points_with_normals = np.hstack([
                vertices[indices],
                normals[indices]
            ]).astype(np.float32)
        else:
            return (class_name, None, False, "Not a valid mesh or point cloud")
        
        # Normalize XYZ coordinates
        points_with_normals[:, 0:3] = pc_normalize(points_with_normals[:, 0:3])
        
        # Create output directory for class
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Save as txt file
        sample_name = f"{class_name}_{sample_idx:04d}"
        output_path = os.path.join(class_dir, f"{sample_name}.txt")
        
        np.savetxt(output_path, points_with_normals, delimiter=',', fmt='%.6f')
        
        return (class_name, sample_name, True, None)
        
    except Exception as e:
        return (class_name, None, False, str(e))


def farthest_point_sample(point, npoint):
    """
    Farthest Point Sampling.
    
    Args:
        point: (N, D) point cloud
        npoint: number of samples
        
    Returns:
        (npoint, D) sampled points
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    
    point = point[centroids.astype(np.int32)]
    return point


def create_dat_cache(output_dir, n_points_cache, dataset_name='custom'):
    """
    Create .dat cache files for faster loading.
    
    Args:
        output_dir: Directory with converted data
        n_points_cache: Number of points for cache
        dataset_name: Prefix for files (default: 'custom')
    """
    print("\nCreating .dat cache files...")
    
    # Load category list
    cat_file = os.path.join(output_dir, f'{dataset_name}_shape_names.txt')
    categories = [line.strip() for line in open(cat_file)]
    classes = dict(zip(categories, range(len(categories))))
    
    for split in ['train', 'test']:
        split_file = os.path.join(output_dir, f'{dataset_name}_{split}.txt')
        if not os.path.exists(split_file):
            print(f"  {split_file} not found, skipping...")
            continue
            
        shape_ids = [line.strip() for line in open(split_file)]
        
        save_path = os.path.join(output_dir, f'{dataset_name}_{split}_{n_points_cache}pts_fps.dat')
        
        if os.path.exists(save_path):
            print(f"  {save_path} already exists, skipping...")
            continue
        
        print(f"  Processing {split} split ({len(shape_ids)} samples)...")
        
        list_of_points = []
        list_of_labels = []
        
        for shape_id in tqdm(shape_ids, desc=f"  {split}"):
            # Parse shape_id: class_XXXX
            parts = shape_id.rsplit('_', 1)
            category = parts[0]
            
            txt_path = os.path.join(output_dir, category, f"{shape_id}.txt")
            
            if not os.path.exists(txt_path):
                print(f"    Warning: {txt_path} not found")
                continue
            
            # Load point cloud
            point_set = np.loadtxt(txt_path, delimiter=',').astype(np.float32)
            
            # FPS sampling
            if len(point_set) >= n_points_cache:
                point_set = farthest_point_sample(point_set, n_points_cache)
            else:
                # Repeat points if not enough
                indices = np.random.choice(len(point_set), n_points_cache, replace=True)
                point_set = point_set[indices]
            
            # Get label
            cls = classes[category]
            label = np.array([cls]).astype(np.int32)
            
            list_of_points.append(point_set)
            list_of_labels.append(label)
        
        # Save cache
        with open(save_path, 'wb') as f:
            pickle.dump([list_of_points, list_of_labels], f)
        
        print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PLY meshes to Point-MAE format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with PLY files (e.g., /path/to/meshes)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for converted data')
    parser.add_argument('--n_points', type=int, default=8192,
                        help='Number of points to sample from each mesh')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: all cores)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training samples')
    parser.add_argument('--dataset_name', type=str, default='custom',
                        help='Dataset name prefix for output files')
    parser.add_argument('--create_cache', action='store_true', default=True,
                        help='Create .dat cache files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine number of workers
    if args.n_workers is None:
        args.n_workers = min(cpu_count(), 32)
    
    print(f"=" * 60)
    print("PLY to Point-MAE Conversion for Custom Dataset")
    print(f"=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Points per mesh:  {args.n_points}")
    print(f"Workers:          {args.n_workers}")
    print(f"Train ratio:      {args.train_ratio}")
    print(f"Dataset name:     {args.dataset_name}")
    print(f"=" * 60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find all PLY files
    ply_files = sorted([
        os.path.join(args.input, f) 
        for f in os.listdir(args.input) 
        if f.endswith('.ply')
    ])
    
    print(f"\nFound {len(ply_files)} PLY files")
    
    if len(ply_files) == 0:
        print("No PLY files found!")
        return 1
    
    # Extract classes and organize files
    class_to_files = {}
    for ply_path in ply_files:
        class_name = extract_class_from_filename(ply_path)
        if class_name not in class_to_files:
            class_to_files[class_name] = []
        class_to_files[class_name].append(ply_path)
    
    print(f"\nDetected {len(class_to_files)} classes:")
    for cls, files in sorted(class_to_files.items()):
        print(f"  {cls}: {len(files)} samples")
    
    # Prepare processing arguments
    process_args = []
    for class_name, files in class_to_files.items():
        for idx, ply_path in enumerate(files, 1):
            process_args.append((ply_path, args.output, args.n_points, class_name, idx))
    
    # Process PLY files in parallel
    print(f"\nProcessing PLY files with {args.n_workers} workers...")
    
    results = []
    with Pool(args.n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_ply, process_args), 
                          total=len(process_args), desc="Converting"):
            results.append(result)
    
    # Collect results
    successful = [(cls, name) for cls, name, success, _ in results if success]
    failed = [(cls, err) for cls, _, success, err in results if not success]
    
    print(f"\nSuccessfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for cls, err in failed[:10]:
            print(f"  {cls}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Get unique categories
    categories = sorted(set(cls for cls, _ in successful))
    print(f"\nFinal classes ({len(categories)}):")
    for cat in categories:
        count = sum(1 for c, _ in successful if c == cat)
        print(f"  {cat}: {count} samples")
    
    # Create category list file
    cat_file = os.path.join(args.output, f'{args.dataset_name}_shape_names.txt')
    with open(cat_file, 'w') as f:
        for cat in categories:
            f.write(f"{cat}\n")
    print(f"\nSaved category list to {cat_file}")
    
    # Create train/test splits
    train_samples = []
    test_samples = []
    
    for category in categories:
        cat_samples = sorted([name for cls, name in successful if cls == category])
        n_train = max(1, int(len(cat_samples) * args.train_ratio))
        
        # Shuffle within category
        np.random.shuffle(cat_samples)
        
        train_samples.extend(cat_samples[:n_train])
        test_samples.extend(cat_samples[n_train:])
    
    # Save train/test splits
    train_file = os.path.join(args.output, f'{args.dataset_name}_train.txt')
    with open(train_file, 'w') as f:
        for name in sorted(train_samples):
            f.write(f"{name}\n")
    print(f"Saved {len(train_samples)} training samples to {train_file}")
    
    test_file = os.path.join(args.output, f'{args.dataset_name}_test.txt')
    with open(test_file, 'w') as f:
        for name in sorted(test_samples):
            f.write(f"{name}\n")
    print(f"Saved {len(test_samples)} test samples to {test_file}")
    
    # Create .dat cache files
    if args.create_cache:
        create_dat_cache(args.output, args.n_points, args.dataset_name)
    
    print(f"\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output directory: {args.output}")
    print(f"Number of classes: {len(categories)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"=" * 60)
    
    # Print next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Update dataset config: cfgs/dataset_configs/CustomDataset.yaml")
    print(f"   - Set DATA_PATH to: {args.output}")
    print(f"   - Set NUM_CATEGORY to: {len(categories)}")
    print(f"2. Update finetune config: cfgs/finetune_custom.yaml")
    print(f"   - Set cls_dim to: {len(categories)}")
    print(f"3. Run finetuning with pretrained weights")
    print(f"=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
