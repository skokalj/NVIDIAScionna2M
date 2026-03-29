#!/usr/bin/env python
"""
run_conversion.py - Convert PLY meshes to Point-MAE compatible format

This script converts PLY mesh files from /data/joshi/modelnet40_meshes to the format
expected by Point-MAE (ModelNet dataset format).

Features:
- Extracts vertices and vertex normals from PLY files
- Samples points from mesh surface with normals
- Saves in ModelNet-compatible format (txt files with x,y,z,nx,ny,nz)
- Creates train/test splits
- Generates category list file
- Uses multiprocessing for 255 cores

Usage:
    python run_conversion.py --input /data/joshi/modelnet40_meshes --output /data/joshi/modelnet40_pointmae

Output structure:
    output_dir/
    ├── modelnet40_shape_names.txt
    ├── modelnet40_train.txt
    ├── modelnet40_test.txt
    ├── airplane/
    │   ├── airplane_0001.txt
    │   └── ...
    └── ...
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
from tqdm import tqdm

# Try to import trimesh for PLY loading
try:
    import trimesh
except ImportError:
    print("Installing trimesh...")
    os.system("pip install trimesh -q")
    import trimesh


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
        args: tuple of (ply_path, output_dir, n_points)
        
    Returns:
        tuple of (category, filename, success)
    """
    ply_path, output_dir, n_points = args
    
    try:
        # Parse filename: category_XXX.ply
        filename = os.path.basename(ply_path)
        name_without_ext = filename.replace('.ply', '')
        
        # Extract category (everything before the last underscore and number)
        parts = name_without_ext.rsplit('_', 1)
        if len(parts) == 2:
            category = parts[0]
            idx = parts[1]
        else:
            category = name_without_ext
            idx = "000"
        
        # Load mesh
        mesh = trimesh.load(ply_path)
        
        if not isinstance(mesh, trimesh.Trimesh):
            # Handle point clouds or scenes
            if hasattr(mesh, 'vertices') and hasattr(mesh, 'vertex_normals'):
                vertices = np.array(mesh.vertices)
                normals = np.array(mesh.vertex_normals)
                
                # If we have enough vertices, sample from them
                if len(vertices) >= n_points:
                    indices = np.random.choice(len(vertices), n_points, replace=False)
                else:
                    indices = np.random.choice(len(vertices), n_points, replace=True)
                
                points_with_normals = np.hstack([
                    vertices[indices],
                    normals[indices]
                ]).astype(np.float32)
            else:
                return (category, filename, False, "Not a valid mesh")
        else:
            # Sample points from mesh surface
            points_with_normals = sample_points_from_mesh(mesh, n_points)
        
        # Normalize XYZ coordinates
        points_with_normals[:, 0:3] = pc_normalize(points_with_normals[:, 0:3])
        
        # Create output directory for category
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Save as txt file (ModelNet format: comma-separated)
        output_filename = f"{category}_{idx.zfill(4)}.txt"
        output_path = os.path.join(category_dir, output_filename)
        
        np.savetxt(output_path, points_with_normals, delimiter=',', fmt='%.6f')
        
        return (category, output_filename.replace('.txt', ''), True, None)
        
    except Exception as e:
        return (None, ply_path, False, str(e))


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


def create_dat_cache(output_dir, n_points_cache, num_workers):
    """
    Create .dat cache files for faster loading (like original ModelNet).
    
    Args:
        output_dir: Directory with converted data
        n_points_cache: Number of points for cache (e.g., 8192)
        num_workers: Number of parallel workers
    """
    print("\nCreating .dat cache files...")
    
    # Load category list
    cat_file = os.path.join(output_dir, 'modelnet40_shape_names.txt')
    categories = [line.strip() for line in open(cat_file)]
    classes = dict(zip(categories, range(len(categories))))
    
    for split in ['train', 'test']:
        split_file = os.path.join(output_dir, f'modelnet40_{split}.txt')
        shape_ids = [line.strip() for line in open(split_file)]
        
        save_path = os.path.join(output_dir, f'modelnet40_{split}_{n_points_cache}pts_fps.dat')
        
        if os.path.exists(save_path):
            print(f"  {save_path} already exists, skipping...")
            continue
        
        print(f"  Processing {split} split ({len(shape_ids)} samples)...")
        
        list_of_points = []
        list_of_labels = []
        
        for shape_id in tqdm(shape_ids, desc=f"  {split}"):
            # Parse shape_id: category_XXXX
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
    parser.add_argument('--input', type=str, default='/data/joshi/modelnet40_meshes',
                        help='Input directory with PLY files')
    parser.add_argument('--output', type=str, default='/data/joshi/modelnet40_pointmae',
                        help='Output directory for converted data')
    parser.add_argument('--n_points', type=int, default=8192,
                        help='Number of points to sample from each mesh')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: all cores)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training samples')
    parser.add_argument('--create_cache', action='store_true', default=True,
                        help='Create .dat cache files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine number of workers
    if args.n_workers is None:
        args.n_workers = min(cpu_count(), 255)  # Use up to 255 cores
    
    print(f"=" * 60)
    print("PLY to Point-MAE Conversion")
    print(f"=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Points per mesh:  {args.n_points}")
    print(f"Workers:          {args.n_workers}")
    print(f"Train ratio:      {args.train_ratio}")
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
    
    # Process PLY files in parallel
    print(f"\nProcessing PLY files with {args.n_workers} workers...")
    
    process_args = [(ply, args.output, args.n_points) for ply in ply_files]
    
    results = []
    with Pool(args.n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_ply, process_args), 
                          total=len(ply_files), desc="Converting"):
            results.append(result)
    
    # Collect results
    successful = [(cat, name) for cat, name, success, _ in results if success]
    failed = [(name, err) for _, name, success, err in results if not success]
    
    print(f"\nSuccessfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    # Get unique categories
    categories = sorted(set(cat for cat, _ in successful))
    print(f"\nFound {len(categories)} categories:")
    for cat in categories:
        count = sum(1 for c, _ in successful if c == cat)
        print(f"  {cat}: {count} samples")
    
    # Create category list file
    cat_file = os.path.join(args.output, 'modelnet40_shape_names.txt')
    with open(cat_file, 'w') as f:
        for cat in categories:
            f.write(f"{cat}\n")
    print(f"\nSaved category list to {cat_file}")
    
    # Create train/test splits
    train_samples = []
    test_samples = []
    
    for category in categories:
        cat_samples = sorted([name for cat, name in successful if cat == category])
        n_train = int(len(cat_samples) * args.train_ratio)
        
        # Shuffle within category
        np.random.shuffle(cat_samples)
        
        train_samples.extend(cat_samples[:n_train])
        test_samples.extend(cat_samples[n_train:])
    
    # Save train/test splits
    train_file = os.path.join(args.output, 'modelnet40_train.txt')
    with open(train_file, 'w') as f:
        for name in sorted(train_samples):
            f.write(f"{name}\n")
    print(f"Saved {len(train_samples)} training samples to {train_file}")
    
    test_file = os.path.join(args.output, 'modelnet40_test.txt')
    with open(test_file, 'w') as f:
        for name in sorted(test_samples):
            f.write(f"{name}\n")
    print(f"Saved {len(test_samples)} test samples to {test_file}")
    
    # Create .dat cache files
    if args.create_cache:
        create_dat_cache(args.output, args.n_points, args.n_workers)
    
    print(f"\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output directory: {args.output}")
    print(f"=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
