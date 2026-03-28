#!/usr/bin/env python3
"""
Convert PLY meshes to point clouds for Point-MAE processing.
Supports multiple datasets: rooms, rooms_update, roomsmoved
"""
import os
import argparse
import numpy as np
import trimesh
from tqdm import tqdm

DATA_ROOT = "/data/hafeez"#"/data/hafeez/graphdata"
N_POINTS = 8192

def pc_normalize(pc):
    centroid = pc.mean(axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt((pc ** 2).sum(axis=1)))
    if scale > 0:
        pc /= scale
    return pc

def sample_mesh(mesh, n_points):
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_idx]
    return np.hstack([pts, normals]).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['rooms', 'rooms_update', 'roomsmoved', 'roomsMO'], 
                    default='rooms', help='Dataset to process')
    parser.add_argument('--n_points', type=int, default=N_POINTS, 
                        help='Number of points to sample per mesh')
    args = parser.parse_args()
    
    input_mesh_dir = os.path.join(DATA_ROOT, args.dataset, "meshes")
    output_pc_dir = os.path.join(DATA_ROOT, args.dataset, "pointclouds")
    
    os.makedirs(output_pc_dir, exist_ok=True)
    
    if not os.path.exists(input_mesh_dir):
        print(f"[ERROR] Mesh directory not found: {input_mesh_dir}")
        return
    
    ply_files = sorted([
        f for f in os.listdir(input_mesh_dir)
        if f.lower().endswith(".ply")
    ])
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Found {len(ply_files)} meshes in {input_mesh_dir}")
    
    for ply in tqdm(ply_files, desc="Converting meshes"):
        ply_path = os.path.join(input_mesh_dir, ply)
        name = os.path.splitext(ply)[0]
        
        try:
            mesh = trimesh.load(ply_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Skipping {ply} (not a mesh)")
                continue
            
            pts = sample_mesh(mesh, args.n_points)
            pts[:, :3] = pc_normalize(pts[:, :3])
            
            out_path = os.path.join(output_pc_dir, f"{name}.txt")
            np.savetxt(out_path, pts, delimiter=",", fmt="%.6f")
        except Exception as e:
            print(f"Error processing {ply}: {e}")
    
    print(f"\n[DONE] Saved point clouds to {output_pc_dir}")

if __name__ == "__main__":
    main()
