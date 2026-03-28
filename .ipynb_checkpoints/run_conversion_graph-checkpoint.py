"""
run_conversion_graph.py

Convert each PLY mesh in Graph/data/meshes/ into a Point-MAE compatible
point cloud (x, y, z, nx, ny, nz).

NO categories
NO train/test split
ONE mesh = ONE graph node
"""

import os
import numpy as np
import trimesh
from tqdm import tqdm

INPUT_MESH_DIR = "data/rooms/meshes"
OUTPUT_PC_DIR = "data/rooms/pointclouds"
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
    os.makedirs(OUTPUT_PC_DIR, exist_ok=True)

    ply_files = sorted([
        f for f in os.listdir(INPUT_MESH_DIR)
        if f.lower().endswith(".ply")
    ])

    print(f"Found {len(ply_files)} meshes")

    for ply in tqdm(ply_files, desc="Converting meshes"):
        ply_path = os.path.join(INPUT_MESH_DIR, ply)
        name = os.path.splitext(ply)[0]

        mesh = trimesh.load(ply_path, force="mesh")

        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Skipping {ply} (not a mesh)")
            continue

        pts = sample_mesh(mesh, N_POINTS)
        pts[:, :3] = pc_normalize(pts[:, :3])

        out_path = os.path.join(OUTPUT_PC_DIR, f"{name}.txt")
        np.savetxt(out_path, pts, delimiter=",", fmt="%.6f")

    print(f"\nDone. Saved point clouds to {OUTPUT_PC_DIR}")


if __name__ == "__main__":
    main()