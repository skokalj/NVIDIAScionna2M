#!/usr/bin/env python3
"""
Extract Point-MAE features from point clouds.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

DATA_ROOT = "/data/hafeez"
POINT_MAE_ROOT = "/data/joshi/modelnet40_trained__mae/src"
CKPT_PATH = "/data/joshi/modelnet40_trained__mae/ckpt-last.pth"
CONFIG_PATH = "cfgs/pretrain.yaml"

sys.path.insert(0, POINT_MAE_ROOT)

from models.build import MODELS
from utils.config import cfg_from_yaml_file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GeometryProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(384, 128)  
    def forward(self, x):
        return self.proj(x)


def load_pointmae():
    cwd = os.getcwd()
    os.chdir(POINT_MAE_ROOT)
    
    cfg = cfg_from_yaml_file(CONFIG_PATH)
    model = MODELS.build(cfg.model).to(DEVICE)
    
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    os.chdir(cwd)
    return model

def load_pointcloud(path):
    pc = np.loadtxt(path, delimiter=",").astype(np.float32)
    pc = pc[:, :3]  # xyz only
    
    # Add dummy normals (zeros)
    normals = np.zeros_like(pc)
    pc_with_normals = np.concatenate([pc, normals], axis=1)  # [N, 6]
    return torch.from_numpy(pc_with_normals).unsqueeze(0).to(DEVICE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['rooms', 'rooms_update', 'roomsmoved', 'roomsMO'], 
                        default='rooms', help='Dataset to process')
    args = parser.parse_args()
    
    pointcloud_dir = os.path.join(DATA_ROOT, args.dataset, "pointclouds")
    output_file = os.path.join(DATA_ROOT, args.dataset, "z_pc.npy")
    
    if not os.path.exists(pointcloud_dir):
        print(f"[ERROR] Point cloud directory not found: {pointcloud_dir}")
        print("Please run run_conversion_graph.py first.")
        return
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Point-MAE source: {POINT_MAE_ROOT}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Loading Point-MAE model...")
    
    model = load_pointmae()
    projector = GeometryProjector().to(DEVICE)
    projector.eval()
    
    files = sorted(f for f in os.listdir(pointcloud_dir) if f.endswith(".txt"))
    print(f"Found {len(files)} point clouds")
    
    z_pc_list = []
    
    with torch.no_grad():
        for fname in tqdm(files, desc="Extracting features"):
            try:
                pc = load_pointcloud(os.path.join(pointcloud_dir, fname))
                neighborhood, center = model.group_divider(pc)
                x_vis, _ = model.MAE_encoder(neighborhood, center)
                global_mean = x_vis.mean(dim=1)
                z_pc = projector(global_mean)
                z_pc_list.append(z_pc.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    
    z_pc = np.stack(z_pc_list)
    np.save(output_file, z_pc)
    print(f"\n[DONE] Saved {len(z_pc)} embeddings to {output_file}")
    print(f"Shape: {z_pc.shape}")


if __name__ == "__main__":
    main()