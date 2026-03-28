#!/usr/bin/env python3
"""
Extract z_XML embeddings from a trained Gine model.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from utils.config import load_config
from data.graph_dataset import SceneGraphDataset
from train_gine import Gine  


DATA_ROOT = "/data/hafeez/graphdata"


def extract_embeddings_chunked(model, cfg, dataset_name, device, chunk_size=500, batch_size=32):
    """Extract embeddings in chunks."""
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    
    cfg_copy = cfg.copy()
    cfg_copy["dataset"]["path"] = os.path.join(dataset_path, "dataset.npy")
    z_pc_path = os.path.join(dataset_path, "z_pc.npy")
    
    # Get total samples
    data = np.load(cfg_copy["dataset"]["path"], allow_pickle=True)
    n_total = len(data[0])
    print(f"Total samples: {n_total}")
    
    all_embeddings = []
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_total)
        chunk_indices = list(range(start_idx, end_idx))
        
        print(f"\nChunk {chunk_idx+1}/{n_chunks}: samples {start_idx}-{end_idx-1}")
        
        dataset = SceneGraphDataset(cfg_copy, chunk_indices, z_pc_path=z_pc_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        chunk_embeddings = []
        for data in tqdm(loader, desc=f"Processing chunk {chunk_idx+1}"):
            data = data.to(device)
            with torch.no_grad():
                g = model.get_graph_embedding(data)
            chunk_embeddings.append(g.cpu().numpy())
        
        chunk_embeddings = np.concatenate(chunk_embeddings, axis=0)
        all_embeddings.append(chunk_embeddings)
        print(f"  Extracted {chunk_embeddings.shape[0]} embeddings")
    
    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract z_XML embeddings")
    parser.add_argument("--dataset", type=str, default="rooms_update")
    parser.add_argument("--model_path", type=str, default="Gine_best.pt")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="z_xml.npy")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Z_XML EXTRACTION")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    
    # Get dimensions from a single sample
    dataset_path = os.path.join(DATA_ROOT, args.dataset)
    cfg_temp = cfg.copy()
    cfg_temp["dataset"]["path"] = os.path.join(dataset_path, "dataset.npy")
    z_pc_path = os.path.join(dataset_path, "z_pc.npy")
    
    temp_dataset = SceneGraphDataset(cfg_temp, [0], z_pc_path=z_pc_path)
    sample = temp_dataset[0]
    
    print(f"\nModel dimensions:")
    print(f"  x_dim: {sample.x.shape[1]}")
    print(f"  edge_dim: {sample.edge_attr.shape[1]}")
    print(f"  out_dim: {sample.y.shape[0]}")
    
    # Create and load model
    model = Gine(
        x_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        out_dim=sample.y.shape[0],
        hidden_dim=192,
        layers=3,
        pooling="txrx",
        geo_dim=16
    ).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"\n[ERROR] Model not found: {args.model_path}")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    model.eval()
    print(f"  Loaded weights from {args.model_path}")
    
    # Extract embeddings
    print(f"\n=== Extracting Embeddings ===")
    z_xml = extract_embeddings_chunked(
        model, cfg, args.dataset, device,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size
    )
    
    # Save
    output_path = os.path.join(ROOT, args.output)
    np.save(output_path, z_xml)
    
    print(f"\n[SUCCESS] Saved: {output_path}")
    print(f"Shape: {z_xml.shape}")


if __name__ == "__main__":
    main()