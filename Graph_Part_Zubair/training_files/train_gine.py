#!/usr/bin/env python3


import os
import sys

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from tqdm import tqdm
import time

from utils.config import load_config
from data.graph_dataset import SceneGraphDataset


# Model Definition (keep as-is)

class EdgeEncoder(nn.Module):
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, e):
        return self.net(e)


class GineLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="add")
        self.msg_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.upd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, e):
        out = self.propagate(edge_index=edge_index, x=x, e=e)
        return self.upd(out)

    def message(self, x_j, e):
        return self.msg_x(x_j) + self.msg_e(e)


class Gine(nn.Module):
    def __init__(
        self,
        x_dim,
        edge_dim,
        out_dim,
        hidden_dim=192,
        layers=3,
        pooling="txrx",
        geo_dim=16,
    ):
        super().__init__()

        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.edge_enc = EdgeEncoder(edge_dim, hidden_dim)

        if geo_dim > 0:
            self.geo_proj = nn.Sequential(
                nn.Linear(128, geo_dim),
                nn.ReLU(),
            )
            in_dim = (x_dim - 128) + geo_dim
        else:
            self.geo_proj = None
            in_dim = x_dim - 128

        self.node_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [GineLayer(hidden_dim) for _ in range(layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)]
        )

        pool_out = hidden_dim * 3 if pooling == "txrx" else hidden_dim

        self.head = nn.Sequential(
            nn.Linear(pool_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x = data.x
        x_geo = x[:, :128]
        x_other = x[:, 128:]

        if self.geo_proj is not None:
            x = torch.cat([self.geo_proj(x_geo), x_other], dim=1)
        else:
            x = x_other

        h = self.node_in(x)
        e = self.edge_enc(data.edge_attr)

        for mp, ln in zip(self.layers, self.norms):
            h = ln(h + mp(h, data.edge_index, e))

        batch = getattr(
            data,
            "batch",
            torch.zeros(h.size(0), dtype=torch.long, device=h.device),
        )

        if self.pooling == "mean":
            g = global_mean_pool(h, batch)
        else:
            nt = data.node_type
            obj = global_mean_pool(h[nt == 0], batch[nt == 0])
            tx = global_mean_pool(h[nt == 1], batch[nt == 1])
            rx = global_mean_pool(h[nt == 2], batch[nt == 2])
            g = torch.cat([obj, tx, rx], dim=1)

        return self.head(g)


# Data Loading

DATA_ROOT = "/data/hafeez/graphdata"

DATASETS = {
    "rooms": os.path.join(DATA_ROOT, "rooms"),
    "rooms_update": os.path.join(DATA_ROOT, "rooms_update"),
    "roomsmoved": os.path.join(DATA_ROOT, "roomsmoved"),
}


def get_num_samples(dataset_path):
    npy_path = os.path.join(dataset_path, "dataset.npy")
    arr = np.load(npy_path, allow_pickle=True)
    return len(arr[0])


def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


# Training

def train_one_epoch(model, loader, optimizer, loss_fn, device, rank, sampler, epoch):
    model.train()
    total_loss = 0.0
    
    # Set epoch for proper shuffling in DistributedSampler
    if sampler is not None:
        sampler.set_epoch(epoch)
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc='Training', leave=False)
    else:
        pbar = loader
    
    for data in pbar:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        y_hat = model(data)
        loss = loss_fn(y_hat, data.y.view_as(y_hat))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if rank == 0 and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # Average loss across all processes
    avg_loss = total_loss / len(loader)
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    
    return avg_loss_tensor.item()


def validate(model, loader, loss_fn, device, rank):
    model.eval()
    total_loss = 0.0
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc='Validation', leave=False)
    else:
        pbar = loader
    
    with torch.no_grad():
        for data in pbar:
            data = data.to(device, non_blocking=True)
            y_hat = model(data)
            loss = loss_fn(y_hat, data.y.view_as(y_hat))
            total_loss += loss.item()
            
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    # Average loss across all processes
    avg_loss = total_loss / len(loader)
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    
    return avg_loss_tensor.item()


def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Gine Training Loss (8 GPUs)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Main Training Function

def train_worker(rank, world_size, cfg, args):
    """Training function for each GPU process"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print(f"Training on {world_size} GPUs")
        print(f"Device: {device}")
    
    # Dataset setup
    train_dataset_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(train_dataset_path, "dataset.npy")
    z_pc_path = os.path.join(train_dataset_path, "z_pc.npy")
    
    n_total = get_num_samples(train_dataset_path)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))
    
    if rank == 0:
        print(f"Train: {n_train}, Val: {n_val}")
        print(f"Samples per GPU: ~{n_train // world_size}")
    
    # Create full datasets (DistributedSampler will handle splitting)
    train_dataset = SceneGraphDataset(cfg, train_indices, z_pc_path=z_pc_path)
    val_dataset = SceneGraphDataset(cfg, val_indices, z_pc_path=z_pc_path)
    
    # Create DistributedSamplers for proper data distribution
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    # Create data loaders with DistributedSampler
    # Note: Don't use shuffle=True when using sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=0,  # Must be 0 for PyG with DDP
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,  # Use sampler
        num_workers=0,  # Must be 0 for PyG with DDP
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Train batches per GPU: {len(train_loader)}")
        print(f"Val batches per GPU: {len(val_loader)}")
    
    # Model
    sample = train_dataset[0]
    model = Gine(
        x_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        out_dim=sample.y.shape[0],
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        pooling=args.pooling,
        geo_dim=args.geo_dim,
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    if rank == 0:
        pbar_epochs = tqdm(range(args.epochs), desc="Training Progress", unit="epoch")
    else:
        pbar_epochs = range(args.epochs)
    
    for epoch in pbar_epochs:
        # Pass sampler and epoch to train_one_epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, rank, train_sampler, epoch
        )
        val_loss = validate(model, val_loader, loss_fn, device, rank)
        
        if rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),  # Save unwrapped model
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "best_val_loss": best_val_loss,
                    },
                    os.path.join(ROOT, "Gine_best.pt"),
                )
            
            percent = (epoch + 1) / args.epochs * 100
            print(f"Epoch {epoch+1:03d}/{args.epochs} ({percent:.1f}%) | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
    
    # Save final model (only rank 0)
    if rank == 0:
        print(f"\nBest val loss: {best_val_loss:.6f}")
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            os.path.join(ROOT, "Gine.pt"),
        )
        plot_loss(train_losses, val_losses, os.path.join(ROOT, "loss_curve.png"))
    
    # Cleanup
    cleanup_distributed()


def main():
    import argparse
    import torch.multiprocessing as mp
    
    p = argparse.ArgumentParser(description="Train Gine model with multi-GPU support")
    p.add_argument("--dataset", type=str, default="rooms_update")
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--pooling", choices=["mean", "txrx"], default="txrx")
    p.add_argument("--geo_dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    
    args = p.parse_args()
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)
    
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"Requested {args.num_gpus} GPUs but only {available_gpus} available.")
        print(f"Using {available_gpus} GPUs instead.")
        args.num_gpus = available_gpus
    
    print(f"Starting distributed training on {args.num_gpus} GPUs")
    
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    
    # Spawn processes for each GPU
    mp.spawn(
        train_worker,
        args=(args.num_gpus, cfg, args),
        nprocs=args.num_gpus,
        join=True
    )


if __name__ == "__main__":
    main()