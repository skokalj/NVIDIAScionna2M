#!/usr/bin/env python3
"""
Train GPS (Graph Transformer) for CIR prediction + extract z_xml embeddings.
=============================================================================
Mirrors the GINE supervised pipeline exactly:
  - Same dataset, same 90/10 split, same DDP setup
  - MSE loss on data.y (CIR targets)
  - Extracts z_xml = graph-level embedding BEFORE the prediction head
  - Saves z_xml.npy over the full dataset

Architecture:
  GPS layers (EdgeConditionedConv local MPNN + Multi-Head Global Attention)
  with Dir-GNN-style directional aggregation.
  Type-aware pooling: [obj_pool | tx_pool | rx_pool] → z_xml (hidden_dim*3)
  Prediction head: z_xml → CIR output

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python training/train_gps.py \\
      --dataset rooms_update --epochs 200 --num_gpus 8

  # Single GPU
  python training/train_gps.py --dataset rooms_update --epochs 50 --num_gpus 1
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GPSConv, global_mean_pool, MessagePassing
from tqdm import tqdm

# ─── Project paths ───────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.exists(os.path.join(ROOT, "utils")):
    ROOT = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(ROOT, "utils")):
        ROOT = os.path.expanduser("~/Graph")
sys.path.insert(0, ROOT)

from utils.config import load_config
from data.graph_dataset import SceneGraphDataset

DATA_ROOT = "/data/hafeez/graphdata"
DATASETS = {
    "rooms":        os.path.join(DATA_ROOT, "rooms"),
    "rooms_update": os.path.join(DATA_ROOT, "rooms_update"),
    "roomsmoved":   os.path.join(DATA_ROOT, "roomsmoved"),
    "roomsMO":      os.path.join(DATA_ROOT, "roomsMO"),
}


# ═══════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════

class EdgeConditionedConv(MessagePassing):
    """
    Memory-efficient edge-conditioned message passing.
    Projects edge features to hidden_dim and adds to messages.
    O(E × hidden_dim) instead of NNConv's O(E × hidden_dim²).
    """
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="add")
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        e = self.edge_proj(edge_attr)
        return self.msg_mlp(x_j + e)


class DirConv(nn.Module):
    """
    Dir-GNN style directional message passing.
    Two parallel passes (forward + backward edges) with learnable alpha mixing.
    """
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.conv_fwd = EdgeConditionedConv(hidden_dim, edge_dim)
        self.conv_bwd = EdgeConditionedConv(hidden_dim, edge_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr):
        h_fwd = self.conv_fwd(x, edge_index, edge_attr)
        edge_index_rev = edge_index.flip(0)
        h_bwd = self.conv_bwd(x, edge_index_rev, edge_attr)
        alpha = torch.sigmoid(self.alpha)
        return alpha * h_fwd + (1 - alpha) * h_bwd


# ═══════════════════════════════════════════════════════════════════
# GPS MODEL (supervised CIR prediction, mirrors GINE structure)
# ═══════════════════════════════════════════════════════════════════

class GPS(nn.Module):
    """
    GPS Graph Transformer for CIR prediction.

    Architecture mirrors GINE:
      - Input projection (with optional geometry compression)
      - GPS message-passing layers (DirConv local + multi-head global attention)
      - Type-aware pooling: [obj_pool | tx_pool | rx_pool]
      - Prediction head → CIR output

    z_xml = the pooled graph embedding BEFORE the prediction head.
    """

    def __init__(
        self,
        x_dim,
        edge_dim,
        out_dim,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        pooling="txrx",
        geo_dim=32,
        dropout=0.1,
    ):
        super().__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        # ── Geometry compression (first 128 dims = Point-MAE z_pc) ──
        if geo_dim > 0:
            self.geo_proj = nn.Sequential(nn.Linear(128, geo_dim), nn.ReLU())
            node_in_dim = geo_dim + (x_dim - 128)
        else:
            self.geo_proj = None
            node_in_dim = x_dim - 128

        # ── Node input projection ──
        self.node_in = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── Edge input projection ──
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── GPS layers ──
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = DirConv(hidden_dim, hidden_dim)
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=num_heads,
                dropout=dropout,
                attn_type='multihead',
                attn_kwargs={'dropout': dropout},
                norm='layer_norm',
            )
            self.gps_layers.append(gps_layer)

        self.final_norm = nn.LayerNorm(hidden_dim)

        # ── Pooling output dimension ──
        pool_out = hidden_dim * 3 if pooling == "txrx" else hidden_dim

        # ── Prediction head (z_xml → CIR) ──
        self.head = nn.Sequential(
            nn.Linear(pool_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode(self, data):
        """
        Forward pass through GPS layers + pooling.
        Returns z_xml (graph-level embedding before head).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(
            data, "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device),
        )

        # Geometry compression
        if self.geo_proj is not None:
            x = torch.cat([self.geo_proj(x[:, :128]), x[:, 128:]], dim=1)
        else:
            x = x[:, 128:]

        # Node projection
        h = self.node_in(x)

        # Edge projection
        e = self.edge_proj(edge_attr)

        # GPS message passing
        for gps_layer in self.gps_layers:
            h = gps_layer(h, edge_index, batch, edge_attr=e)

        h = self.final_norm(h)

        # Type-aware pooling → z_xml
        if self.pooling == "mean":
            g = global_mean_pool(h, batch)
        else:
            nt = data.node_type
            obj = global_mean_pool(h[nt == 0], batch[nt == 0])
            tx  = global_mean_pool(h[nt == 1], batch[nt == 1])
            rx  = global_mean_pool(h[nt == 2], batch[nt == 2])
            g = torch.cat([obj, tx, rx], dim=1)

        return g

    def forward(self, data):
        """Full forward: returns CIR prediction."""
        g = self._encode(data)
        return self.head(g)

    def get_graph_embedding(self, data):
        """Extract z_xml (graph embedding before head). Used for extraction."""
        return self._encode(data)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def get_num_samples(dataset_path):
    npy_path = os.path.join(dataset_path, "dataset.npy")
    arr = np.load(npy_path, allow_pickle=True)
    return len(arr[0])


# ═══════════════════════════════════════════════════════════════════
# DISTRIBUTED HELPERS
# ═══════════════════════════════════════════════════════════════════

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank,
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════
# TRAINING + VALIDATION
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, loss_fn, device, rank, sampler, epoch, scaler):
    model.train()
    total_loss = 0.0

    if sampler is not None:
        sampler.set_epoch(epoch)

    bar = tqdm(loader, desc='Training', leave=False) if rank == 0 else loader

    for data in bar:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            y_hat = model(data)
            loss = loss_fn(y_hat, data.y.view_as(y_hat))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if rank == 0 and hasattr(bar, 'set_postfix'):
            bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(loader)
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    return avg_loss_tensor.item()


@torch.no_grad()
def validate(model, loader, loss_fn, device, rank):
    model.eval()
    total_loss = 0.0

    bar = tqdm(loader, desc='Validation', leave=False) if rank == 0 else loader

    for data in bar:
        data = data.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            y_hat = model(data)
            loss = loss_fn(y_hat, data.y.view_as(y_hat))
        total_loss += loss.item()
        if rank == 0 and hasattr(bar, 'set_postfix'):
            bar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(loader)
    avg_loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
    return avg_loss_tensor.item()


# ═══════════════════════════════════════════════════════════════════
# Z_XML EXTRACTION (chunked, single-GPU, after training)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings_chunked(model, cfg, dataset_name, device,
                                chunk_size=500, batch_size=32):
    """Extract z_xml embeddings in chunks over the FULL dataset."""
    dataset_path = os.path.join(DATA_ROOT, dataset_name)

    cfg_copy = cfg.copy()
    cfg_copy["dataset"]["path"] = os.path.join(dataset_path, "dataset.npy")
    z_pc_path = os.path.join(dataset_path, "z_pc.npy")

    data = np.load(cfg_copy["dataset"]["path"], allow_pickle=True)
    n_total = len(data[0])
    print(f"  Total samples for extraction: {n_total}")

    all_embeddings = []
    n_chunks = (n_total + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_total)
        chunk_indices = list(range(start_idx, end_idx))

        print(f"  Chunk {chunk_idx+1}/{n_chunks}: samples {start_idx}-{end_idx-1}")

        dataset = SceneGraphDataset(cfg_copy, chunk_indices, z_pc_path=z_pc_path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        chunk_embs = []
        for batch_data in tqdm(loader, desc=f"  Chunk {chunk_idx+1}", leave=False):
            batch_data = batch_data.to(device)
            g = model.get_graph_embedding(batch_data)
            chunk_embs.append(g.cpu().numpy())

        chunk_embs = np.concatenate(chunk_embs, axis=0)
        all_embeddings.append(chunk_embs)
        print(f"    Extracted {chunk_embs.shape[0]} embeddings, dim={chunk_embs.shape[1]}")

    return np.concatenate(all_embeddings, axis=0)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_loss(train_losses, val_losses, save_path, num_gpus):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'GPS Training Loss ({num_gpus} GPUs)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# DDP WORKER
# ═══════════════════════════════════════════════════════════════════

def train_worker(rank, world_size, cfg, args):
    """Training function for each GPU process."""

    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n  Training on {world_size} GPUs")

    # ── Dataset ──
    train_dataset_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(train_dataset_path, "dataset.npy")
    z_pc_path = os.path.join(train_dataset_path, "z_pc.npy")

    n_total = get_num_samples(train_dataset_path)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val

    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))

    if rank == 0:
        print(f"  Dataset: {args.dataset} | Train: {n_train}, Val: {n_val}")
        print(f"  Per-GPU batch: {args.batch_size} | "
              f"Effective: {args.batch_size * world_size}")

    train_dataset = SceneGraphDataset(cfg, train_indices, z_pc_path=z_pc_path)
    val_dataset = SceneGraphDataset(cfg, val_indices, z_pc_path=z_pc_path)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=False,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=0, pin_memory=True,
    )

    if rank == 0:
        print(f"  Train batches/GPU: {len(train_loader)} | "
              f"Val batches/GPU: {len(val_loader)}")

    # ── Model ──
    sample = train_dataset[0]
    model = GPS(
        x_dim=sample.x.shape[1],
        edge_dim=sample.edge_attr.shape[1],
        out_dim=sample.y.shape[0],
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.num_heads,
        pooling=args.pooling,
        geo_dim=args.geo_dim,
        dropout=args.dropout,
    ).to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        z_dim = args.hidden_dim * 3 if args.pooling == "txrx" else args.hidden_dim
        print(f"  Parameters: {n_params:,}")
        print(f"  z_xml dim: {z_dim}")
        print(f"  GPS: {args.layers} layers, {args.num_heads} heads, "
              f"hidden={args.hidden_dim}")

    # ── Optimizer + scheduler (matches GINE but with cosine LR) ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # ── Training loop ──
    save_dir = os.path.join(ROOT, args.save_dir)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    t0 = time.time()

    if rank == 0:
        print(f"\n  Training {args.epochs} epochs (patience={args.patience})")
        print(f"  Mixed precision: ON (AMP)")
        print("─" * 100)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, rank, train_sampler, epoch, scaler,
        )
        val_loss = validate(model, val_loader, loss_fn, device, rank)
        scheduler.step()

        if rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "best_val_loss": best_val_loss,
                        "args": vars(args),
                    },
                    os.path.join(save_dir, "GPS_best.pt"),
                )
            else:
                patience_counter += 1

            lr = optimizer.param_groups[0]["lr"]
            eta = (time.time() - t0) / (epoch + 1) * (args.epochs - epoch - 1)
            print(
                f"Epoch {epoch+1:03d}/{args.epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {lr:.1e} | ETA: {eta/60:.1f}m | "
                f"{'★BEST' if is_best else f'p={patience_counter}/{args.patience}'}"
            )

        # Early stopping broadcast
        stop_flag = torch.tensor(
            [1 if (rank == 0 and patience_counter >= args.patience) else 0],
            device=device,
        )
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            if rank == 0:
                print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # ── Post-training (rank 0 only) ──
    if rank == 0:
        elapsed = time.time() - t0
        print(f"\n  Training done in {elapsed/60:.1f} min")
        print(f"  Best val loss: {best_val_loss:.6f}")

        # Save final checkpoint
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "args": vars(args),
            },
            os.path.join(save_dir, "GPS.pt"),
        )

        # Plot loss curves
        plot_loss(
            train_losses, val_losses,
            os.path.join(save_dir, "loss_curve.png"),
            world_size,
        )

        # ── Extract z_xml from best checkpoint ──
        print("\n" + "=" * 65)
        print("Z_XML EXTRACTION")
        print("=" * 65)

        # Load best model (unwrapped)
        best_model = GPS(
            x_dim=sample.x.shape[1],
            edge_dim=sample.edge_attr.shape[1],
            out_dim=sample.y.shape[0],
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            num_heads=args.num_heads,
            pooling=args.pooling,
            geo_dim=args.geo_dim,
            dropout=args.dropout,
        ).to(device)

        ckpt = torch.load(
            os.path.join(save_dir, "GPS_best.pt"),
            map_location=device, weights_only=False,
        )
        best_model.load_state_dict(ckpt["model_state_dict"])
        best_model.eval()
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']+1})")

        # Extract over full dataset
        z_xml = extract_embeddings_chunked(
            best_model, cfg, args.dataset, device,
            chunk_size=args.chunk_size, batch_size=args.extract_batch_size,
        )

        z_xml_path = os.path.join(save_dir, "z_xml.npy")
        np.save(z_xml_path, z_xml)

        # ── Summary ──
        print("\n" + "=" * 65)
        print("GPS — SUMMARY")
        print("=" * 65)
        print(f"  Dataset      : {args.dataset} ({get_num_samples(DATASETS[args.dataset])} samples)")
        print(f"  Architecture : GPS ({args.layers} layers, {args.num_heads} heads)")
        print(f"  Hidden dim   : {args.hidden_dim}")
        print(f"  z_xml dim    : {z_xml.shape[1]}")
        print(f"  z_xml shape  : {z_xml.shape}")
        print(f"  GPUs         : {world_size}")
        print(f"  Train time   : {elapsed/60:.1f} min")
        print(f"  Best val MSE : {best_val_loss:.6f}")
        print(f"\n  Outputs → {save_dir}/")
        print(f"    GPS_best.pt      — best checkpoint")
        print(f"    GPS.pt           — final checkpoint")
        print(f"    loss_curve.png   — training curves")
        print(f"    z_xml.npy        — extracted embeddings ({z_xml.shape})")
        print("=" * 65)

    cleanup_distributed()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Train GPS for CIR prediction + extract z_xml"
    )

    # Dataset
    p.add_argument("--dataset", type=str, default="rooms_update",
                   choices=list(DATASETS.keys()))

    # Architecture
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--pooling", choices=["mean", "txrx"], default="txrx")
    p.add_argument("--geo_dim", type=int, default=32,
                   help="Compressed dim for 128-dim z_pc (0=skip)")
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num_gpus", type=int, default=8)

    # Extraction
    p.add_argument("--chunk_size", type=int, default=500)
    p.add_argument("--extract_batch_size", type=int, default=32)

    # Output
    p.add_argument("--save_dir", type=str, default="gps_results",
                   help="Output directory (relative to project root)")

    args = p.parse_args()

    # ── GPU check ──
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"  [WARN] Requested {args.num_gpus} GPUs but "
              f"device_count()={available_gpus}")
        print(f"  [WARN] Set CUDA_VISIBLE_DEVICES if more are available")
    args.num_gpus = min(args.num_gpus, max(available_gpus, 1))

    # ── Print config ──
    print("=" * 65)
    print("GPS — CIR Prediction + z_xml Extraction")
    print("=" * 65)
    print(f"  Architecture : GPS (EdgeCondConv + Dir-GNN + MultiHead Attn)")
    print(f"  GPUs         : {args.num_gpus} detected")
    for i in range(args.num_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Hidden dim   : {args.hidden_dim}")
    print(f"  Layers       : {args.layers}")
    print(f"  Heads        : {args.num_heads}")
    print(f"  Pooling      : {args.pooling}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch/GPU    : {args.batch_size}")
    print(f"  Effective BS : {args.batch_size * args.num_gpus}")
    print(f"  LR           : {args.lr}")
    print(f"  Save to      : {args.save_dir}/")

    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))

    mp.spawn(
        train_worker,
        args=(args.num_gpus, cfg, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()