#!/usr/bin/env python3
"""
XML Scene Graph Autoencoder — Standalone z_xml Encoder-Decoder (Multi-GPU DDP)
===============================================================================
Trains a fresh GNN autoencoder purely on scene graph structure with NO CIR labels.
After training, compares the new z_xml against an existing embedding (z_xml_ae.npy).

Architecture:
  - Encoder : GineXmlEncoder  → z_xml  (hidden_dim * 3, txrx pooling)
  - Decoder : GineXmlDecoder  → reconstructs node features + edge attributes

Reconstruction targets (from node feature layout, 170-dim):
  [128:160]  material embeddings  (32-dim)   → MSE
  [160:163]  centroids            (3-dim)    → MSE
  [166:169]  node_type one-hot    (3-dim)    → CrossEntropy
  [169:170]  frequency            (1-dim)    → MSE
  edge_attr  (4-dim)                         → MSE

Usage:
  # 8-GPU DDP (default)
  python train_xml_autoencoder.py --dataset rooms_update --epochs 150

  # Compare against existing z_xml_ae after training
  python train_xml_autoencoder.py --dataset rooms_update --epochs 150 \\
      --compare_with ae_results/z_xml_ae.npy

  # Single GPU
  python train_xml_autoencoder.py --dataset rooms_update --epochs 150 --num_gpus 1
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
from torch_geometric.nn import MessagePassing, global_mean_pool
from scipy.spatial.distance import cdist
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.config import load_config
from data.graph_dataset import SceneGraphDataset

DATA_ROOT = "/data/hafeez/graphdata"
DATASETS = {
    "rooms":        os.path.join(DATA_ROOT, "rooms"),
    "rooms_update": os.path.join(DATA_ROOT, "rooms_update"),
    "roomsmoved":   os.path.join(DATA_ROOT, "roomsmoved"),
}
LOSS_KEYS = ["total", "centroid", "material", "edge", "node_type", "frequency"]


# ─────────────────────────────────────────────
# DISTRIBUTED HELPERS
# ─────────────────────────────────────────────

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"   # unique port — won't clash with other scripts
    dist.init_process_group("nccl", init_method="env://",
                             world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


# ─────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────

class EdgeEncoder(nn.Module):
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, e):
        return self.net(e)


class GineLayer(MessagePassing):
    """Single GINE message-passing layer."""
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")
        self.msg_x = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.msg_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.upd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, e):
        return self.upd(self.propagate(edge_index=edge_index, x=x, e=e))

    def message(self, x_j, e):
        return self.msg_x(x_j) + self.msg_e(e)


# ─────────────────────────────────────────────
# ENCODER
# ─────────────────────────────────────────────

class GineXmlEncoder(nn.Module):
    """
    Scene graph → z_xml.

    Output dim = hidden_dim * 3  (obj pool | tx pool | rx pool).
    geo_dim > 0 compresses the leading 128-dim PC embedding before processing.
    """

    def __init__(self, x_dim: int, edge_dim: int,
                 hidden_dim: int = 256, layers: int = 4, geo_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = hidden_dim * 3

        # Optional geo-feature compression
        if geo_dim > 0:
            self.geo_proj = nn.Sequential(nn.Linear(128, geo_dim), nn.ReLU())
            in_dim = geo_dim + (x_dim - 128)
        else:
            self.geo_proj = None
            in_dim = x_dim - 128           # skip raw PC features entirely

        self.node_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = EdgeEncoder(edge_dim, hidden_dim)
        self.mp_layers = nn.ModuleList([GineLayer(hidden_dim) for _ in range(layers)])
        self.norms     = nn.ModuleList([nn.LayerNorm(hidden_dim)  for _ in range(layers)])

    def forward(self, data):
        x = data.x
        if self.geo_proj is not None:
            x = torch.cat([self.geo_proj(x[:, :128]), x[:, 128:]], dim=1)
        else:
            x = x[:, 128:]

        h = self.node_in(x)
        e = self.edge_enc(data.edge_attr)

        for mp, ln in zip(self.mp_layers, self.norms):
            h = ln(h + mp(h, data.edge_index, e))

        batch = getattr(data, "batch",
                        torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        nt = data.node_type
        obj = global_mean_pool(h[nt == 0], batch[nt == 0])
        tx  = global_mean_pool(h[nt == 1], batch[nt == 1])
        rx  = global_mean_pool(h[nt == 2], batch[nt == 2])
        z   = torch.cat([obj, tx, rx], dim=1)      # (B, hidden_dim*3)
        return z, h, batch


# ─────────────────────────────────────────────
# DECODER
# ─────────────────────────────────────────────

class DecoderLayer(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")
        self.msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.upd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index):
        return self.upd(self.propagate(edge_index=edge_index, x=x))

    def message(self, x_j):
        return self.msg(x_j)


class GineXmlDecoder(nn.Module):
    """
    z_xml + graph topology → reconstructed node/edge features.

    Uses learnable node queries conditioned on z to initialise node states,
    then refines with message-passing over the original edge_index.
    """

    def __init__(self, z_dim: int, max_nodes: int, hidden_dim: int = 256, layers: int = 4):
        super().__init__()
        self.max_nodes  = max_nodes
        self.hidden_dim = hidden_dim

        # Per-node learnable prototypes
        self.node_queries = nn.Parameter(torch.randn(max_nodes, hidden_dim) * 0.02)

        # Project z into node-conditioning vector
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Merge query + conditioning
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mp_layers = nn.ModuleList([DecoderLayer(hidden_dim) for _ in range(layers)])
        self.norms     = nn.ModuleList([nn.LayerNorm(hidden_dim)  for _ in range(layers)])

        # Output heads
        self.centroid_head  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                            nn.Linear(hidden_dim // 2, 3))
        self.material_head  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, 32))
        self.node_type_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                            nn.Linear(hidden_dim // 2, 3))
        self.freq_head      = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
                                            nn.Linear(hidden_dim // 4, 1))
        self.edge_head      = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, 4))

    def forward(self, z, edge_index, batch, num_nodes_per_graph):
        B = z.shape[0]
        z_cond = self.z_proj(z)

        # Build initial node states per graph
        parts = []
        for i in range(B):
            n = num_nodes_per_graph[i].item()
            q   = self.node_queries[:n]
            c   = z_cond[i].unsqueeze(0).expand(n, -1)
            parts.append(self.combine(torch.cat([q, c], dim=1)))
        h = torch.cat(parts, dim=0)

        for mp, ln in zip(self.mp_layers, self.norms):
            h = ln(h + mp(h, edge_index))

        src, dst = edge_index
        return {
            "centroids":  self.centroid_head(h),
            "materials":  self.material_head(h),
            "node_types": self.node_type_head(h),
            "frequency":  self.freq_head(h),
            "edge_attrs": self.edge_head(torch.cat([h[src], h[dst]], dim=1)),
        }


# ─────────────────────────────────────────────
# FULL AUTOENCODER
# ─────────────────────────────────────────────

class XmlAutoencoder(nn.Module):
    def __init__(self, x_dim, edge_dim, max_nodes,
                 hidden_dim=256, enc_layers=4, dec_layers=4, geo_dim=32):
        super().__init__()
        self.encoder = GineXmlEncoder(x_dim, edge_dim, hidden_dim, enc_layers, geo_dim)
        self.decoder = GineXmlDecoder(self.encoder.z_dim, max_nodes, hidden_dim, dec_layers)
        self.z_dim   = self.encoder.z_dim

    def forward(self, data):
        z, h_enc, batch = self.encoder(data)
        num_nodes = torch.bincount(batch) if batch.max() > 0 else torch.tensor([batch.numel()])
        recon = self.decoder(z, data.edge_index, batch, num_nodes)
        return z, recon

    @torch.no_grad()
    def encode(self, data):
        z, _, _ = self.encoder(data)
        return z


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

class XmlAELoss(nn.Module):
    def __init__(self, lam_centroid=1.0, lam_material=0.5,
                 lam_edge=0.5, lam_node_type=0.2, lam_freq=0.1):
        super().__init__()
        self.w = dict(centroid=lam_centroid, material=lam_material,
                      edge=lam_edge, node_type=lam_node_type, frequency=lam_freq)

    def forward(self, recon, data):
        x = data.x
        L = {
            "centroid":  F.mse_loss(recon["centroids"],  x[:, 160:163]),
            "material":  F.mse_loss(recon["materials"],  x[:, 128:160]),
            "edge":      F.mse_loss(recon["edge_attrs"],  data.edge_attr),
            "node_type": F.cross_entropy(recon["node_types"], data.node_type.long()),
            "frequency": F.mse_loss(recon["frequency"],  x[:, 169:170]),
        }
        L["total"] = sum(self.w[k] * L[k] for k in self.w)
        return L


# ─────────────────────────────────────────────
# DDP TRAIN / VAL
# ─────────────────────────────────────────────

def _avg_losses(raw: dict, n_batches: int, device) -> dict:
    out = {}
    for k, v in raw.items():
        t = torch.tensor([v / max(n_batches, 1)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        out[k] = t.item()
    return out


def train_epoch(model, loader, optim, loss_fn, device, rank, sampler, epoch):
    model.train()
    sampler.set_epoch(epoch)
    acc, n = {k: 0.0 for k in LOSS_KEYS}, 0
    bar = tqdm(loader, desc="Train", leave=False) if rank == 0 else loader

    for data in bar:
        data = data.to(device, non_blocking=True)
        optim.zero_grad()
        z, recon = model(data)
        L = loss_fn(recon, data)
        L["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        for k in LOSS_KEYS:
            acc[k] += L[k].item()
        n += 1
        if rank == 0 and hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{L['total'].item():.5f}")

    return _avg_losses(acc, n, device)


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, rank):
    model.eval()
    acc, n = {k: 0.0 for k in LOSS_KEYS}, 0
    for data in loader:
        data = data.to(device, non_blocking=True)
        z, recon = model(data)
        L = loss_fn(recon, data)
        for k in LOSS_KEYS:
            acc[k] += L[k].item()
        n += 1
    return _avg_losses(acc, n, device)


# ─────────────────────────────────────────────
# EXTRACTION & ANALYSIS (rank 0 only)
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_z(model, loader, device) -> np.ndarray:
    model.eval()
    parts = []
    for data in tqdm(loader, desc="Extracting z_xml", leave=False):
        parts.append(model.encode(data.to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)


def compare_embeddings(z_new: np.ndarray, z_old: np.ndarray, save_dir: str, label_old="z_xml_ae"):
    """Compute CKA, mutual k-NN overlap, distance correlation; save plot."""
    print("\n" + "=" * 65)
    print(f"EMBEDDING COMPARISON: new z_xml  vs  {label_old}")
    print("=" * 65)

    n = min(len(z_new), len(z_old))
    z1, z2 = z_new[:n], z_old[:n]

    def linear_cka(X, Y):
        X, Y = X - X.mean(0), Y - Y.mean(0)
        num  = np.linalg.norm(X.T @ Y, "fro") ** 2
        den  = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
        return num / (den + 1e-10)

    cka = linear_cka(z1, z2)
    print(f"  Linear CKA:                   {cka:.4f}  (1=identical, 0=unrelated)")

    k     = 10
    n_sub = min(2000, n)
    idx   = np.random.choice(n, n_sub, replace=False)
    s1, s2 = z1[idx], z2[idx]
    d1, d2 = cdist(s1, s1), cdist(s2, s2)
    knn1   = np.argsort(d1, axis=1)[:, 1:k+1]
    knn2   = np.argsort(d2, axis=1)[:, 1:k+1]
    overlaps = [len(set(knn1[i]) & set(knn2[i])) / k for i in range(n_sub)]
    mean_ov  = float(np.mean(overlaps))
    print(f"  Mutual {k}-NN overlap:            {mean_ov:.4f}")

    triu   = np.triu_indices(n_sub, k=1)
    d1f, d2f = d1[triu], d2[triu]
    corr   = float(np.corrcoef(d1f, d2f)[0, 1])
    print(f"  Pairwise distance correlation: {corr:.4f}")

    # ── Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    si = np.random.choice(len(d1f), min(5000, len(d1f)), replace=False)

    axes[0].scatter(d1f[si], d2f[si], alpha=0.1, s=1, c="steelblue")
    axes[0].set_xlabel(f"z_xml dist"); axes[0].set_ylabel(f"{label_old} dist")
    axes[0].set_title(f"Pairwise Dist Corr: {corr:.3f}")

    axes[1].hist(overlaps, bins=20, edgecolor="black", color="coral", alpha=0.8)
    axes[1].axvline(mean_ov, color="darkred", linestyle="--", label=f"Mean={mean_ov:.3f}")
    axes[1].set_title(f"Mutual {k}-NN Overlap"); axes[1].legend()

    axes[2].bar(["CKA", "kNN", "Dist Corr"], [cka, mean_ov, corr],
                color=["steelblue", "coral", "seagreen"], edgecolor="black")
    axes[2].set_ylim(0, 1); axes[2].set_title("Similarity Summary")

    plt.suptitle(f"New z_xml  vs  {label_old}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "z_xml_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved plot → {path}")
    return {"cka": cka, "knn_overlap": mean_ov, "dist_corr": corr}


def plot_losses(train_h, val_h, save_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, key in zip(axes.flatten(), LOSS_KEYS):
        ax.plot([h[key] for h in train_h], label="Train", lw=2)
        ax.plot([h[key] for h in val_h],   label="Val",   lw=2)
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("XML Autoencoder Loss Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xml_ae_losses.png"), dpi=150)
    plt.close()


def plot_centroid_recon(model, loader, device, save_dir: str, n_samples=5):
    model.eval()
    data = next(iter(loader)).to(device)
    with torch.no_grad():
        z, recon = model(data)
    gt   = data.x[:, 160:163].cpu().numpy()
    pred = recon["centroids"].cpu().numpy()
    nt   = data.node_type.cpu().numpy()
    batch = (data.batch.cpu().numpy()
             if hasattr(data, "batch") else np.zeros(len(gt), dtype=int))

    n_g = min(n_samples, int(batch.max()) + 1)
    fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 5))
    if n_g == 1: axes = [axes]
    color_map = {0: "blue", 1: "red", 2: "green"}
    for idx, ax in enumerate(axes):
        m  = batch == idx
        gc, pc, gt_t = gt[m], pred[m], nt[m]
        colors = [color_map.get(int(t), "gray") for t in gt_t]
        ax.scatter(gc[:, 0], gc[:, 1], c=colors, marker="o", s=40, alpha=0.7, label="GT")
        ax.scatter(pc[:, 0], pc[:, 1], c=colors, marker="x", s=40, alpha=0.7, label="Pred")
        for j in range(len(gc)):
            ax.plot([gc[j,0], pc[j,0]], [gc[j,1], pc[j,1]], "gray", alpha=0.3, lw=0.5)
        ax.set_title(f"Graph {idx}\nBlue=Obj  Red=TX  Green=RX")
        ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.suptitle("Centroid Reconstruction Quality", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "centroid_recon.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# DDP WORKER
# ─────────────────────────────────────────────

def worker(rank, world_size, cfg, args, sample_info):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # ── Dataset ─────────────────────────────
    ds_path   = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(ds_path, "dataset.npy")
    z_pc_path = os.path.join(ds_path, "z_pc.npy")

    n_total = len(np.load(cfg["dataset"]["path"], allow_pickle=True)[0])
    n_val   = int(n_total * 0.1)
    n_train = n_total - n_val

    train_ds = SceneGraphDataset(cfg, list(range(n_train)),          z_pc_path=z_pc_path)
    val_ds   = SceneGraphDataset(cfg, list(range(n_train, n_total)), z_pc_path=z_pc_path)

    tr_samp = DistributedSampler(train_ds, world_size, rank, shuffle=True,  drop_last=False)
    va_samp = DistributedSampler(val_ds,   world_size, rank, shuffle=False, drop_last=False)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=tr_samp,
                           num_workers=0, pin_memory=True)
    va_loader = DataLoader(val_ds,   batch_size=args.batch_size, sampler=va_samp,
                           num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"\n  Dataset: {args.dataset}  |  Train: {n_train}  Val: {n_val}")
        print(f"  Effective batch size: {args.batch_size * world_size}")

    # ── Model ───────────────────────────────
    model = XmlAutoencoder(
        x_dim=sample_info["x_dim"], edge_dim=sample_info["edge_dim"],
        max_nodes=sample_info["max_nodes"],
        hidden_dim=args.hidden_dim, enc_layers=args.enc_layers,
        dec_layers=args.dec_layers, geo_dim=args.geo_dim,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_p:,}  |  z_dim: {model.module.z_dim}")

    # ── Optimiser ───────────────────────────
    optim     = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=args.epochs, eta_min=1e-6)
    loss_fn   = XmlAELoss(args.lam_centroid, args.lam_material,
                           args.lam_edge, args.lam_node_type, args.lam_freq)

    save_dir = os.path.join(ROOT, "xml_ae_results")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # ── Training loop ───────────────────────
    train_h, val_h = [], []
    best_val = float("inf")
    patience  = 0
    t0 = time.time()

    if rank == 0:
        print(f"\n  Training for {args.epochs} epochs  (patience={args.patience})\n" + "─"*110)

    for epoch in range(args.epochs):
        tr_L = train_epoch(model, tr_loader, optim, loss_fn, device, rank, tr_samp, epoch)
        va_L = val_epoch(model, va_loader, loss_fn, device, rank)
        scheduler.step()

        if rank == 0:
            train_h.append(tr_L); val_h.append(va_L)
            is_best = va_L["total"] < best_val
            if is_best:
                best_val = va_L["total"]; patience = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "train_history": train_h, "val_history": val_h,
                    "best_val_loss": best_val, "args": vars(args),
                }, os.path.join(save_dir, "xml_ae_best.pt"))
            else:
                patience += 1

            lr  = optim.param_groups[0]["lr"]
            eta = (time.time() - t0) / (epoch + 1) * (args.epochs - epoch - 1)
            print(
                f"Ep {epoch+1:03d}/{args.epochs} | "
                f"Tr {tr_L['total']:.5f} | Va {va_L['total']:.5f} | "
                f"C:{va_L['centroid']:.4f} M:{va_L['material']:.4f} "
                f"E:{va_L['edge']:.4f} NT:{va_L['node_type']:.3f} | "
                f"LR:{lr:.1e} | ETA:{eta/60:.1f}m | "
                f"{'★BEST' if is_best else f'p={patience}/{args.patience}'}"
            )

        # broadcast early-stop flag
        flag = torch.tensor(
            [1 if (rank == 0 and patience >= args.patience) else 0], device=device)
        dist.broadcast(flag, src=0)
        if flag.item():
            if rank == 0:
                print(f"\n  Early stop at epoch {epoch+1}")
            break

    # ── Post-training (rank 0 only) ─────────
    if rank == 0:
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed/60:.1f} min  |  Best val loss: {best_val:.6f}")

        # Save final checkpoint
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "train_history": train_h, "val_history": val_h, "args": vars(args),
        }, os.path.join(save_dir, "xml_ae_final.pt"))

        # Plots
        plot_losses(train_h, val_h, save_dir)

        # Load best model (single-GPU) for extraction & viz
        best_model = XmlAutoencoder(
            x_dim=sample_info["x_dim"], edge_dim=sample_info["edge_dim"],
            max_nodes=sample_info["max_nodes"],
            hidden_dim=args.hidden_dim, enc_layers=args.enc_layers,
            dec_layers=args.dec_layers, geo_dim=args.geo_dim,
        ).to(device)
        ckpt = torch.load(os.path.join(save_dir, "xml_ae_best.pt"), map_location=device)
        best_model.load_state_dict(ckpt["model_state_dict"])

        # Centroid reconstruction visualisation
        vis_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
        print("\n  Generating centroid reconstruction plot...")
        plot_centroid_recon(best_model, vis_loader, device, save_dir)

        # Extract z_xml over full dataset
        full_ds     = SceneGraphDataset(cfg, list(range(n_total)), z_pc_path=z_pc_path)
        full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)
        print("  Extracting z_xml embeddings...")
        z_new = extract_z(best_model, full_loader, device)
        out_path = os.path.join(save_dir, "z_xml.npy")
        np.save(out_path, z_new)
        print(f"  Saved → {out_path}  shape: {z_new.shape}")

        # ── Compare with old z_xml_ae ────────
        if args.compare_with:
            old_path = (args.compare_with if os.path.isabs(args.compare_with)
                        else os.path.join(ROOT, args.compare_with))
            if os.path.exists(old_path):
                z_old = np.load(old_path)
                label = os.path.basename(old_path).replace(".npy", "")
                print(f"\n  Loaded old embedding: {old_path}  shape: {z_old.shape}")
                metrics = compare_embeddings(z_new, z_old, save_dir, label_old=label)

                # Save metrics as text
                with open(os.path.join(save_dir, "comparison_metrics.txt"), "w") as f:
                    f.write(f"Comparison: new z_xml  vs  {label}\n")
                    f.write(f"  Linear CKA:             {metrics['cka']:.4f}\n")
                    f.write(f"  Mutual 10-NN overlap:   {metrics['knn_overlap']:.4f}\n")
                    f.write(f"  Dist correlation:       {metrics['dist_corr']:.4f}\n")
            else:
                print(f"\n  [WARN] --compare_with path not found: {old_path}")

        # ── Summary ─────────────────────────
        print("\n" + "=" * 65)
        print("SUMMARY")
        print("=" * 65)
        print(f"  Dataset    : {args.dataset}  ({n_total} samples)")
        print(f"  GPUs       : {world_size}")
        print(f"  z_dim      : {best_model.z_dim}")
        print(f"  Train time : {elapsed/60:.1f} min")
        print(f"  Best val   : {best_val:.6f}")
        print(f"\n  Per-component val losses (best epoch):")
        best_h = min(val_h, key=lambda h: h["total"])
        for k in LOSS_KEYS[1:]:
            print(f"    {k:12s}: {best_h[k]:.6f}")
        print(f"\n  Outputs saved to: {save_dir}/")
        print(f"    z_xml.npy, xml_ae_best.pt, xml_ae_final.pt")
        print(f"    xml_ae_losses.png, centroid_recon.png")
        if args.compare_with:
            print(f"    z_xml_comparison.png, comparison_metrics.txt")

    cleanup_ddp()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="XML Scene Graph Autoencoder — standalone z_xml")
    p.add_argument("--dataset",      type=str, default="rooms_update",
                   choices=list(DATASETS.keys()))
    p.add_argument("--hidden_dim",   type=int,   default=256,
                   help="Encoder/decoder hidden dim (z_dim = hidden_dim * 3)")
    p.add_argument("--enc_layers",   type=int,   default=4)
    p.add_argument("--dec_layers",   type=int,   default=4)
    p.add_argument("--geo_dim",      type=int,   default=32,
                   help="Compressed dim for leading 128-dim PC features (0 = skip entirely)")
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--num_gpus",     type=int,   default=8)
    # Loss weights
    p.add_argument("--lam_centroid",  type=float, default=1.0)
    p.add_argument("--lam_material",  type=float, default=0.5)
    p.add_argument("--lam_edge",      type=float, default=0.5)
    p.add_argument("--lam_node_type", type=float, default=0.2)
    p.add_argument("--lam_freq",      type=float, default=0.1)
    # Comparison
    p.add_argument("--compare_with", type=str, default=None,
                   help="Path to existing z_xml_ae.npy (or any .npy) to compare against")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available."); sys.exit(1)

    available = torch.cuda.device_count()
    args.num_gpus = min(args.num_gpus, available)

    print("=" * 65)
    print("XML SCENE GRAPH AUTOENCODER  —  standalone z_xml")
    print("=" * 65)
    for i in range(args.num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load config + inspect a sample before spawning
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    ds_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(ds_path, "dataset.npy")
    z_pc_path = os.path.join(ds_path, "z_pc.npy")

    tmp = SceneGraphDataset(cfg, [0], z_pc_path=z_pc_path)
    s   = tmp[0]
    sample_info = {
        "x_dim":     s.x.shape[1],
        "edge_dim":  s.edge_attr.shape[1],
        "max_nodes": s.x.shape[0],
    }
    print(f"\n  Graph dims — nodes: {sample_info['max_nodes']}  "
          f"x_dim: {sample_info['x_dim']}  edge_dim: {sample_info['edge_dim']}")
    print(f"  z_dim will be: {256 * 3}  (hidden_dim={args.hidden_dim})")
    print(f"  Epochs: {args.epochs}  Batch/GPU: {args.batch_size}  "
          f"Effective: {args.batch_size * args.num_gpus}\n")

    mp.spawn(worker,
             args=(args.num_gpus, cfg, args, sample_info),
             nprocs=args.num_gpus,
             join=True)


if __name__ == "__main__":
    main()