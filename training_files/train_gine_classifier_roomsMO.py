#!/usr/bin/env python3
"""
train_gine_classifier_roomsMO.py  

Binary environment classifier on a GINE backbone (trained from scratch).
Trained on /data/hafeez/roomsMO to distinguish:
    label 0 = "no sphere"   (dataset_nosphere.npy)  — 54 objects
    label 1 = "with sphere"  (dataset.npy)           — 55 objects

"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, classification_report

from utils.config import load_config
from data.graph_dataset import SceneGraphDataset


# Model Definition

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

    def extract_embedding(self, data):
        """Return the pooled graph embedding BEFORE the regression head."""
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
            return global_mean_pool(h, batch)

        nt = data.node_type
        obj = global_mean_pool(h[nt == 0], batch[nt == 0])
        tx = global_mean_pool(h[nt == 1], batch[nt == 1])
        rx = global_mean_pool(h[nt == 2], batch[nt == 2])
        return torch.cat([obj, tx, rx], dim=1)

    def get_pool_out_dim(self):
        return self.hidden_dim * 3 if self.pooling == "txrx" else self.hidden_dim

    def forward(self, data):
        g = self.extract_embedding(data)
        return self.head(g)


# Classifier


class GineClassifier(nn.Module):
    """
    GINE backbone + classification head.

    Concatenates the GNN pooled embedding with graph-structural features
    (node count per type) to give the head an explicit signal about scene
    composition differences (55 vs 54 objects).
    """

    STRUCT_DIM = 4  # [num_obj, num_tx, num_rx, total_nodes]

    def __init__(self, backbone: Gine, dropout: float = 0.3,
                 freeze_backbone: bool = False, use_struct_features: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.use_struct_features = use_struct_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        gnn_dim = backbone.get_pool_out_dim()
        in_dim = gnn_dim + (self.STRUCT_DIM if use_struct_features else 0)
        hidden = backbone.hidden_dim

        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1),
        )

    def _struct_features(self, data):
        """
        Extract graph-structural features per graph in the batch.
        
        Key: normalize so that the 55 vs 54 object difference is clearly
        visible. Use centered features: (count - 54.5) so the difference
        is +0.5 vs -0.5 instead of 0.917 vs 0.900.
        """
        batch = getattr(
            data, "batch",
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device),
        )
        nt = data.node_type
        batch_size = batch.max().item() + 1

        feats = torch.zeros(batch_size, self.STRUCT_DIM, device=data.x.device)
        for b in range(batch_size):
            mask = batch == b
            nt_b = nt[mask]
            n_obj = (nt_b == 0).sum().float()
            n_tx = (nt_b == 1).sum().float()
            n_rx = (nt_b == 2).sum().float()
            n_total = mask.sum().float()

            # Centered around expected midpoint so difference is clear
            feats[b, 0] = (n_obj - 54.5)        # +0.5 for sphere, -0.5 for no sphere
            feats[b, 1] = n_tx                   # should always be 1
            feats[b, 2] = n_rx                   # should always be 1
            feats[b, 3] = (n_total - 56.5)       # +0.5 for sphere, -0.5 for no sphere
        return feats

    def forward(self, data):
        if self.freeze_backbone:
            with torch.no_grad():
                g = self.backbone.extract_embedding(data)
        else:
            g = self.backbone.extract_embedding(data)

        if self.use_struct_features:
            sf = self._struct_features(data)
            g = torch.cat([g, sf], dim=1)

        return self.cls_head(g)

    def head_parameters(self):
        return list(self.cls_head.parameters())

    def backbone_parameters(self):
        return list(self.backbone.parameters())


# Dataset

ROOMSMO_ROOT = "/data/hafeez/roomsMO"


class LabeledSceneGraphDataset(torch.utils.data.Dataset):
    """
    Thin wrapper that overrides data.y with a binary label.
    
    IMPORTANT: We clone the data object to avoid mutating the cached
    version in the inner dataset, which could cause label issues.
    """

    def __init__(self, cfg, indices, z_pc_path, env_label: int):
        self.inner = SceneGraphDataset(cfg, indices, z_pc_path=z_pc_path)
        self.env_label = float(env_label)
        self._len = len(indices)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data = self.inner[idx].clone()  # CLONE to avoid cache mutation
        data.y = torch.tensor([self.env_label], dtype=torch.float32)
        return data


def get_z_pc_path():
    """Return path to z_pc.npy for roomsMO."""
    local = os.path.join(ROOMSMO_ROOT, "z_pc.npy")
    if os.path.exists(local):
        return local

    for fallback_name in ["rooms_update", "rooms", "roomsmoved"]:
        fallback = os.path.join("/data/hafeez/graphdata", fallback_name, "z_pc.npy")
        if os.path.exists(fallback):
            return fallback

    raise FileNotFoundError("Cannot find z_pc.npy for roomsMO or any fallback.")


def _count_samples(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    if isinstance(arr[0], (list, np.ndarray)):
        return len(arr[0])
    return len(arr)


def build_roomsMO_datasets(cfg, val_fraction=0.15, test_fraction=0.15, rank=0):
    """
    Build train/val/test datasets from roomsMO.
    - dataset.npy          → label 1 (with sphere, 55 objects)
    - dataset_nosphere.npy → label 0 (no sphere, 54 objects)
    """
    z_pc_path = get_z_pc_path()

    sphere_npy = os.path.join(ROOMSMO_ROOT, "dataset.npy")
    nosphere_npy = os.path.join(ROOMSMO_ROOT, "dataset_nosphere.npy")

    datasets = {
        "withsphere": (sphere_npy, 1),
        "nosphere": (nosphere_npy, 0),
    }

    train_sets, val_sets, test_sets = [], [], []

    for name, (npy_path, label) in datasets.items():
        n_total = _count_samples(npy_path)

        n_test = max(1, int(n_total * test_fraction))
        n_val = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val - n_test

        if rank == 0:
            print(f"  {name:12s}: {n_total:5d} total → "
                  f"train={n_train}, val={n_val}, test={n_test}")

        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["dataset"]["path"] = npy_path

        train_sets.append(
            LabeledSceneGraphDataset(cfg_copy, list(range(n_train)), z_pc_path, label)
        )
        val_sets.append(
            LabeledSceneGraphDataset(
                cfg_copy, list(range(n_train, n_train + n_val)), z_pc_path, label
            )
        )
        test_sets.append(
            LabeledSceneGraphDataset(
                cfg_copy, list(range(n_train + n_val, n_total)), z_pc_path, label
            )
        )

    return (
        torch.utils.data.ConcatDataset(train_sets),
        torch.utils.data.ConcatDataset(val_sets),
        torch.utils.data.ConcatDataset(test_sets),
    )


# Data sanity check


def run_sanity_check(train_dataset, val_dataset, test_dataset, train_loader):
    """
    Verify labels are correct and node counts match expectations.
    This runs ONLY on rank 0 before training starts.
    """
    print("\n" + "=" * 60)
    print("DATA SANITY CHECK")
    print("=" * 60)

    for split_name, dataset in [("Train", train_dataset),
                                 ("Val", val_dataset),
                                 ("Test", test_dataset)]:
        label_counts = {0: 0, 1: 0}
        obj_counts_by_label = {0: [], 1: []}
        n_check = min(200, len(dataset))

        for i in range(n_check):
            d = dataset[i]
            label = int(d.y.item())
            n_obj = (d.node_type == 0).sum().item()
            label_counts[label] = label_counts.get(label, 0) + 1
            obj_counts_by_label[label].append(n_obj)

        print(f"\n  {split_name} (checked {n_check}/{len(dataset)}):")
        print(f"    Label distribution: {label_counts}")
        for lab in sorted(obj_counts_by_label.keys()):
            counts = obj_counts_by_label[lab]
            if counts:
                print(f"    Label {lab}: obj_nodes min={min(counts)} "
                      f"max={max(counts)} mean={np.mean(counts):.1f}")

    # Check a few batches from loader
    print("\n  Batch check (first 3 batches from train_loader):")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        labels = batch.y.squeeze(-1).tolist()
        if not isinstance(labels, list):
            labels = [labels]

        # Count object nodes per graph in batch
        batch_idx = batch.batch
        nt = batch.node_type
        bs = batch_idx.max().item() + 1
        obj_per_graph = []
        for b in range(bs):
            mask = batch_idx == b
            obj_per_graph.append((nt[mask] == 0).sum().item())

        print(f"    Batch {i}: labels={labels[:8]}... "
              f"obj_counts={obj_per_graph[:8]}...")

    # Verify the critical assumption: label 1 → 55 obj, label 0 → 54 obj
    print("\n  CRITICAL CHECK: Does label match node count?")
    mismatches = 0
    for i in range(min(500, len(train_dataset))):
        d = train_dataset[i]
        label = int(d.y.item())
        n_obj = (d.node_type == 0).sum().item()
        expected_obj = 55 if label == 1 else 54
        if n_obj != expected_obj:
            mismatches += 1
            if mismatches <= 5:
                print(f"    MISMATCH idx={i}: label={label}, "
                      f"expected {expected_obj} obj, got {n_obj}")

    if mismatches == 0:
        print("    ✓ All labels match expected node counts!")
    else:
        print(f"    ✗ {mismatches} mismatches found — DATA ISSUE DETECTED")

    print("=" * 60 + "\n")
    return mismatches == 0


# Distributed helpers


def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    dist.destroy_process_group()


# Metrics


def binary_accuracy(logits, labels):
    preds = (logits.squeeze(-1) > 0.0).float()
    return (preds == labels.squeeze(-1)).float().mean().item()


# Train / validate


def train_one_epoch(model, loader, optimizer, loss_fn, device, rank, sampler,
                    epoch, grad_clip=1.0):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    if sampler is not None:
        sampler.set_epoch(epoch)

    pbar = tqdm(loader, desc=f"Train E{epoch+1}", leave=False) if rank == 0 else loader

    for data in pbar:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits, data.y.view_as(logits))
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )

        optimizer.step()

        total_loss += loss.item()
        total_acc += binary_accuracy(logits.detach(), data.y)
        n_batches += 1

        if rank == 0 and hasattr(pbar, "set_postfix"):
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)

    loss_t = torch.tensor([avg_loss], device=device)
    acc_t = torch.tensor([avg_acc], device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
    dist.all_reduce(acc_t, op=dist.ReduceOp.AVG)

    return loss_t.item(), acc_t.item()


@torch.no_grad()
def validate(model, loader, loss_fn, device, rank):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Val", leave=False) if rank == 0 else loader

    for data in pbar:
        data = data.to(device, non_blocking=True)
        logits = model(data)
        loss = loss_fn(logits, data.y.view_as(logits))
        total_loss += loss.item()
        total_acc += binary_accuracy(logits, data.y)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_batches, 1)

    loss_t = torch.tensor([avg_loss], device=device)
    acc_t = torch.tensor([avg_acc], device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
    dist.all_reduce(acc_t, op=dist.ReduceOp.AVG)

    return loss_t.item(), acc_t.item()


@torch.no_grad()
def evaluate_test(model, loader, device, rank):
    """Full evaluation on test set."""
    model.eval()
    all_preds, all_labels = [], []

    for data in loader:
        data = data.to(device, non_blocking=True)
        logits = model(data)
        preds = (logits.squeeze(-1) > 0.0).long()
        all_preds.append(preds.cpu())
        all_labels.append(data.y.squeeze(-1).long().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if rank == 0:
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        print(f"  Samples: {len(all_labels)}")
        print(f"  Accuracy: {(all_preds == all_labels).mean():.4f}")
        print()
        print("Confusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        print(f"  {'':>12s} Pred=NoSph  Pred=Sphere")
        print(f"  {'True=NoSph':>12s}   {cm[0, 0]:5d}       {cm[0, 1]:5d}")
        print(f"  {'True=Sphere':>12s}   {cm[1, 0]:5d}       {cm[1, 1]:5d}")
        print()
        print("Classification Report:")
        print(
            classification_report(
                all_labels, all_preds,
                target_names=["no_sphere", "with_sphere"],
                zero_division=0,
            )
        )

    return (all_preds == all_labels).mean()


# Struct-only baseline (for debugging)


class StructOnlyClassifier(nn.Module):
    """
    Baseline that ONLY uses node counts (no GNN).
    Should achieve ~100% accuracy if labels are correct.
    Use --struct_only to test this.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, data):
        batch = getattr(
            data, "batch",
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device),
        )
        nt = data.node_type
        batch_size = batch.max().item() + 1

        feats = torch.zeros(batch_size, 4, device=data.x.device)
        for b in range(batch_size):
            mask = batch == b
            nt_b = nt[mask]
            feats[b, 0] = (nt_b == 0).sum().float() - 54.5
            feats[b, 1] = (nt_b == 1).sum().float()
            feats[b, 2] = (nt_b == 2).sum().float()
            feats[b, 3] = mask.sum().float() - 56.5
        return self.net(feats)



# Plotting


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label="Train Loss", linewidth=2)
    ax1.plot(val_losses, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("GINE Classifier – Loss (roomsMO)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(train_accs, label="Train Acc", linewidth=2)
    ax2.plot(val_accs, label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("GINE Classifier – Accuracy (roomsMO)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.4, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved curves to {save_path}")


# Main worker 
#


def train_worker(rank, world_size, cfg, args):
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # ── 1. Datasets ──────────────────────────────────────────────────────────
    if rank == 0:
        print("\nBuilding datasets...")

    train_dataset, val_dataset, test_dataset = build_roomsMO_datasets(
        cfg, val_fraction=0.15, test_fraction=0.15, rank=rank
    )

    if rank == 0:
        print(
            f"\nTotal → Train: {len(train_dataset)}  "
            f"Val: {len(val_dataset)}  Test: {len(test_dataset)}"
        )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=0, pin_memory=True,
    )

    # ── 1b. Sanity check (rank 0 only) ──────────────────────────────────────
    if rank == 0:
        data_ok = run_sanity_check(
            train_dataset, val_dataset, test_dataset, train_loader
        )
        if not data_ok:
            print("WARNING: Data issues detected! Results may be unreliable.")
            print("Continuing anyway for debugging purposes...\n")

    dist.barrier()  # Wait for rank 0 to finish sanity check

    # ── 2. Model ─────────────────────────────────────────────────────────────
    sample = train_dataset[0]
    if rank == 0:
        print(f"Sample shapes: x={sample.x.shape}, "
              f"edge_attr={sample.edge_attr.shape}, y={sample.y.shape}")
        n_obj = (sample.node_type == 0).sum().item()
        n_tx = (sample.node_type == 1).sum().item()
        n_rx = (sample.node_type == 2).sum().item()
        print(f"Node types: {n_obj} objects, {n_tx} TX, {n_rx} RX "
              f"({sample.node_type.shape[0]} total)")

    if args.struct_only:
        # ── Struct-only baseline ─────────────────────────────────────────────
        if rank == 0:
            print("\n*** STRUCT-ONLY BASELINE (no GNN) ***")
        model = StructOnlyClassifier().to(device)
        model = DDP(model, device_ids=[rank])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()

    else:
        # ── Full GINE classifier ─────────────────────────────────────────────
        backbone = Gine(
            x_dim=sample.x.shape[1],
            edge_dim=sample.edge_attr.shape[1],
            out_dim=1,  # Will be removed below
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            pooling=args.pooling,
            geo_dim=args.geo_dim,
        )

        backbone.head = nn.Identity()

        if rank == 0:
            print(f"\nTraining from scratch (no pretrained weights)")

        model = GineClassifier(
            backbone,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,  # FIXED: use actual arg
            use_struct_features=args.use_struct_features,
        ).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        if rank == 0:
            total_p = sum(p.numel() for p in model.parameters())
            train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total params    : {total_p:,}")
            print(f"Trainable params: {train_p:,}")
            print(f"Backbone frozen : {args.freeze_backbone}")
            print(f"Struct features : {args.use_struct_features}")

        # ── Optimizer ────────────────────────────────────────────────────────
        if not args.freeze_backbone:
            param_groups = [
                {"params": model.module.head_parameters(), "lr": args.lr},
                {"params": model.module.backbone_parameters(),
                 "lr": args.lr * args.backbone_lr_scale},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
            if rank == 0:
                print(f"Fine-tuning: head LR={args.lr}, "
                      f"backbone LR={args.lr * args.backbone_lr_scale}")
        else:
            optimizer = torch.optim.AdamW(
                model.module.head_parameters(), lr=args.lr, weight_decay=1e-4
            )
            if rank == 0:
                print(f"Frozen backbone: head-only LR={args.lr}")

        loss_fn = nn.BCEWithLogitsLoss()

    # ── Test-only mode ───────────────────────────────────────────────────────
    if args.test_only:
        ckpt_path = args.test_checkpoint or os.path.join(
            ROOT, "results_classifier_roomsMO", "GineClassifier_roomsMO_best.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.module.load_state_dict(state)
            if rank == 0:
                print(f"Loaded classifier from {ckpt_path}")
        else:
            if rank == 0:
                print(f"ERROR: checkpoint not found: {ckpt_path}")
            cleanup_distributed()
            return

        dist.barrier()
        evaluate_test(model, test_loader, device, rank)
        cleanup_distributed()
        return

    # ── LR schedule ──────────────────────────────────────────────────────────
    warmup_epochs = min(5, args.epochs // 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0

    save_dir = os.path.join(ROOT, "results_classifier_roomsMO")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, rank,
            train_sampler, epoch, grad_clip=args.grad_clip
        )
        vl_loss, vl_acc = validate(model, val_loader, loss_fn, device, rank)
        scheduler.step()

        if rank == 0:
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            train_accs.append(tr_acc)
            val_accs.append(vl_acc)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} │ "
                f"Train loss {tr_loss:.4f}  acc {tr_acc:.3f} │ "
                f"Val loss {vl_loss:.4f}  acc {vl_acc:.3f} │ "
                f"LR {current_lr:.2e}"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "train_accs": train_accs,
                        "val_accs": val_accs,
                        "best_val_acc": best_val_acc,
                    },
                    os.path.join(save_dir, "GineClassifier_roomsMO_best.pt"),
                )
                print(f"  ★ New best val acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1

        # Broadcast patience for early stopping
        patience_t = torch.tensor([patience_counter], device=device)
        dist.broadcast(patience_t, src=0)
        patience_counter = int(patience_t.item())

        if args.patience > 0 and patience_counter >= args.patience:
            if rank == 0:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {args.patience} epochs)")
            break

    # ── Final test evaluation ────────────────────────────────────────────────
    best_ckpt = os.path.join(save_dir, "GineClassifier_roomsMO_best.pt")
    dist.barrier()

    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
        model.module.load_state_dict(state)
        if rank == 0:
            print(f"\nReloaded best checkpoint (val_acc={best_val_acc:.4f}) for test eval")

    dist.barrier()
    test_acc = evaluate_test(model, test_loader, device, rank)

    # ── Save final artefacts ─────────────────────────────────────────────────
    if rank == 0:
        print(f"\nBest val accuracy : {best_val_acc:.4f}")
        print(f"Test accuracy     : {test_acc:.4f}")

        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
            },
            os.path.join(save_dir, "GineClassifier_roomsMO_final.pt"),
        )
        plot_curves(
            train_losses, val_losses, train_accs, val_accs,
            os.path.join(save_dir, "cls_curves_roomsMO.png"),
        )

    cleanup_distributed()


# CLI



def main():
    import torch.multiprocessing as mp

    p = argparse.ArgumentParser(
        description="Train binary sphere classifier on GINE backbone (roomsMO) — from scratch"
    )

    # Backbone arch
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--pooling", choices=["mean", "txrx"], default="txrx")
    p.add_argument("--geo_dim", type=int, default=16)

    # Classifier training
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)

    # Backbone mode
    p.add_argument(
        "--freeze_backbone", action="store_true",
        help="Freeze backbone (train head only).",
    )
    p.add_argument("--backbone_lr_scale", type=float, default=0.1,
                   help="LR multiplier for backbone when fine-tuning (default 0.1).")

    # Structural features
    p.add_argument(
        "--no_struct_features", action="store_true",
        help="Disable graph-structural features (node counts).",
    )

    # Debug: struct-only baseline
    p.add_argument(
        "--struct_only", action="store_true",
        help="Use struct-only baseline (no GNN). Should get ~100%% if data is correct.",
    )

    p.add_argument("--num_gpus", type=int, default=8)

    # Eval mode
    p.add_argument("--test_only", action="store_true")
    p.add_argument("--test_checkpoint", type=str, default=None)

    args = p.parse_args()
    args.use_struct_features = not args.no_struct_features

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        print(f"Requested {args.num_gpus} GPUs but only {available_gpus} available "
              f"– using {available_gpus}.")
        args.num_gpus = available_gpus

    print(f"roomsMO sphere classifier (from scratch) │ {args.num_gpus} GPUs")
    print(f"Data root: {ROOMSMO_ROOT}")
    print(f"Config: hidden_dim={args.hidden_dim}, layers={args.layers}, "
          f"pooling={args.pooling}")
    print(f"Training: epochs={args.epochs}, bs={args.batch_size}, "
          f"lr={args.lr}, patience={args.patience}")
    if args.struct_only:
        print("Mode: STRUCT-ONLY BASELINE (no GNN)")
    else:
        print(f"Backbone: {'FROZEN' if args.freeze_backbone else 'FINE-TUNING'} "
              f"(lr_scale={args.backbone_lr_scale})")
        print(f"Struct features: {not args.no_struct_features}")

    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))

    mp.spawn(
        train_worker,
        args=(args.num_gpus, cfg, args),
        nprocs=args.num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()