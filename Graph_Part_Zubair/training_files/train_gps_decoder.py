#!/usr/bin/env python3
"""
Stage 2: Train decoder on frozen pretrained GPS encoder.
Loads GPS_best.pt from Stage 1 (CIR prediction), freezes encoder,
trains decoder to reconstruct scene graph from z_xml.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python training/train_gps_decoder.py \
      --dataset rooms_update --epochs 100 --num_gpus 8 \
      --encoder_ckpt gps_results/GPS_best.pt

  # Single GPU
  python training/train_gps_decoder.py --encoder_ckpt gps_results/GPS_best.pt --num_gpus 1
"""

import os
import sys
import argparse
import time
import math
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

LOSS_KEYS = [
    "total", "centroid_obj", "centroid_txrx", "edge_dist",
    "edge_dir", "node_type", "frequency",
]


class EdgeConditionedConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__(aggr="add")
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        e = self.edge_proj(edge_attr)
        return self.msg_mlp(x_j + e)


class DirConv(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.conv_fwd = EdgeConditionedConv(hidden_dim, edge_dim)
        self.conv_bwd = EdgeConditionedConv(hidden_dim, edge_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr):
        h_fwd = self.conv_fwd(x, edge_index, edge_attr)
        h_bwd = self.conv_bwd(x, edge_index.flip(0), edge_attr)
        alpha = torch.sigmoid(self.alpha)
        return alpha * h_fwd + (1 - alpha) * h_bwd


class GPSEncoder(nn.Module):
    def __init__(self, x_dim, edge_dim, hidden_dim=256, num_layers=4,
                 num_heads=4, geo_dim=32, dropout=0.1, pooling="txrx"):
        super().__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.z_dim = hidden_dim * 3 if pooling == "txrx" else hidden_dim

        if geo_dim > 0:
            self.geo_proj = nn.Sequential(nn.Linear(128, geo_dim), nn.ReLU())
            node_in_dim = geo_dim + (x_dim - 128)
        else:
            self.geo_proj = None
            node_in_dim = x_dim - 128

        self.node_in = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            local_conv = DirConv(hidden_dim, hidden_dim)
            self.gps_layers.append(GPSConv(
                channels=hidden_dim, conv=local_conv, heads=num_heads,
                dropout=dropout, attn_type='multihead',
                attn_kwargs={'dropout': dropout}, norm='layer_norm',
            ))
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        if self.geo_proj is not None:
            x = torch.cat([self.geo_proj(x[:, :128]), x[:, 128:]], dim=1)
        else:
            x = x[:, 128:]

        h = self.node_in(x)
        e = self.edge_proj(edge_attr)

        for gps_layer in self.gps_layers:
            h = gps_layer(h, edge_index, batch, edge_attr=e)

        h = self.final_norm(h)

        if self.pooling == "mean":
            g = global_mean_pool(h, batch)
        else:
            nt = data.node_type
            obj = global_mean_pool(h[nt == 0], batch[nt == 0])
            tx  = global_mean_pool(h[nt == 1], batch[nt == 1])
            rx  = global_mean_pool(h[nt == 2], batch[nt == 2])
            g = torch.cat([obj, tx, rx], dim=1)

        return g, h, batch


class DecoderMPLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr="add")
        self.msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.upd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index):
        return self.upd(self.propagate(edge_index, x=x))

    def message(self, x_j):
        return self.msg(x_j)


class SceneDecoder(nn.Module):
    def __init__(self, z_dim, max_nodes, hidden_dim=256, num_layers=4):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim

        self.node_queries = nn.Parameter(torch.randn(max_nodes, hidden_dim) * 0.02)
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mp_layers = nn.ModuleList([DecoderMPLayer(hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.centroid_obj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3))
        self.centroid_txrx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3))

        self.node_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3))

        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1))

        self.edge_dist_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.edge_dir_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3))


class GPSDecoderModel(nn.Module):
    def __init__(self, encoder, decoder, max_nodes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_nodes = max_nodes

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, data):
        with torch.no_grad():
            z, h_enc, batch = self.encoder(data)

        num_nodes = torch.bincount(batch) if batch.max() > 0 \
            else torch.tensor([batch.numel()], device=batch.device)

        recon = self._decode(z, data, batch, num_nodes)
        return z, recon

    def _decode(self, z, data, batch, num_nodes):
        B = z.shape[0]
        device = z.device
        z_cond = self.decoder.z_proj(z)
        N = num_nodes[0].item()
        all_same = (num_nodes[0] == num_nodes).all() if B > 1 else True

        if all_same:
            q_raw = self.decoder.node_queries[:N].unsqueeze(0).expand(B, -1, -1)
            c = z_cond.unsqueeze(1).expand(-1, N, -1)
            h_dec = self.decoder.combine(torch.cat([q_raw, c], dim=2)).reshape(B * N, -1)
            q_skip = q_raw.reshape(B * N, -1)
        else:
            parts, q_parts = [], []
            for i in range(B):
                n = num_nodes[i].item()
                q = self.decoder.node_queries[:n]
                c_i = z_cond[i].unsqueeze(0).expand(n, -1)
                parts.append(self.decoder.combine(torch.cat([q, c_i], dim=1)))
                q_parts.append(q)
            h_dec = torch.cat(parts, dim=0)
            q_skip = torch.cat(q_parts, dim=0)

        for mp_layer, norm in zip(self.decoder.mp_layers, self.decoder.norms):
            h_dec = norm(h_dec + mp_layer(h_dec, data.edge_index))

        nt_input = torch.cat([h_dec, q_skip], dim=1)

        nt = data.node_type
        obj_mask = nt == 0
        all_centroids_obj = self.decoder.centroid_obj_head(h_dec)
        all_centroids_txrx = self.decoder.centroid_txrx_head(h_dec)
        centroids = torch.where(
            obj_mask.unsqueeze(1), all_centroids_obj, all_centroids_txrx
        )

        src, dst = data.edge_index
        edge_h = torch.cat([h_dec[src], h_dec[dst]], dim=1)
        pred_edge_dist = self.decoder.edge_dist_head(edge_h)
        pred_edge_dir = self.decoder.edge_dir_head(edge_h)

        return {
            "centroids":  centroids,
            "node_types": self.decoder.node_type_head(nt_input),
            "frequency":  self.decoder.freq_head(h_dec),
            "edge_dist":  pred_edge_dist,
            "edge_dir":   pred_edge_dir,
        }

    @torch.no_grad()
    def encode(self, data):
        z, _, _ = self.encoder(data)
        return z

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


class ReconLoss(nn.Module):
    def __init__(self, lam_centroid_obj=1.0, lam_centroid_txrx=2.0,
                 lam_edge_dist=0.5, lam_edge_dir=0.5,
                 lam_node_type=0.2, lam_freq=0.1):
        super().__init__()
        self.w = dict(centroid_obj=lam_centroid_obj, centroid_txrx=lam_centroid_txrx,
                      edge_dist=lam_edge_dist, edge_dir=lam_edge_dir,
                      node_type=lam_node_type, frequency=lam_freq)

    def forward(self, recon, data):
        x = data.x
        nt = data.node_type
        L = {}

        gt_centroids = x[:, 160:163]
        obj_mask = nt == 0
        txrx_mask = (nt == 1) | (nt == 2)

        if obj_mask.any():
            L["centroid_obj"] = F.mse_loss(recon["centroids"][obj_mask], gt_centroids[obj_mask])
        else:
            L["centroid_obj"] = torch.tensor(0.0, device=x.device)

        if txrx_mask.any():
            L["centroid_txrx"] = F.mse_loss(recon["centroids"][txrx_mask], gt_centroids[txrx_mask])
        else:
            L["centroid_txrx"] = torch.tensor(0.0, device=x.device)

        L["node_type"] = F.cross_entropy(recon["node_types"], data.node_type.long())
        L["frequency"] = F.mse_loss(recon["frequency"], x[:, 169:170])

        gt_dist = data.edge_attr[:, 0:1]
        L["edge_dist"] = F.mse_loss(recon["edge_dist"], gt_dist)

        gt_dir = data.edge_attr[:, 1:4]
        pred_dir_norm = F.normalize(recon["edge_dir"], dim=-1)
        gt_dir_norm = F.normalize(gt_dir, dim=-1)
        L["edge_dir"] = (1 - F.cosine_similarity(pred_dir_norm, gt_dir_norm, dim=-1)).mean()

        L["total"] = sum(self.w[k] * L[k] for k in self.w)
        return L


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12399"
    dist.init_process_group("nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def _avg_losses(raw, n_batches, device):
    out = {}
    for k, v in raw.items():
        t = torch.tensor([v / max(n_batches, 1)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        out[k] = t.item()
    return out


def train_epoch(model, loader, optim, loss_fn, device, rank, sampler, epoch, scaler):
    model.train()
    sampler.set_epoch(epoch)
    acc = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    bar = tqdm(loader, desc="Train", leave=False) if rank == 0 else loader

    for data in bar:
        data = data.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            z, recon = model(data)
            L = loss_fn(recon, data)

        scaler.scale(L["total"]).backward()
        scaler.unscale_(optim)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optim)
        scaler.update()

        for k in LOSS_KEYS:
            acc[k] += L[k].item()
        n += 1
        if rank == 0 and hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{L['total'].item():.5f}")

    return _avg_losses(acc, n, device)


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, rank):
    model.eval()
    acc = {k: 0.0 for k in LOSS_KEYS}
    n = 0
    for data in loader:
        data = data.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            z, recon = model(data)
            L = loss_fn(recon, data)
        for k in LOSS_KEYS:
            acc[k] += L[k].item()
        n += 1
    return _avg_losses(acc, n, device)


@torch.no_grad()
def extract_z(model, loader, device):
    model.eval()
    parts = []
    for data in tqdm(loader, desc="Extracting z_xml", leave=False):
        parts.append(model.encode(data.to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)


def plot_losses(train_h, val_h, save_dir):
    n_keys = len(LOSS_KEYS)
    cols = 4
    rows = math.ceil(n_keys / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for ax, key in zip(axes[:n_keys], LOSS_KEYS):
        ax.plot([h[key] for h in train_h], label="Train", lw=2)
        ax.plot([h[key] for h in val_h],   label="Val",   lw=2)
        ax.set_title(key.replace("_", " ").title())
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    for ax in axes[n_keys:]:
        ax.axis("off")
    plt.suptitle("Stage 2: Decoder Loss Curves (Frozen GPS Encoder)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "decoder_losses.png"), dpi=150)
    plt.close()


def plot_centroid_recon(model, loader, device, save_dir, n_samples=5):
    model.eval()
    data = next(iter(loader)).to(device)
    with torch.no_grad():
        z, recon = model(data)

    gt   = data.x[:, 160:163].detach().cpu().numpy()
    pred = recon["centroids"].detach().cpu().numpy()
    nt   = data.node_type.detach().cpu().numpy()
    batch = data.batch.detach().cpu().numpy() if hasattr(data, "batch") else np.zeros(len(gt), dtype=int)

    n_g = min(n_samples, int(batch.max()) + 1)
    fig, axes = plt.subplots(1, n_g, figsize=(5 * n_g, 5))
    if n_g == 1: axes = [axes]
    cmap = {0: "blue", 1: "red", 2: "green"}
    for idx, ax in enumerate(axes):
        m = batch == idx
        gc, pc, gt_t = gt[m], pred[m], nt[m]
        colors = [cmap.get(int(t), "gray") for t in gt_t]
        ax.scatter(gc[:, 0], gc[:, 1], c=colors, marker="o", s=40, alpha=0.7, label="GT")
        ax.scatter(pc[:, 0], pc[:, 1], c=colors, marker="x", s=40, alpha=0.7, label="Pred")
        for j in range(len(gc)):
            ax.plot([gc[j, 0], pc[j, 0]], [gc[j, 1], pc[j, 1]], "gray", alpha=0.3, lw=0.5)
        ax.set_title(f"Graph {idx} | Blue=Obj Red=TX Green=RX")
        ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.suptitle("Stage 2: Centroid Reconstruction (Frozen Encoder)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "centroid_recon.png"), dpi=150)
    plt.close()


def worker(rank, world_size, cfg, args, sample_info):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    ds_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(ds_path, "dataset.npy")
    z_pc_path = os.path.join(ds_path, "z_pc.npy")

    n_total = len(np.load(cfg["dataset"]["path"], allow_pickle=True)[0])
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val

    train_ds = SceneGraphDataset(cfg, list(range(n_train)), z_pc_path=z_pc_path)
    val_ds   = SceneGraphDataset(cfg, list(range(n_train, n_total)), z_pc_path=z_pc_path)

    tr_samp = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    va_samp = DistributedSampler(val_ds,   world_size, rank, shuffle=False)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size,
                           sampler=tr_samp, num_workers=0, pin_memory=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size,
                           sampler=va_samp, num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"Dataset: {args.dataset} | Train: {n_train}  Val: {n_val}")
        print(f"Per-GPU batch: {args.batch_size} | Effective: {args.batch_size * world_size}")

    enc_args = args.enc_args
    encoder = GPSEncoder(
        x_dim=sample_info["x_dim"], edge_dim=sample_info["edge_dim"],
        hidden_dim=enc_args["hidden_dim"], num_layers=enc_args["layers"],
        num_heads=enc_args["num_heads"], geo_dim=enc_args["geo_dim"],
        dropout=enc_args["dropout"], pooling=enc_args["pooling"],
    )

    ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
    enc_state = {k: v for k, v in ckpt["model_state_dict"].items()
                 if not k.startswith("head.")}
    encoder.load_state_dict(enc_state, strict=False)

    if rank == 0:
        print(f"Loaded pretrained encoder from {args.encoder_ckpt}")
        print(f"  Encoder z_dim: {encoder.z_dim}")

    decoder = SceneDecoder(
        z_dim=encoder.z_dim, max_nodes=sample_info["max_nodes"],
        hidden_dim=args.dec_hidden_dim, num_layers=args.dec_layers,
    )

    model = GPSDecoderModel(encoder, decoder, sample_info["max_nodes"]).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Trainable params (decoder): {trainable:,}")
        print(f"Frozen params (encoder):    {frozen:,}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)
    loss_fn = ReconLoss(
        lam_centroid_obj=args.lam_centroid_obj, lam_centroid_txrx=args.lam_centroid_txrx,
        lam_edge_dist=args.lam_edge_dist, lam_edge_dir=args.lam_edge_dir,
        lam_node_type=args.lam_node_type, lam_freq=args.lam_freq,
    )
    scaler = torch.amp.GradScaler('cuda')

    save_dir = os.path.join(ROOT, args.save_dir)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    train_h, val_h = [], []
    best_val = float("inf")
    patience = 0
    t0 = time.time()

    if rank == 0:
        print(f"Training decoder for {args.epochs} epochs (patience={args.patience})")
        print(f"Decoder: {args.dec_layers} layers, hidden={args.dec_hidden_dim}")

    for epoch in range(args.epochs):
        tr_L = train_epoch(model, tr_loader, optim, loss_fn, device, rank, tr_samp, epoch, scaler)
        va_L = val_epoch(model, va_loader, loss_fn, device, rank)
        scheduler.step()

        if rank == 0:
            train_h.append(tr_L)
            val_h.append(va_L)
            is_best = va_L["total"] < best_val

            if is_best:
                best_val = va_L["total"]
                patience = 0
                torch.save({
                    "epoch": epoch,
                    "decoder_state_dict": model.module.decoder.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "train_history": train_h,
                    "val_history": val_h,
                    "best_val_loss": best_val,
                    "args": vars(args),
                    "enc_args": enc_args,
                }, os.path.join(save_dir, "decoder_best.pt"))
            else:
                patience += 1

            lr = optim.param_groups[0]["lr"]
            eta = (time.time() - t0) / (epoch + 1) * (args.epochs - epoch - 1)
            print(
                f"Ep {epoch+1:03d}/{args.epochs} | "
                f"Tr {tr_L['total']:.5f} | Va {va_L['total']:.5f} | "
                f"Cobj:{va_L['centroid_obj']:.4f} Ctx:{va_L['centroid_txrx']:.4f} "
                f"ED:{va_L['edge_dist']:.4f} Edir:{va_L['edge_dir']:.4f} "
                f"NT:{va_L['node_type']:.3f} F:{va_L['frequency']:.4f} | "
                f"LR:{lr:.1e} | ETA:{eta/60:.1f}m | "
                f"{'*BEST' if is_best else f'p={patience}/{args.patience}'}"
            )

        flag = torch.tensor([1 if (rank == 0 and patience >= args.patience) else 0], device=device)
        dist.broadcast(flag, src=0)
        if flag.item():
            if rank == 0:
                print(f"Early stop at epoch {epoch+1}")
            break

    if rank == 0:
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed/60:.1f} min | Best val: {best_val:.6f}")

        torch.save({
            "decoder_state_dict": model.module.decoder.state_dict(),
            "train_history": train_h, "val_history": val_h, "args": vars(args),
        }, os.path.join(save_dir, "decoder_final.pt"))

        plot_losses(train_h, val_h, save_dir)

        best_decoder = SceneDecoder(
            z_dim=encoder.z_dim, max_nodes=sample_info["max_nodes"],
            hidden_dim=args.dec_hidden_dim, num_layers=args.dec_layers,
        )
        best_ckpt = torch.load(os.path.join(save_dir, "decoder_best.pt"),
                                map_location=device, weights_only=False)
        best_decoder.load_state_dict(best_ckpt["decoder_state_dict"])
        best_model = GPSDecoderModel(encoder, best_decoder, sample_info["max_nodes"]).to(device)

        vis_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
        print("Generating visualizations...")
        plot_centroid_recon(best_model, vis_loader, device, save_dir)

        full_ds = SceneGraphDataset(cfg, list(range(n_total)), z_pc_path=z_pc_path)
        full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)
        print("Extracting z_xml embeddings...")
        z_new = extract_z(best_model, full_loader, device)
        z_path = os.path.join(save_dir, "z_xml.npy")
        np.save(z_path, z_new)

        print(f"\nSaved z_xml -> {z_path}  shape: {z_new.shape}")
        print(f"Dataset     : {args.dataset} ({n_total} samples)")
        print(f"Encoder     : frozen from {args.encoder_ckpt}")
        print(f"Decoder     : {args.dec_layers} layers, hidden={args.dec_hidden_dim}")
        print(f"z_xml dim   : {z_new.shape[1]}")
        print(f"Train time  : {elapsed/60:.1f} min")
        print(f"Best val    : {best_val:.6f}")
        best_h = min(val_h, key=lambda h: h["total"])
        print(f"Per-component val losses (best epoch):")
        for k in LOSS_KEYS[1:]:
            print(f"  {k:14s}: {best_h[k]:.6f}")
        print(f"\nOutputs -> {save_dir}/")
        print(f"  decoder_best.pt, decoder_final.pt")
        print(f"  decoder_losses.png, centroid_recon.png")
        print(f"  z_xml.npy")

    cleanup_ddp()


def main():
    p = argparse.ArgumentParser(description="Stage 2: Train decoder on frozen GPS encoder")

    p.add_argument("--dataset", type=str, default="rooms_update", choices=list(DATASETS.keys()))
    p.add_argument("--encoder_ckpt", type=str, required=True,
                   help="Path to GPS_best.pt from Stage 1")

    p.add_argument("--dec_hidden_dim", type=int, default=256)
    p.add_argument("--dec_layers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num_gpus", type=int, default=8)

    p.add_argument("--lam_centroid_obj", type=float, default=1.0)
    p.add_argument("--lam_centroid_txrx", type=float, default=2.0)
    p.add_argument("--lam_edge_dist", type=float, default=0.5)
    p.add_argument("--lam_edge_dir", type=float, default=0.5)
    p.add_argument("--lam_node_type", type=float, default=0.2)
    p.add_argument("--lam_freq", type=float, default=0.1)

    p.add_argument("--save_dir", type=str, default="gps_decoder_results")

    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available."); sys.exit(1)

    available = torch.cuda.device_count()
    args.num_gpus = min(args.num_gpus, max(available, 1))

    ckpt = torch.load(args.encoder_ckpt, map_location="cpu", weights_only=False)
    enc_args = ckpt.get("args", {})
    if not enc_args:
        enc_args = {"hidden_dim": 256, "layers": 4, "num_heads": 4,
                    "geo_dim": 32, "dropout": 0.1, "pooling": "txrx"}
        print(f"[WARN] No args in checkpoint, using defaults: {enc_args}")
    args.enc_args = enc_args

    print(f"Stage 2: Decoder Training (Frozen GPS Encoder)")
    print(f"Encoder checkpoint: {args.encoder_ckpt}")
    print(f"GPUs: {args.num_gpus}")

    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    ds_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(ds_path, "dataset.npy")
    z_pc_path = os.path.join(ds_path, "z_pc.npy")

    tmp = SceneGraphDataset(cfg, [0], z_pc_path=z_pc_path)
    s = tmp[0]
    sample_info = {
        "x_dim": s.x.shape[1],
        "edge_dim": s.edge_attr.shape[1],
        "max_nodes": s.x.shape[0],
    }
    print(f"Graph: {sample_info['max_nodes']} nodes, x_dim={sample_info['x_dim']}, "
          f"edge_dim={sample_info['edge_dim']}")

    mp.spawn(worker, args=(args.num_gpus, cfg, args, sample_info),
             nprocs=args.num_gpus, join=True)


if __name__ == "__main__":
    main()