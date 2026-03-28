#!/usr/bin/env python3
"""
GPS Graph Autoencoder — Standalone z_xml Encoder-Decoder (Multi-GPU DDP)
=========================================================================
Trains a GPS-based (Graph Transformer) autoencoder on scene graphs.
Reconstructs node features, edge features, AND predicts adjacency
(edge existence + edge weights).

Architecture:
  Encoder: GPS layers (NNConv local MPNN + Multi-Head Global Attention)
           with Dir-GNN-style directional aggregation.
           Produces z_xml via type-aware pooling (obj|tx|rx).

  Decoder: Learnable node queries conditioned on z_xml,
           refined with message-passing, then:
           - Node feature heads (centroid, material, node_type, frequency)
           - Edge feature head (distance + direction)
           - Adjacency head (edge existence + edge weight prediction)

Input graph (from SceneGraphDataset):
  data.x          : (N, 170)  — [z_pc(128) | material(32) | centroid(3) | unused(3) | type(3) | freq(1)]
  data.edge_index : (2, E)    — directed edges
  data.edge_attr  : (E, 4)    — [distance(1) | direction(3)]
  data.node_type  : (N,)      — 0=object, 1=TX, 2=RX
  data.y          : (107,)    — CIR (not used in AE, but kept for compatibility)

Output:
  z_xml.npy       : (N_samples, z_dim) graph-level embeddings
  Reconstruction metrics + adjacency prediction metrics

Usage:
  # 8-GPU DDP (default)
  python train_gps_autoencoder.py --dataset rooms_update --epochs 200
OR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python training/train_gps_autoencoder.py --dataset rooms_update --epochs 10 --num_gpus 8

  # Single GPU
  python train_gps_autoencoder.py --dataset rooms_update --epochs 200 --num_gpus 1

  # Compare with existing embeddings
  python train_gps_autoencoder.py --dataset rooms_update --epochs 200 \\
      --compare_with ae_results/z_xml_ae.npy
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
from torch_geometric.nn import (
    GPSConv, global_mean_pool, global_add_pool, MessagePassing,
)
from torch_geometric.utils import to_dense_adj
from scipy.spatial.distance import cdist
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

LOSS_KEYS = [
    "total", "centroid", "material", "edge_attr",
    "node_type", "frequency", "adj_exist", "adj_weight",
]


# DISTRIBUTED HELPERS

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12398"
    dist.init_process_group("nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()



# NOTE: No positional encoding preprocessing needed.
# GPS already captures structure via its local MPNN (DirConv) branch.
# Node features contain 3D coordinates + node types + spatial edges
# which implicitly encode positional information.
# This allows using SceneGraphDataset directly .



# BUILDING BLOCKS


class EdgeMLP(nn.Module):
    """MLP that maps edge features to a weight matrix for NNConv."""
    def __init__(self, edge_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, edge_attr):
        return self.net(edge_attr).view(-1, self.hidden_dim, self.hidden_dim)


class EdgeConditionedConv(MessagePassing):
    """
    Memory-efficient edge-conditioned message passing.
    Instead of NNConv (which generates a hidden×hidden matrix per edge = OOM),
    this projects edge features to hidden_dim and adds them to messages.
    Memory: O(E × hidden_dim) instead of O(E × hidden_dim²).
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
        # Add projected edge features to neighbor features, then transform
        e = self.edge_proj(edge_attr)
        return self.msg_mlp(x_j + e)


class DirConv(nn.Module):
    """
    Dir-GNN style directional message passing.
    Runs two parallel passes (in-edges and out-edges) and mixes with learnable alpha.
    Uses EdgeConditionedConv instead of NNConv to avoid OOM.
    """
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.conv_fwd = EdgeConditionedConv(hidden_dim, edge_dim)
        self.conv_bwd = EdgeConditionedConv(hidden_dim, edge_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index, edge_attr):
        # Forward: messages along original directed edges
        h_fwd = self.conv_fwd(x, edge_index, edge_attr)

        # Backward: messages along reversed edges
        edge_index_rev = edge_index.flip(0)
        h_bwd = self.conv_bwd(x, edge_index_rev, edge_attr)

        # Learnable mix
        alpha = torch.sigmoid(self.alpha)
        return alpha * h_fwd + (1 - alpha) * h_bwd


# GPS ENCODER

class GPSEncoder(nn.Module):
    """
    GPS (General, Powerful, Scalable) Graph Transformer Encoder.

    Each GPS layer combines:
      - Local MPNN: DirNNConv (directional NNConv with edge features)
      - Global: Multi-head self-attention over all nodes

    Produces z_xml via type-aware pooling: [obj_pool | tx_pool | rx_pool]
    """

    def __init__(
        self,
        x_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        geo_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = hidden_dim * 3  # obj|tx|rx pooling

        # ── Input projections 
        # Geometry compression (first 128 dims = Point-MAE z_pc)
        if geo_dim > 0:
            self.geo_proj = nn.Sequential(nn.Linear(128, geo_dim), nn.ReLU())
            node_in_dim = geo_dim + (x_dim - 128)
        else:
            self.geo_proj = None
            node_in_dim = x_dim - 128

        # Node input projection → hidden_dim
        # No PE needed: node features already contain 3D coordinates,
        # node types, and edges carry spatial distances/directions.
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge input projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── GPS Layers 
        # Each GPSConv wraps a local MPNN conv + global attention
        self.gps_layers = nn.ModuleList()

        for _ in range(num_layers):
            # Local MPNN: DirConv for directional edge-aware message passing
            local_conv = DirConv(hidden_dim, hidden_dim)

            # GPS layer: local + global attention
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

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        # ── Geometry compression 
        if self.geo_proj is not None:
            x = torch.cat([self.geo_proj(x[:, :128]), x[:, 128:]], dim=1)
        else:
            x = x[:, 128:]

        # ── Node projection (no PE needed) 
        h = self.node_proj(x)

        # ── Edge projection 
        e = self.edge_proj(edge_attr)

        # ── GPS message passing 
        for gps_layer in self.gps_layers:
            h = gps_layer(h, edge_index, batch, edge_attr=e)

        h = self.final_norm(h)

        # ── Type-aware pooling → z_xml 
        nt = data.node_type
        obj_pool = global_mean_pool(h[nt == 0], batch[nt == 0])
        tx_pool  = global_mean_pool(h[nt == 1], batch[nt == 1])
        rx_pool  = global_mean_pool(h[nt == 2], batch[nt == 2])
        z = torch.cat([obj_pool, tx_pool, rx_pool], dim=1)  # (B, hidden_dim*3)

        return z, h, batch



# DECODER (Node Features + Edge Features + Adjacency)

class DecoderMPLayer(MessagePassing):
    """Simple message-passing layer for decoder refinement."""
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


class GPSDecoder(nn.Module):
    """
    Decoder that reconstructs node features, edge features, AND adjacency.

    From z_xml + graph topology:
      1. Initialize node states from learnable queries + z conditioning
      2. Refine with message passing over original edges
      3. Predict: centroids, materials, node_types, frequency, edge_attrs
      4. Predict adjacency: for ALL node pairs (i,j), predict
         edge existence (binary) and edge weight (continuous)
    """

    def __init__(
        self,
        z_dim: int,
        max_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim

        # Learnable node prototypes
        self.node_queries = nn.Parameter(
            torch.randn(max_nodes, hidden_dim) * 0.02
        )

        # z → node conditioning
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Combine query + conditioning
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing refinement
        self.mp_layers = nn.ModuleList(
            [DecoderMPLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # ── Node feature reconstruction heads 
        self.centroid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.material_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 32),
        )
        self.node_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # ── Edge attribute reconstruction head 
        # Predicts edge_attr (4-dim) for existing edges
        self.edge_attr_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        # ── Adjacency prediction head (MLP Edge Decoder) 
        # For EVERY (i,j) pair: predict existence + weight
        # Input: [h_i || h_j || h_i * h_j]  (3 * hidden_dim)
        self.adj_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [logit_exists, weight]
        )

    def forward(self, z, edge_index, edge_attr, batch, num_nodes_per_graph):
        """
        Args:
            z:                   (B, z_dim) graph-level embeddings
            edge_index:          (2, E) directed edges
            edge_attr:           (E, 4) edge features [distance, dx, dy, dz]
            batch:               (N,) batch assignment per node
            num_nodes_per_graph: (B,) number of nodes in each graph
        """
        B = z.shape[0]
        device = z.device
        z_cond = self.z_proj(z)

        # ── Initialize node states per graph 
        parts = []
        for i in range(B):
            n = num_nodes_per_graph[i].item()
            q = self.node_queries[:n]
            c = z_cond[i].unsqueeze(0).expand(n, -1)
            parts.append(self.combine(torch.cat([q, c], dim=1)))
        h = torch.cat(parts, dim=0)

        # ── Message passing refinement 
        for mp_layer, norm in zip(self.mp_layers, self.norms):
            h = norm(h + mp_layer(h, edge_index))

        # ── Node feature predictions 
        src, dst = edge_index

        # ── Edge attribute prediction (for existing edges) 
        edge_h = torch.cat([h[src], h[dst]], dim=1)
        pred_edge_attr = self.edge_attr_head(edge_h)

        # ── Adjacency prediction (for ALL node pairs) 
        adj_logits_list = []
        adj_weights_list = []
        adj_targets_exist_list = []
        adj_targets_weight_list = []

        node_offset = 0
        for i in range(B):
            n = num_nodes_per_graph[i].item()
            h_graph = h[node_offset : node_offset + n]

            # All ordered pairs (i,j) where i != j → directed
            idx_i = torch.arange(n, device=device).repeat_interleave(n - 1)
            idx_j_parts = []
            for j in range(n):
                idx_j_parts.append(
                    torch.cat([
                        torch.arange(j, device=device),
                        torch.arange(j + 1, n, device=device),
                    ])
                )
            idx_j = torch.cat(idx_j_parts)

            h_i = h_graph[idx_i]
            h_j = h_graph[idx_j]
            pair_feat = torch.cat([h_i, h_j, h_i * h_j], dim=1)
            pred = self.adj_head(pair_feat)  # (n*(n-1), 2)
            adj_logits_list.append(pred[:, 0])
            adj_weights_list.append(pred[:, 1])

            # Build ground-truth from edge_index + edge_attr
            graph_mask = (
                (edge_index[0] >= node_offset) &
                (edge_index[0] < node_offset + n)
            )
            g_edges = edge_index[:, graph_mask] - node_offset
            g_eattr = edge_attr[graph_mask]  # (E_graph, 4)

            gt_exist = torch.zeros(n, n, device=device)
            gt_weight = torch.zeros(n, n, device=device)
            if g_edges.size(1) > 0:
                gt_exist[g_edges[0], g_edges[1]] = 1.0
                gt_weight[g_edges[0], g_edges[1]] = g_eattr[:, 0]  # distance

            # Extract ordered pairs excluding diagonal
            exist_flat = []
            weight_flat = []
            for ii in range(n):
                for jj in range(n):
                    if ii != jj:
                        exist_flat.append(gt_exist[ii, jj])
                        weight_flat.append(gt_weight[ii, jj])

            adj_targets_exist_list.append(torch.stack(exist_flat))
            adj_targets_weight_list.append(torch.stack(weight_flat))

            node_offset += n

        return {
            "centroids":  self.centroid_head(h),
            "materials":  self.material_head(h),
            "node_types": self.node_type_head(h),
            "frequency":  self.freq_head(h),
            "edge_attrs": pred_edge_attr,
            "adj_logits":        torch.cat(adj_logits_list),
            "adj_weights":       torch.cat(adj_weights_list),
            "adj_target_exist":  torch.cat(adj_targets_exist_list),
            "adj_target_weight": torch.cat(adj_targets_weight_list),
        }



# FULL GPS AUTOENCODER

class GPSAutoencoder(nn.Module):
    """
    GPS Encoder + Decoder with adjacency prediction.
    """

    def __init__(
        self,
        x_dim: int,
        edge_dim: int,
        max_nodes: int,
        hidden_dim: int = 256,
        enc_layers: int = 4,
        dec_layers: int = 4,
        num_heads: int = 4,
        geo_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GPSEncoder(
            x_dim=x_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
            num_layers=enc_layers, num_heads=num_heads,
            geo_dim=geo_dim, dropout=dropout,
        )
        self.decoder = GPSDecoder(
            z_dim=self.encoder.z_dim, max_nodes=max_nodes,
            hidden_dim=hidden_dim, num_layers=dec_layers,
        )
        self.z_dim = self.encoder.z_dim

    def forward(self, data):
        z, h_enc, batch = self.encoder(data)
        num_nodes = torch.bincount(batch) if batch.max() > 0 \
            else torch.tensor([batch.numel()], device=batch.device)

        # Pass edge_attr to decoder for adjacency target construction
        recon = self._decode_with_adj(z, data, batch, num_nodes, h_enc)
        return z, recon

    def _decode_with_adj(self, z, data, batch, num_nodes, h_enc):
        """
        Run decoder and build adjacency targets from ground truth edges.
        FULLY VECTORIZED — no Python for-loops for pair/target construction.
        """
        B = z.shape[0]
        z_cond = self.decoder.z_proj(z)
        device = z.device

        # ── Initialize node states 
        # Since all graphs have same number of nodes (57), we can vectorize
        # For variable-size graphs, fall back to loop
        all_same = (num_nodes[0] == num_nodes).all() if B > 1 else True
        N = num_nodes[0].item()

        if all_same:
            # Fast path: all graphs have N nodes
            q = self.decoder.node_queries[:N].unsqueeze(0).expand(B, -1, -1)  # (B, N, H)
            c = z_cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, H)
            h_init = self.decoder.combine(torch.cat([q, c], dim=2))  # (B, N, H)
            h_dec = h_init.reshape(B * N, -1)  # (B*N, H)
        else:
            parts = []
            for i in range(B):
                n = num_nodes[i].item()
                q = self.decoder.node_queries[:n]
                c = z_cond[i].unsqueeze(0).expand(n, -1)
                parts.append(self.decoder.combine(torch.cat([q, c], dim=1)))
            h_dec = torch.cat(parts, dim=0)

        # ── Message passing refinement 
        for mp_layer, norm in zip(self.decoder.mp_layers, self.decoder.norms):
            h_dec = norm(h_dec + mp_layer(h_dec, data.edge_index))

        # ── Node feature predictions 
        src, dst = data.edge_index
        edge_h = torch.cat([h_dec[src], h_dec[dst]], dim=1)
        pred_edge_attr = self.decoder.edge_attr_head(edge_h)

        # ── Adjacency prediction (VECTORIZED) 
        # Build all off-diagonal pair indices for all graphs at once
        # For N=57: each graph has N*(N-1) = 3192 pairs
        # Precompute pair indices once (same for all graphs if same size)

        if all_same:
            # Build local pair indices (0..N-1) once, then offset per graph
            # off-diagonal mask
            arange = torch.arange(N, device=device)
            grid_i, grid_j = torch.meshgrid(arange, arange, indexing='ij')
            mask = grid_i != grid_j  # (N, N) bool
            local_i = grid_i[mask]  # (N*(N-1),)
            local_j = grid_j[mask]  # (N*(N-1),)
            n_pairs = local_i.size(0)

            # Global indices: offset by graph position in batch
            offsets = torch.arange(B, device=device) * N  # (B,)
            global_i = (local_i.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)  # (B*n_pairs,)
            global_j = (local_j.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

            # Gather node embeddings for all pairs at once
            h_i = h_dec[global_i]  # (B*n_pairs, H)
            h_j = h_dec[global_j]
            pair_feat = torch.cat([h_i, h_j, h_i * h_j], dim=1)  # (B*n_pairs, 3H)
            pred = self.decoder.adj_head(pair_feat)  # (B*n_pairs, 2)
            all_adj_logits = pred[:, 0]
            all_adj_weights = pred[:, 1]

            # Build ground-truth adjacency using to_dense_adj (vectorized)
            gt_adj_dense = to_dense_adj(data.edge_index, batch, max_num_nodes=N)  # (B, N, N)

            # Build weight adjacency: use edge_attr[:, 0] (distance)
            gt_weight_dense = torch.zeros(B, N, N, device=device)
            edge_batch = batch[data.edge_index[0]]  # which graph each edge belongs to
            local_src = data.edge_index[0] - edge_batch * N
            local_dst = data.edge_index[1] - edge_batch * N
            gt_weight_dense[edge_batch, local_src, local_dst] = data.edge_attr[:, 0]

            # Extract off-diagonal entries (same mask for all graphs)
            all_adj_target_exist = gt_adj_dense[:, mask].reshape(-1)    # (B*n_pairs,)
            all_adj_target_weight = gt_weight_dense[:, mask].reshape(-1)
        else:
            # Variable-size fallback (slower but correct)
            adj_logits_list = []
            adj_weights_list = []
            adj_targets_exist_list = []
            adj_targets_weight_list = []

            node_offset = 0
            for i in range(B):
                n = num_nodes[i].item()
                h_graph = h_dec[node_offset : node_offset + n]

                arange = torch.arange(n, device=device)
                grid_i, grid_j = torch.meshgrid(arange, arange, indexing='ij')
                off_diag = grid_i != grid_j
                idx_i = grid_i[off_diag]
                idx_j = grid_j[off_diag]

                h_i = h_graph[idx_i]
                h_j = h_graph[idx_j]
                pair_feat = torch.cat([h_i, h_j, h_i * h_j], dim=1)
                pred = self.decoder.adj_head(pair_feat)
                adj_logits_list.append(pred[:, 0])
                adj_weights_list.append(pred[:, 1])

                graph_edge_mask = (
                    (data.edge_index[0] >= node_offset) &
                    (data.edge_index[0] < node_offset + n)
                )
                g_edges = data.edge_index[:, graph_edge_mask] - node_offset
                g_eattr = data.edge_attr[graph_edge_mask]

                gt_exist = torch.zeros(n, n, device=device)
                gt_weight = torch.zeros(n, n, device=device)
                if g_edges.size(1) > 0:
                    gt_exist[g_edges[0], g_edges[1]] = 1.0
                    gt_weight[g_edges[0], g_edges[1]] = g_eattr[:, 0]

                adj_targets_exist_list.append(gt_exist[off_diag])
                adj_targets_weight_list.append(gt_weight[off_diag])
                node_offset += n

            all_adj_logits = torch.cat(adj_logits_list)
            all_adj_weights = torch.cat(adj_weights_list)
            all_adj_target_exist = torch.cat(adj_targets_exist_list)
            all_adj_target_weight = torch.cat(adj_targets_weight_list)

        return {
            "centroids":         self.decoder.centroid_head(h_dec),
            "materials":         self.decoder.material_head(h_dec),
            "node_types":        self.decoder.node_type_head(h_dec),
            "frequency":         self.decoder.freq_head(h_dec),
            "edge_attrs":        pred_edge_attr,
            "adj_logits":        all_adj_logits,
            "adj_weights":       all_adj_weights,
            "adj_target_exist":  all_adj_target_exist,
            "adj_target_weight": all_adj_target_weight,
        }

    @torch.no_grad()
    def encode(self, data):
        z, _, _ = self.encoder(data)
        return z



# LOSS FUNCTION

class GPSAELoss(nn.Module):
    """
    Combined loss for node features, edge features, and adjacency.
    """

    def __init__(
        self,
        lam_centroid=1.0,
        lam_material=0.5,
        lam_edge_attr=0.5,
        lam_node_type=0.2,
        lam_freq=0.1,
        lam_adj_exist=1.0,
        lam_adj_weight=0.5,
    ):
        super().__init__()
        self.w = dict(
            centroid=lam_centroid,
            material=lam_material,
            edge_attr=lam_edge_attr,
            node_type=lam_node_type,
            frequency=lam_freq,
            adj_exist=lam_adj_exist,
            adj_weight=lam_adj_weight,
        )

    def forward(self, recon, data):
        x = data.x
        L = {}

        # ── Node feature losses 
        L["centroid"]  = F.mse_loss(recon["centroids"],  x[:, 160:163])
        L["material"]  = F.mse_loss(recon["materials"],  x[:, 128:160])
        L["node_type"] = F.cross_entropy(recon["node_types"], data.node_type.long())
        L["frequency"] = F.mse_loss(recon["frequency"],  x[:, 169:170])

        # ── Edge attribute loss 
        L["edge_attr"] = F.mse_loss(recon["edge_attrs"], data.edge_attr)

        # ── Adjacency existence loss (BCE with logits) 
        # Use pos_weight to handle class imbalance
        # (fully connected = many edges, but still some non-edges)
        n_pos = recon["adj_target_exist"].sum().clamp(min=1)
        n_neg = (recon["adj_target_exist"].numel() - n_pos).clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=10.0)

        L["adj_exist"] = F.binary_cross_entropy_with_logits(
            recon["adj_logits"],
            recon["adj_target_exist"],
            pos_weight=pos_weight.expand_as(recon["adj_logits"]),
        )

        # ── Adjacency weight loss (only for existing edges) ────
        exist_mask = recon["adj_target_exist"] > 0.5
        if exist_mask.any():
            L["adj_weight"] = F.mse_loss(
                recon["adj_weights"][exist_mask],
                recon["adj_target_weight"][exist_mask],
            )
        else:
            L["adj_weight"] = torch.tensor(0.0, device=x.device)

        # ── Total weighted loss 
        L["total"] = sum(self.w[k] * L[k] for k in self.w)
        return L


# DDP TRAINING LOOP

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

        # Mixed precision forward + loss
        with torch.amp.autocast("cuda"):
            z, recon = model(data)
            L = loss_fn(recon, data)

        # Scaled backward
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


# EXTRACTION & ANALYSIS (rank 0 only)

@torch.no_grad()
def extract_z(model, loader, device):
    model.eval()
    parts = []
    for data in tqdm(loader, desc="Extracting z_xml", leave=False):
        parts.append(model.encode(data.to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)


def compare_embeddings(z_new, z_old, save_dir, label_old="z_xml_old"):
    """CKA, mutual k-NN, distance correlation."""
    print("\n" + "=" * 65)
    print(f"EMBEDDING COMPARISON: new z_xml  vs  {label_old}")
    print("=" * 65)

    n = min(len(z_new), len(z_old))
    z1, z2 = z_new[:n], z_old[:n]

    def linear_cka(X, Y):
        X, Y = X - X.mean(0), Y - Y.mean(0)
        num = np.linalg.norm(X.T @ Y, "fro") ** 2
        den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
        return num / (den + 1e-10)

    cka = linear_cka(z1, z2)
    print(f"  Linear CKA:                   {cka:.4f}")

    k, n_sub = 10, min(2000, n)
    idx = np.random.choice(n, n_sub, replace=False)
    s1, s2 = z1[idx], z2[idx]
    d1, d2 = cdist(s1, s1), cdist(s2, s2)
    knn1 = np.argsort(d1, axis=1)[:, 1:k+1]
    knn2 = np.argsort(d2, axis=1)[:, 1:k+1]
    overlaps = [len(set(knn1[i]) & set(knn2[i])) / k for i in range(n_sub)]
    mean_ov = float(np.mean(overlaps))
    print(f"  Mutual {k}-NN overlap:            {mean_ov:.4f}")

    triu = np.triu_indices(n_sub, k=1)
    corr = float(np.corrcoef(d1[triu], d2[triu])[0, 1])
    print(f"  Pairwise distance correlation: {corr:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    si = np.random.choice(len(d1[triu]), min(5000, len(d1[triu])), replace=False)
    axes[0].scatter(d1[triu][si], d2[triu][si], alpha=0.1, s=1, c="steelblue")
    axes[0].set_xlabel("z_xml dist"); axes[0].set_ylabel(f"{label_old} dist")
    axes[0].set_title(f"Dist Corr: {corr:.3f}")
    axes[1].hist(overlaps, bins=20, edgecolor="black", color="coral", alpha=0.8)
    axes[1].axvline(mean_ov, color="darkred", ls="--", label=f"Mean={mean_ov:.3f}")
    axes[1].set_title(f"Mutual {k}-NN Overlap"); axes[1].legend()
    axes[2].bar(["CKA", "kNN", "Dist Corr"], [cka, mean_ov, corr],
                color=["steelblue", "coral", "seagreen"], edgecolor="black")
    axes[2].set_ylim(0, 1); axes[2].set_title("Summary")
    plt.suptitle(f"GPS z_xml vs {label_old}", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "z_xml_comparison.png"), dpi=150); plt.close()
    return {"cka": cka, "knn_overlap": mean_ov, "dist_corr": corr}


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
    plt.suptitle("GPS Autoencoder Loss Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gps_ae_losses.png"), dpi=150); plt.close()


def plot_adj_quality(model, loader, device, save_dir, n_graphs=4):
    """Visualise adjacency prediction quality."""
    model.eval()
    data = next(iter(loader)).to(device)
    with torch.no_grad():
        z, recon = model(data)

    batch = data.batch.detach().cpu()
    edge_index = data.edge_index.detach().cpu()

    fig, axes = plt.subplots(n_graphs, 2, figsize=(10, 4 * n_graphs))
    if n_graphs == 1:
        axes = axes.reshape(1, -1)

    node_offset = 0
    num_nodes = torch.bincount(batch)

    for g in range(min(n_graphs, num_nodes.size(0))):
        n = num_nodes[g].item()

        # Ground truth adjacency
        mask = (edge_index[0] >= node_offset) & (edge_index[0] < node_offset + n)
        g_edges = edge_index[:, mask] - node_offset
        gt_adj = torch.zeros(n, n)
        if g_edges.size(1) > 0:
            gt_adj[g_edges[0], g_edges[1]] = 1.0

        axes[g, 0].imshow(gt_adj.numpy(), cmap="Blues", vmin=0, vmax=1)
        axes[g, 0].set_title(f"Graph {g} — GT Adjacency")

        # Predicted adjacency (reconstruct from flat logits)
        n_pairs = n * (n - 1)
        start_pair = sum(num_nodes[gi].item() * (num_nodes[gi].item() - 1) for gi in range(g))
        logits = recon["adj_logits"][start_pair : start_pair + n_pairs].detach().cpu()
        probs = torch.sigmoid(logits)

        pred_adj = torch.zeros(n, n)
        pair_idx = 0
        for ii in range(n):
            for jj in range(n):
                if ii != jj:
                    pred_adj[ii, jj] = probs[pair_idx]
                    pair_idx += 1

        axes[g, 1].imshow(pred_adj.numpy(), cmap="Blues", vmin=0, vmax=1)
        axes[g, 1].set_title(f"Graph {g} — Predicted (sigmoid)")

        node_offset += n

    plt.suptitle("Adjacency Prediction Quality", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "adj_prediction.png"), dpi=150); plt.close()


def plot_centroid_recon(model, loader, device, save_dir, n_samples=5):
    """Visualise centroid reconstruction."""
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
            ax.plot([gc[j,0], pc[j,0]], [gc[j,1], pc[j,1]], "gray", alpha=0.3, lw=0.5)
        ax.set_title(f"Graph {idx}\nBlue=Obj Red=TX Green=RX")
        ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    plt.suptitle("Centroid Reconstruction", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "centroid_recon.png"), dpi=150); plt.close()



# DDP WORKER

def worker(rank, world_size, cfg, args, sample_info):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # ── Dataset  
    ds_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(ds_path, "dataset.npy")
    z_pc_path = os.path.join(ds_path, "z_pc.npy")

    n_total = len(np.load(cfg["dataset"]["path"], allow_pickle=True)[0])
    n_val   = int(n_total * 0.1)
    n_train = n_total - n_val

    train_ds = SceneGraphDataset(cfg, list(range(n_train)),          z_pc_path=z_pc_path)
    val_ds   = SceneGraphDataset(cfg, list(range(n_train, n_total)), z_pc_path=z_pc_path)

    tr_samp = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    va_samp = DistributedSampler(val_ds,   world_size, rank, shuffle=False)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size,
                           sampler=tr_samp, num_workers=0, pin_memory=True)
    va_loader = DataLoader(val_ds,   batch_size=args.batch_size,
                           sampler=va_samp, num_workers=0, pin_memory=True)

    if rank == 0:
        print(f"\n  Dataset: {args.dataset} | Train: {n_train}  Val: {n_val}")
        print(f"  Per-GPU batch: {args.batch_size} | Effective: {args.batch_size * world_size}")

    # ── Model 
    model = GPSAutoencoder(
        x_dim=sample_info["x_dim"],
        edge_dim=sample_info["edge_dim"],
        max_nodes=sample_info["max_nodes"],
        hidden_dim=args.hidden_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        num_heads=args.num_heads,
        geo_dim=args.geo_dim,
        dropout=args.dropout,
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_p:,} | z_dim: {model.module.z_dim}")
        print(f"  Encoder: {args.enc_layers} GPS layers, {args.num_heads} heads")

    # ── Optimiser 
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs, eta_min=1e-6
    )
    loss_fn = GPSAELoss(
        lam_centroid=args.lam_centroid,
        lam_material=args.lam_material,
        lam_edge_attr=args.lam_edge_attr,
        lam_node_type=args.lam_node_type,
        lam_freq=args.lam_freq,
        lam_adj_exist=args.lam_adj_exist,
        lam_adj_weight=args.lam_adj_weight,
    )

    save_dir = os.path.join(ROOT, "gps_ae_results")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # ── AMP GradScaler 
    scaler = torch.amp.GradScaler('cuda')

    # ── Training 
    train_h, val_h = [], []
    best_val = float("inf")
    patience = 0
    t0 = time.time()

    if rank == 0:
        print(f"\n  Training {args.epochs} epochs (patience={args.patience})")
        print(f"  Mixed precision: ON (AMP)")
        print("─" * 120)

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
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "train_history": train_h,
                    "val_history": val_h,
                    "best_val_loss": best_val,
                    "args": vars(args),
                }, os.path.join(save_dir, "gps_ae_best.pt"))
            else:
                patience += 1

            lr = optim.param_groups[0]["lr"]
            eta = (time.time() - t0) / (epoch + 1) * (args.epochs - epoch - 1)
            print(
                f"Ep {epoch+1:03d}/{args.epochs} | "
                f"Tr {tr_L['total']:.5f} | Va {va_L['total']:.5f} | "
                f"C:{va_L['centroid']:.4f} M:{va_L['material']:.4f} "
                f"E:{va_L['edge_attr']:.4f} NT:{va_L['node_type']:.3f} "
                f"AE:{va_L['adj_exist']:.4f} AW:{va_L['adj_weight']:.4f} | "
                f"LR:{lr:.1e} | ETA:{eta/60:.1f}m | "
                f"{'★BEST' if is_best else f'p={patience}/{args.patience}'}"
            )

        # Early stopping broadcast
        flag = torch.tensor(
            [1 if (rank == 0 and patience >= args.patience) else 0], device=device
        )
        dist.broadcast(flag, src=0)
        if flag.item():
            if rank == 0:
                print(f"\n  Early stop at epoch {epoch+1}")
            break

    # ── Post-training (rank 0) 
    if rank == 0:
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed/60:.1f} min | Best val: {best_val:.6f}")

        # Save final
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "train_history": train_h, "val_history": val_h, "args": vars(args),
        }, os.path.join(save_dir, "gps_ae_final.pt"))

        # Plots
        plot_losses(train_h, val_h, save_dir)

        # Load best for extraction
        best_model = GPSAutoencoder(
            x_dim=sample_info["x_dim"], edge_dim=sample_info["edge_dim"],
            max_nodes=sample_info["max_nodes"], hidden_dim=args.hidden_dim,
            enc_layers=args.enc_layers, dec_layers=args.dec_layers,
            num_heads=args.num_heads, geo_dim=args.geo_dim,
            dropout=args.dropout,
        ).to(device)
        ckpt = torch.load(os.path.join(save_dir, "gps_ae_best.pt"), map_location=device, weights_only=False)
        best_model.load_state_dict(ckpt["model_state_dict"])

        # Visualisations
        vis_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
        print("\n  Generating visualisations...")
        plot_centroid_recon(best_model, vis_loader, device, save_dir)
        plot_adj_quality(best_model, vis_loader, device, save_dir)

        # Extract z_xml over full dataset
        full_ds = SceneGraphDataset(cfg, list(range(n_total)), z_pc_path=z_pc_path)
        full_loader = DataLoader(full_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)
        print("  Extracting z_xml embeddings...")
        z_new = extract_z(best_model, full_loader, device)
        out_path = os.path.join(save_dir, "z_xml.npy")
        np.save(out_path, z_new)
        print(f"  Saved → {out_path}  shape: {z_new.shape}")

        # Compare
        if args.compare_with:
            old_path = (args.compare_with if os.path.isabs(args.compare_with)
                        else os.path.join(ROOT, args.compare_with))
            if os.path.exists(old_path):
                z_old = np.load(old_path)
                label = os.path.basename(old_path).replace(".npy", "")
                metrics = compare_embeddings(z_new, z_old, save_dir, label)
                with open(os.path.join(save_dir, "comparison_metrics.txt"), "w") as f:
                    for k, v in metrics.items():
                        f.write(f"{k}: {v:.4f}\n")

        # Summary
        print("\n" + "=" * 65)
        print("GPS AUTOENCODER — SUMMARY")
        print("=" * 65)
        print(f"  Dataset     : {args.dataset} ({n_total} samples)")
        print(f"  Architecture: GPS ({args.enc_layers} enc + {args.dec_layers} dec layers)")
        print(f"  z_dim       : {best_model.z_dim}")
        print(f"  GPUs        : {world_size}")
        print(f"  Train time  : {elapsed/60:.1f} min")
        print(f"  Best val    : {best_val:.6f}")
        best_h = min(val_h, key=lambda h: h["total"])
        print(f"\n  Per-component val losses (best epoch):")
        for k in LOSS_KEYS[1:]:
            print(f"    {k:14s}: {best_h[k]:.6f}")
        print(f"\n  Outputs: {save_dir}/")
        print(f"    z_xml.npy, gps_ae_best.pt, gps_ae_final.pt")
        print(f"    gps_ae_losses.png, centroid_recon.png, adj_prediction.png")

    cleanup_ddp()


 
# MAIN


def main():
    p = argparse.ArgumentParser(description="GPS Graph Autoencoder — standalone z_xml")

    # Dataset
    p.add_argument("--dataset", type=str, default="rooms_update",
                   choices=list(DATASETS.keys()))

    # Architecture
    p.add_argument("--hidden_dim",  type=int,   default=256)
    p.add_argument("--enc_layers",  type=int,   default=4)
    p.add_argument("--dec_layers",  type=int,   default=4)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--geo_dim",     type=int,   default=32,
                   help="Compressed dim for 128-dim PC features (0=skip)")
    p.add_argument("--dropout",     type=float, default=0.1)

    # Training
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=25)
    p.add_argument("--num_gpus",     type=int,   default=8)

    # Loss weights
    p.add_argument("--lam_centroid",   type=float, default=1.0)
    p.add_argument("--lam_material",   type=float, default=0.5)
    p.add_argument("--lam_edge_attr",  type=float, default=0.5)
    p.add_argument("--lam_node_type",  type=float, default=0.2)
    p.add_argument("--lam_freq",       type=float, default=0.1)
    p.add_argument("--lam_adj_exist",  type=float, default=1.0)
    p.add_argument("--lam_adj_weight", type=float, default=0.5)

    # Comparison
    p.add_argument("--compare_with", type=str, default=None,
                   help="Path to existing .npy embeddings to compare against")

    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available."); sys.exit(1)

    available = torch.cuda.device_count()
    if args.num_gpus > available:
        print(f"  [WARN] Requested {args.num_gpus} GPUs but device_count()={available}")
        print(f"  [WARN] If you have more GPUs, set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    args.num_gpus = min(args.num_gpus, max(available, 1))

    print("=" * 65)
    print("GPS GRAPH AUTOENCODER — standalone z_xml")
    print("=" * 65)
    print(f"  Architecture: GPS (EdgeCondConv + Dir-GNN + MultiHead Attention)")
    print(f"  GPUs detected: {available} | Using: {args.num_gpus}")
    for i in range(args.num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load config + inspect sample
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    ds_path = DATASETS[args.dataset]
    cfg["dataset"]["path"] = os.path.join(ds_path, "dataset.npy")
    z_pc_path = os.path.join(ds_path, "z_pc.npy")

    tmp = SceneGraphDataset(cfg, [0], z_pc_path=z_pc_path)
    s = tmp[0]
    sample_info = {
        "x_dim":     s.x.shape[1],
        "edge_dim":  s.edge_attr.shape[1],
        "max_nodes": s.x.shape[0],
    }
    print(f"\n  Graph: {sample_info['max_nodes']} nodes, "
          f"x_dim={sample_info['x_dim']}, edge_dim={sample_info['edge_dim']}")
    print(f"  z_dim: {args.hidden_dim * 3}")
    print(f"  GPS: {args.enc_layers} layers, {args.num_heads} heads, "
          f"hidden={args.hidden_dim}")
    print(f"  Reconstruction: nodes + edges + adjacency (exist + weight)")
    print(f"  Epochs: {args.epochs} | Batch/GPU: {args.batch_size} | "
          f"Effective: {args.batch_size * args.num_gpus}\n")

    mp.spawn(worker,
             args=(args.num_gpus, cfg, args, sample_info),
             nprocs=args.num_gpus, join=True)


if __name__ == "__main__":
    main()