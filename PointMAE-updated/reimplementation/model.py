"""
05_model.py - Point-MAE Model Architecture

This module implements the complete Point-MAE architecture:
- Encoder: Mini-PointNet for patch embedding
- Group: FPS + KNN for point cloud grouping
- MaskTransformer: Transformer encoder with masking
- Point_MAE: Full pretraining model
- PointTransformer: Finetuning model for classification

Architecture Flow:
1. Input: B x N x 3 point cloud
2. FPS + KNN → B x G x M x 3 (G=64 groups, M=32 points each)
3. Mini-PointNet → B x G x C tokens (C=384)
4. Mask 60% of tokens
5. Transformer Encoder → encoded visible tokens
6. Concat with mask tokens + Transformer Decoder
7. Predict masked patches using Chamfer Distance loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from timm.models.layers import DropPath, trunc_normal_

# Try to import CUDA-accelerated KNN, fall back to pure PyTorch
try:
    from knn_cuda import KNN
    HAS_KNN_CUDA = True
except ImportError:
    HAS_KNN_CUDA = False
    print("[Warning] knn_cuda not available, using PyTorch KNN")

# Try to import pointnet2_ops for FPS
try:
    from pointnet2_ops import pointnet2_utils
    HAS_POINTNET2 = True
except (ImportError, Exception) as e:
    HAS_POINTNET2 = False
    print(f"[Warning] pointnet2_ops not available ({e}), using PyTorch FPS")


# ============================================================================
# Utility Functions
# ============================================================================

def fps_pytorch(xyz, npoint):
    """
    Farthest Point Sampling in pure PyTorch.
    
    Args:
        xyz: (B, N, 3) point cloud
        npoint: Number of points to sample
        
    Returns:
        (B, npoint, 3) sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    # Random starting point
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    
    # Gather sampled points
    idx = centroids.unsqueeze(-1).expand(-1, -1, C)
    sampled_xyz = torch.gather(xyz, 1, idx)
    
    return sampled_xyz


def fps(data, number):
    """
    Farthest Point Sampling.
    Uses CUDA implementation if available, otherwise pure PyTorch.
    
    Args:
        data: (B, N, 3) point cloud
        number: Number of points to sample
        
    Returns:
        (B, number, 3) sampled points
    """
    if HAS_POINTNET2:
        fps_idx = pointnet2_utils.furthest_point_sample(data, number)
        fps_data = pointnet2_utils.gather_operation(
            data.transpose(1, 2).contiguous(), fps_idx
        ).transpose(1, 2).contiguous()
        return fps_data
    else:
        return fps_pytorch(data, number)


def knn_pytorch(ref, query, k):
    """
    K-Nearest Neighbors in pure PyTorch.
    
    Args:
        ref: (B, N, 3) reference points
        query: (B, M, 3) query points
        k: Number of neighbors
        
    Returns:
        (B, M, k) indices of k nearest neighbors
    """
    # Compute pairwise distances
    # ref: B x N x 3, query: B x M x 3
    # dist: B x M x N
    dist = torch.cdist(query, ref)
    _, idx = torch.topk(dist, k, dim=-1, largest=False)
    return idx


# ============================================================================
# Model Components
# ============================================================================

class Encoder(nn.Module):
    """
    Mini-PointNet Encoder for patch embedding.
    
    Converts local point patches to feature tokens.
    Input: B x G x M x 3 (G groups, M points per group)
    Output: B x G x C (C-dimensional token per group)
    """
    
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        
        # First convolution: 3 -> 128 -> 256
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        
        # Second convolution: 512 -> 512 -> encoder_channel
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    
    def forward(self, point_groups):
        """
        Args:
            point_groups: (B, G, M, 3) - G groups of M points each
            
        Returns:
            (B, G, C) - C-dimensional feature per group
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        
        # First conv: BG x 3 x N -> BG x 256 x N
        feature = self.first_conv(point_groups.transpose(2, 1))
        
        # Global max pooling: BG x 256 x 1
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        
        # Concatenate global and local features: BG x 512 x N
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        
        # Second conv: BG x 512 x N -> BG x C x N
        feature = self.second_conv(feature)
        
        # Global max pooling: BG x C
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """
    Point cloud grouping using FPS + KNN.
    
    1. FPS to select G center points
    2. KNN to find M nearest neighbors for each center
    3. Normalize to local coordinates (subtract center)
    """
    
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        
        if HAS_KNN_CUDA:
            self.knn = KNN(k=self.group_size, transpose_mode=True)
        else:
            self.knn = None
    
    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) input point cloud
            
        Returns:
            neighborhood: (B, G, M, 3) local point patches
            center: (B, G, 3) center points
        """
        batch_size, num_points, _ = xyz.shape
        
        # FPS to get center points: B x G x 3
        center = fps(xyz, self.num_group)
        
        # KNN to get neighborhoods
        if self.knn is not None:
            _, idx = self.knn(xyz, center)  # B x G x M
        else:
            idx = knn_pytorch(xyz, center, self.group_size)  # B x G x M
        
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        
        # Gather neighborhood points
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        
        # Normalize to local coordinates
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, center


class Mlp(nn.Module):
    """MLP block for Transformer."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks for encoding."""
    
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)
        ])
    
    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer blocks for decoding."""
    
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, pos, return_token_num):
        for block in self.blocks:
            x = block(x + pos)
        
        # Only return the last return_token_num tokens (masked tokens)
        x = self.head(self.norm(x[:, -return_token_num:]))
        return x


class MaskTransformer(nn.Module):
    """
    Transformer encoder with masking for Point-MAE pretraining.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        # Transformer config
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        self.mask_type = config.transformer_config.mask_type
        
        print(f'[MaskTransformer] Config: {config.transformer_config}')
        
        # Patch encoder
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        # Positional embedding (learned from 3D coordinates)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        
        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        
        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _mask_center_block(self, center, noaug=False):
        """Block masking: mask spatially contiguous patches."""
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )
            
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device)
        return bool_masked_pos
    
    def _mask_center_rand(self, center, noaug=False):
        """Random masking: randomly select patches to mask."""
        B, G, _ = center.shape
        
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        self.num_mask = int(self.mask_ratio * G)
        
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)
    
    def forward(self, neighborhood, center, noaug=False):
        # Generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)
        
        # Encode patches
        group_input_tokens = self.encoder(neighborhood)  # B x G x C
        
        batch_size, seq_len, C = group_input_tokens.size()
        
        # Select visible tokens
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        
        # Position embedding for visible tokens
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)
        
        # Transformer encoding
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        
        return x_vis, bool_masked_pos


class Point_MAE(nn.Module):
    """
    Point-MAE: Masked Autoencoder for Point Cloud Pretraining.
    
    Architecture:
    1. Group point cloud into patches (FPS + KNN)
    2. Encode patches with Mini-PointNet
    3. Mask 60% of patches
    4. Encode visible patches with Transformer
    5. Decode with mask tokens to reconstruct masked patches
    6. Chamfer Distance loss for reconstruction
    """
    
    def __init__(self, config):
        super().__init__()
        print(f'[Point_MAE] Initializing...')
        self.config = config
        
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        
        # Encoder
        self.MAE_encoder = MaskTransformer(config)
        
        # Grouping
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        print(f'[Point_MAE] Divide point cloud into G{self.num_group} x S{self.group_size} points')
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        
        # Decoder position embedding
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        # Decoder
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        
        # Prediction head: predict 3D coordinates for each point in masked patches
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        
        trunc_normal_(self.mask_token, std=.02)
        
        # Loss
        self.loss = config.loss
        self.build_loss_func(self.loss)
    
    def build_loss_func(self, loss_type):
        """Build reconstruction loss function."""
        from losses import ChamferDistanceL1, ChamferDistanceL2
        
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2()
        else:
            raise NotImplementedError(f"Loss {loss_type} not implemented")
    
    def forward(self, pts, vis=False, **kwargs):
        """
        Args:
            pts: (B, N, 3) input point cloud
            vis: Whether to return visualization data
            
        Returns:
            loss: Chamfer distance loss (if not vis)
            or visualization data (if vis)
        """
        # Group points into patches
        neighborhood, center = self.group_divider(pts)
        
        # Encode with masking
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape
        
        # Position embeddings
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        
        _, N, _ = pos_emd_mask.shape
        
        # Expand mask token
        mask_token = self.mask_token.expand(B, N, -1)
        
        # Concatenate visible and mask tokens
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
        
        # Decode
        x_rec = self.MAE_decoder(x_full, pos_full, N)
        
        # Predict points
        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        
        # Ground truth
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        
        # Compute loss
        loss = self.loss_func(rebuild_points, gt_points)
        
        if vis:
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            return ret1, ret2, full_center
        else:
            return loss


class PointTransformer(nn.Module):
    """
    Point Transformer for classification (finetuning).
    
    Uses pretrained encoder weights from Point_MAE.
    Adds classification head for downstream tasks.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        
        # Grouping
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # Patch encoder
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        
        # Position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        
        self.norm = nn.LayerNorm(self.trans_dim)
        
        # Classification head
        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )
        
        self.build_loss_func()
        
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
    
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, ret, gt):
        """Compute loss and accuracy."""
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100
    
    def load_model_from_ckpt(self, bert_ckpt_path):
        """Load pretrained weights from Point_MAE checkpoint."""
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu')
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            
            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            
            incompatible = self.load_state_dict(base_ckpt, strict=False)
            
            if incompatible.missing_keys:
                print('[Transformer] Missing keys:', incompatible.missing_keys)
            if incompatible.unexpected_keys:
                print('[Transformer] Unexpected keys:', incompatible.unexpected_keys)
            
            print(f'[Transformer] Loaded checkpoint from {bert_ckpt_path}')
        else:
            print('Training from scratch!')
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pts):
        """
        Args:
            pts: (B, N, 3) input point cloud
            
        Returns:
            (B, num_classes) classification logits
        """
        # Group points
        neighborhood, center = self.group_divider(pts)
        
        # Encode patches
        group_input_tokens = self.encoder(neighborhood)  # B x G x C
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        
        # Position embedding
        pos = self.pos_embed(center)
        
        # Concatenate CLS token
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        # Transformer encoding
        x = self.blocks(x, pos)
        x = self.norm(x)
        
        # Feature aggregation: concat CLS token and max-pooled features
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        
        # Classification
        ret = self.cls_head_finetune(concat_f)
        
        return ret


# Model registry
MODELS = {
    'Point_MAE': Point_MAE,
    'PointTransformer': PointTransformer,
}


def build_model(config):
    """Build model from config."""
    model_name = config.NAME
    if model_name not in MODELS:
        raise KeyError(f'{model_name} is not in the model registry')
    return MODELS[model_name](config)


if __name__ == '__main__':
    # Test model
    print("Testing model module...")
    
    from easydict import EasyDict
    
    # Create config
    config = EasyDict({
        'NAME': 'Point_MAE',
        'group_size': 32,
        'num_group': 64,
        'loss': 'cdl2',
        'transformer_config': EasyDict({
            'mask_ratio': 0.6,
            'mask_type': 'rand',
            'trans_dim': 384,
            'encoder_dims': 384,
            'depth': 12,
            'drop_path_rate': 0.1,
            'num_heads': 6,
            'decoder_depth': 4,
            'decoder_num_heads': 6,
        }),
    })
    
    # Build model
    model = build_model(config)
    print(f"Model built: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Test forward pass
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(2, 1024, 3).cuda()
    else:
        x = torch.randn(2, 1024, 3)
    
    print(f"Input shape: {x.shape}")
    
    try:
        loss = model(x)
        print(f"Loss: {loss.item():.4f}")
        print("Forward pass: OK")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    print("\nModel tests completed!")
