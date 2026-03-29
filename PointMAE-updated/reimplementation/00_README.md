# Point-MAE: Masked Autoencoders for Point Cloud Self-supervised Learning

## Complete Architecture Documentation & Reimplementation Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture Flowchart (ASCII)](#architecture-flowchart)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture Details](#model-architecture-details)
5. [Training Pipeline](#training-pipeline)
6. [Configuration](#configuration)
7. [File Structure](#file-structure)
8. [Usage](#usage)

---

## Overview

Point-MAE is a self-supervised learning framework for 3D point clouds based on Masked Autoencoders (MAE). The key idea is to:
1. Divide point cloud into local patches (groups)
2. Mask a high ratio (60%) of patches
3. Encode visible patches with a Transformer
4. Decode and reconstruct masked patches
5. Use Chamfer Distance as reconstruction loss

**Key Features:**
- **No normals required**: Only uses XYZ coordinates (3 channels)
- **Pretraining on ShapeNet55**: 52,470 training shapes
- **Finetuning on ModelNet40/ScanObjectNN**: Classification downstream task
- **8-GPU distributed training support**

---

## Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              POINT-MAE ARCHITECTURE                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT POINT CLOUD
                                    ┌─────────────────┐
                                    │  B x N x 3      │
                                    │  (B=batch,      │
                                    │   N=1024 pts,   │
                                    │   3=xyz coords) │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │              POINT PATCH EMBEDDING               │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  1. FPS (Farthest Point Sampling)           │ │
                    │  │     - Select G=64 center points             │ │
                    │  │     - Centers: B x G x 3                    │ │
                    │  │                                             │ │
                    │  │  2. KNN Grouping (k=32)                     │ │
                    │  │     - Find 32 nearest neighbors per center  │ │
                    │  │     - Neighborhoods: B x G x M x 3          │ │
                    │  │       (G=64 groups, M=32 points each)       │ │
                    │  │                                             │ │
                    │  │  3. Normalize to local coordinates          │ │
                    │  │     - Subtract center from each point       │ │
                    │  └─────────────────────────────────────────────┘ │
                    └────────────────────────┬────────────────────────┘
                                             │
                                             ▼
                    ┌─────────────────────────────────────────────────┐
                    │              MINI-POINTNET ENCODER               │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  Input: B x G x M x 3                       │ │
                    │  │                                             │ │
                    │  │  Conv1d(3→128) + BN + ReLU                  │ │
                    │  │  Conv1d(128→256)                            │ │
                    │  │  MaxPool → Global Feature (256-dim)         │ │
                    │  │  Concat with local features → 512-dim       │ │
                    │  │  Conv1d(512→512) + BN + ReLU                │ │
                    │  │  Conv1d(512→384)                            │ │
                    │  │  MaxPool → Token (384-dim)                  │ │
                    │  │                                             │ │
                    │  │  Output: B x G x C (C=384)                  │ │
                    │  │  (64 tokens, each 384-dim)                  │ │
                    │  └─────────────────────────────────────────────┘ │
                    └────────────────────────┬────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │                  MASKING (60%)                   │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  Random Masking:                            │ │
                    │  │    - Randomly select 60% of 64 tokens       │ │
                    │  │    - ~38 tokens masked, ~26 visible         │ │
                    │  │                                             │ │
                    │  │  Block Masking (alternative):               │ │
                    │  │    - Pick random center, mask nearest 60%   │ │
                    │  │    - Creates spatially contiguous mask      │ │
                    │  │                                             │ │
                    │  │  Visible tokens: B x V x C (V≈26)           │ │
                    │  │  Masked positions: B x M (M≈38)             │ │
                    │  └─────────────────────────────────────────────┘ │
                    └────────────────────────┬────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │           POSITIONAL EMBEDDING (3D)              │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  Input: Center coordinates (B x G x 3)      │ │
                    │  │                                             │ │
                    │  │  MLP: Linear(3→128) + GELU + Linear(128→384)│ │
                    │  │                                             │ │
                    │  │  Output: B x G x 384                        │ │
                    │  │  (Learnable positional encoding from xyz)   │ │
                    │  └─────────────────────────────────────────────┘ │
                    └────────────────────────┬────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                              TRANSFORMER ENCODER                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Depth: 12 layers                                                                  │ │
│  │  Heads: 6                                                                          │ │
│  │  Dim: 384                                                                          │ │
│  │  MLP Ratio: 4 (hidden=1536)                                                        │ │
│  │  Drop Path Rate: 0.1 (stochastic depth)                                            │ │
│  │                                                                                    │ │
│  │  Each Block:                                                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  x = x + DropPath(Attention(LayerNorm(x + pos)))                             │  │ │
│  │  │  x = x + DropPath(MLP(LayerNorm(x)))                                         │  │ │
│  │  │                                                                              │  │ │
│  │  │  Attention:                                                                  │  │ │
│  │  │    - Q, K, V = Linear(x) split into 6 heads                                  │  │ │
│  │  │    - Attn = softmax(Q @ K.T / sqrt(d)) @ V                                   │  │ │
│  │  │    - Output = Linear(concat(heads))                                          │  │ │
│  │  │                                                                              │  │ │
│  │  │  MLP:                                                                        │  │ │
│  │  │    - Linear(384→1536) + GELU + Dropout                                       │  │ │
│  │  │    - Linear(1536→384) + Dropout                                              │  │ │
│  │  └──────────────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                                    │ │
│  │  Input: Visible tokens only (B x V x 384)                                          │ │
│  │  Output: Encoded visible tokens (B x V x 384)                                      │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │              DECODER PREPARATION                 │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  1. Learnable [MASK] token: 1 x 1 x 384     │ │
                    │  │     - Expand to B x M x 384                 │ │
                    │  │                                             │ │
                    │  │  2. Concatenate:                            │ │
                    │  │     - [Encoded visible] + [Mask tokens]     │ │
                    │  │     - Full sequence: B x G x 384            │ │
                    │  │                                             │ │
                    │  │  3. Position embeddings for all tokens      │ │
                    │  │     - Same MLP as encoder                   │ │
                    │  └─────────────────────────────────────────────┘ │
                    └────────────────────────┬────────────────────────┘
                                             │
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                              TRANSFORMER DECODER                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Depth: 4 layers                                                                   │ │
│  │  Heads: 6                                                                          │ │
│  │  Dim: 384                                                                          │ │
│  │  Same block structure as encoder                                                   │ │
│  │                                                                                    │ │
│  │  Input: Full sequence (B x G x 384)                                                │ │
│  │  Output: Only masked token predictions (B x M x 384)                               │ │
│  │          (Last M tokens after LayerNorm)                                           │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │              PREDICTION HEAD                     │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  Conv1d(384 → 3*32 = 96)                    │ │
                    │  │                                             │ │
                    │  │  Reshape: B x M x 96 → B*M x 32 x 3         │ │
                    │  │  (Predict 32 points per masked patch)       │ │
                    │  └─────────────────────────────────────────────┘ │
                    └────────────────────────┬────────────────────────┘
                                             │
                    ┌────────────────────────▼────────────────────────┐
                    │              RECONSTRUCTION LOSS                 │
                    │  ┌─────────────────────────────────────────────┐ │
                    │  │  Ground Truth: Original masked patches      │ │
                    │  │                B*M x 32 x 3                 │ │
                    │  │                                             │ │
                    │  │  Chamfer Distance L2:                       │ │
                    │  │    CD(P, Q) = mean(min_q ||p-q||²)          │ │
                    │  │            + mean(min_p ||p-q||²)           │ │
                    │  │                                             │ │
                    │  │  Loss = CD(predicted, ground_truth)         │ │
                    │  └─────────────────────────────────────────────┘ │
                    └─────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         FINETUNING ARCHITECTURE (Classification)                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT POINT CLOUD
                                    ┌─────────────────┐
                                    │  B x N x 3      │
                                    └────────┬────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │  Same Patch Embedding + Encoder       │
                         │  (Load pretrained weights)            │
                         │                                       │
                         │  + Learnable [CLS] token              │
                         │  Prepended to sequence                │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │  Transformer Encoder (12 layers)      │
                         │  All tokens (no masking)              │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │  Feature Aggregation:                 │
                         │  concat([CLS], max_pool(tokens))      │
                         │  → 768-dim feature                    │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │  Classification Head:                 │
                         │  Linear(768→256) + BN + ReLU + Drop   │
                         │  Linear(256→256) + BN + ReLU + Drop   │
                         │  Linear(256→num_classes)              │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                         ┌───────────────────────────────────────┐
                         │  Cross-Entropy Loss                   │
                         └───────────────────────────────────────┘
```

---

## Data Pipeline

### Supported Data Formats

| Format | Extension | Description | Columns |
|--------|-----------|-------------|---------|
| NumPy | `.npy` | Binary numpy array | N x 3 (xyz) or N x 6 (xyz + normals) |
| Text | `.txt` | Space/comma separated | N x 3 or N x 6 |
| HDF5 | `.h5` | HDF5 with 'data' key | N x 3 or N x 6 |

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────┘

  Raw Data Files                    Processing                    Model Input
  ─────────────────────────────────────────────────────────────────────────

  ShapeNet55/
  ├── shapenet_pc/
  │   ├── 02691156-xxx.npy  ──┐
  │   └── ...                 │
  └── ShapeNet-55/            │
      ├── train.txt ──────────┼──► Load .npy files
      └── test.txt            │    (8192 points each)
                              │         │
                              │         ▼
                              │    Random Sample
                              │    (8192 → 1024 pts)
                              │         │
                              │         ▼
                              │    Normalize (pc_norm)
                              │    - Center to origin
                              │    - Scale to unit sphere
                              │         │
                              │         ▼
                              │    Data Augmentation
                              │    - Scale: [0.67, 1.5]
                              │    - Translate: [-0.2, 0.2]
                              │         │
                              │         ▼
                              └────► Tensor: B x 1024 x 3


  ModelNet40/
  ├── modelnet40_normal_resampled/
  │   ├── airplane/
  │   │   └── airplane_0001.txt ──┐
  │   └── ...                      │
  └── modelnet40_train.txt ────────┼──► Load .txt files
                                   │    (comma-separated)
                                   │    N x 6 (xyz + normals)
                                   │         │
                                   │         ▼
                                   │    FPS Sampling
                                   │    (to 8192 points)
                                   │         │
                                   │         ▼
                                   │    Cache to .dat file
                                   │    (pickle format)
                                   │         │
                                   │         ▼
                                   │    Extract xyz only
                                   │    (discard normals)
                                   │         │
                                   │         ▼
                                   │    Normalize (pc_norm)
                                   │         │
                                   │         ▼
                                   └────► Tensor: B x 8192 x 3
                                          + Label: B
```

### txt to dat Conversion (ModelNet)

The ModelNet dataset uses a caching mechanism:

1. **First Run**: 
   - Read `.txt` files (comma-separated: x,y,z,nx,ny,nz)
   - Apply FPS to sample exactly N points
   - Save to `.dat` file using pickle

2. **Subsequent Runs**:
   - Load directly from `.dat` cache
   - Much faster loading

```python
# Cache file naming convention:
# modelnet{num_category}_{split}_{npoints}pts_fps.dat
# Example: modelnet40_train_8192pts_fps.dat
```

### Normalization (pc_norm)

```python
def pc_norm(pc):
    """Normalize point cloud to unit sphere centered at origin"""
    centroid = np.mean(pc, axis=0)  # Find center
    pc = pc - centroid               # Center at origin
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # Max distance
    pc = pc / m                      # Scale to unit sphere
    return pc
```

---

## Model Architecture Details

### Hyperparameters (Pretrain)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_group` | 64 | Number of point patches |
| `group_size` | 32 | Points per patch |
| `mask_ratio` | 0.6 | Fraction of patches to mask |
| `mask_type` | 'rand' | Random or block masking |
| `trans_dim` | 384 | Transformer hidden dimension |
| `encoder_dims` | 384 | Patch encoder output dimension |
| `depth` | 12 | Number of encoder layers |
| `num_heads` | 6 | Attention heads |
| `decoder_depth` | 4 | Number of decoder layers |
| `decoder_num_heads` | 6 | Decoder attention heads |
| `drop_path_rate` | 0.1 | Stochastic depth rate |
| `loss` | 'cdl2' | Chamfer Distance L2 |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Patch Encoder | ~0.5M |
| Position Embedding | ~50K |
| Transformer Encoder | ~7M |
| Transformer Decoder | ~2.3M |
| Prediction Head | ~37K |
| **Total** | **~10M** |

---

## Training Pipeline

### Pretraining (Self-supervised)

```
Optimizer: AdamW
  - lr: 0.001
  - weight_decay: 0.05

Scheduler: Cosine LR
  - epochs: 300
  - warmup_epochs: 10
  - min_lr: 1e-6

Batch Size: 128 (total across GPUs)
  - Per GPU: 128 / 8 = 16

Data Augmentation:
  - PointcloudScaleAndTranslate
    - scale: [0.67, 1.5]
    - translate: [-0.2, 0.2]
```

### Finetuning (Classification)

```
Optimizer: AdamW
  - lr: 0.0005
  - weight_decay: 0.05

Scheduler: Cosine LR
  - epochs: 300
  - warmup_epochs: 10

Batch Size: 32

Data Augmentation:
  - Same as pretraining
```

### Multi-GPU Training (8 GPUs)

```bash
# Launch command
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    main.py \
    --config cfgs/pretrain.yaml \
    --launcher pytorch \
    --exp_name pretrain_8gpu
```

---

## Configuration

### pretrain.yaml (Exact Copy)

```yaml
optimizer : {
  type: AdamW,
  kwargs: {
  lr : 0.001,
  weight_decay : 0.05
}}

scheduler: {
  type: CosLR,
  kwargs: {
    epochs: 300,
    initial_epochs : 10
}}

dataset : {
  train : { _base_: cfgs/dataset_configs/ShapeNet-55.yaml,
            others: {subset: 'train', npoints: 1024}},
  val : { _base_: cfgs/dataset_configs/ShapeNet-55.yaml,
            others: {subset: 'test', npoints: 1024}},
  test : { _base_: cfgs/dataset_configs/ShapeNet-55.yaml,
            others: {subset: 'test', npoints: 1024}}}

model : {
  NAME: Point_MAE,
  group_size: 32,
  num_group: 64,
  loss: cdl2,
  transformer_config: {
    mask_ratio: 0.6,
    mask_type: 'rand',
    trans_dim: 384,
    encoder_dims: 384,
    depth: 12,
    drop_path_rate: 0.1,
    num_heads: 6,
    decoder_depth: 4,
    decoder_num_heads: 6,
  },
}

npoints: 1024
total_bs : 128
step_per_update : 1
max_epoch : 300
```

---

## File Structure

```
reimplementation/
├── 00_README.md              # This documentation
├── config.py                 # Configuration management (YAML parsing, EasyDict)
├── data_io.py                # Data I/O utilities (npy, txt, h5 loading)
├── datasets.py               # Dataset classes (ShapeNet, ModelNet)
├── transforms.py             # Data augmentation transforms
├── model.py                  # Point-MAE model architecture
├── losses.py                 # Chamfer distance loss (pure PyTorch)
├── train_pretrain.py         # Pretraining runner
├── train_finetune.py         # Finetuning runner
├── main.py                   # Main entry point
├── test_reimplementation.py  # Test script to verify everything works
├── __init__.py               # Package init
├── cfgs/                     # Configuration files
│   ├── pretrain.yaml
│   ├── finetune_modelnet.yaml
│   └── dataset_configs/
│       ├── ShapeNet-55.yaml
│       └── ModelNet40.yaml
└── requirements.txt          # Dependencies
```

---

## Usage

### Installation

```bash
# Activate environment
source /home/joshi/experiments/.pointMAEenv/bin/activate

# Install dependencies (if needed)
pip install torch torchvision timm tensorboardX easydict pyyaml h5py tqdm scikit-learn

# Install pointnet2_ops (for FPS)
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Install knn_cuda
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Pretraining

```bash
cd /home/joshi/experiments/Point-MAE/reimplementation

# Single GPU
python 09_main.py --config cfgs/pretrain.yaml --exp_name pretrain_test

# 8 GPU Distributed
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    09_main.py \
    --config cfgs/pretrain.yaml \
    --launcher pytorch \
    --exp_name pretrain_8gpu
```

### Finetuning

```bash
# With pretrained weights
python 09_main.py \
    --config cfgs/finetune_modelnet.yaml \
    --finetune_model \
    --ckpts experiments/pretrain/ckpt-last.pth \
    --exp_name finetune_modelnet
```

### Testing

```bash
python 09_main.py \
    --config cfgs/finetune_modelnet.yaml \
    --test \
    --ckpts experiments/finetune/ckpt-best.pth \
    --exp_name test_modelnet
```

---

## Key Implementation Notes

1. **No Normals**: Point-MAE only uses XYZ coordinates. Normals are discarded if present.

2. **FPS + KNN**: Uses CUDA-accelerated operations from pointnet2_ops and knn_cuda.

3. **Chamfer Distance**: The original uses CUDA extension. Our reimplementation provides a pure PyTorch fallback.

4. **Masking Strategy**: Default is random masking. Block masking available for ablation.

5. **Position Encoding**: Learned from 3D coordinates via MLP, not sinusoidal.

6. **Decoder Only Predicts Masked**: Efficiency optimization - decoder output is only for masked tokens.

---

## Expected Results

| Dataset | Metric | Expected |
|---------|--------|----------|
| ModelNet40 | Accuracy | 93.2% |
| ScanObjectNN (hardest) | Accuracy | 85.2% |
| ShapeNet55 | Reconstruction CD | ~0.5 |

---

*Reimplementation by following the original Point-MAE paper and codebase.*
