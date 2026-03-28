(zubairEnv) hafeez@brev-36hgjdx8a:~/Graph$ python benchmark_genie.py
Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1+cu118
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
============================================================
GENIE BENCHMARK
============================================================
GPUs available: 1
GPU used: NVIDIA A100-SXM4-80GB

Dataset: rooms_update
Total samples: 10000
Loading dataset from /data/hafeez/graphdata/rooms_update/dataset.npy...
  Loaded 10000 samples
Parsing XML scene...
  Found 55 objects in scene
Loading dataset from /data/hafeez/graphdata/rooms_update/dataset.npy...
  Loaded 10000 samples
Parsing XML scene...
  Found 55 objects in scene

============================================================
1. TRAINING TIME
============================================================
Train samples: 9000
Val samples: 1000
Batch size: 32
Epochs: 10

Training 10 epochs...
  Epoch 8 done                                                                           
  Epoch 9 done                                                                           
  Epoch 10 done                                                                          

>>> TRAINING TIME: 8561.23s (142.69 min)
>>> Per epoch: 856.12s

============================================================
2. INFERENCE TIME (Z_XML extraction)
============================================================
Loading dataset from /data/hafeez/graphdata/rooms_update/dataset.npy...
  Loaded 10000 samples
Parsing XML scene...
  Found 55 objects in scene
Loaded weights from /home/hafeez/Graph/Genie_best.pt

Preloading 10000 samples to GPU (batch_size=128)...
Preloading: 100%|████████████████████████████████████████| 79/79 [15:57<00:00, 12.12s/it]
Preloaded 79 batches

Measuring inference time (2 runs)...
  Run 1: 8.1546s
  Run 2: 1.7447s

>>> INFERENCE TIME: 4.9496s
>>> Z_XML shape: (10000, 576)
>>> Throughput: 2020.35 samples/second
>>> Per sample: 0.4950 ms

============================================================
SUMMARY
============================================================

Dataset: rooms_update (10000 samples)
Hardware: 1x A100-80GB available, 1 GPU used
Note: PyTorch Geometric graphs don't parallelize across GPUs

TRAINING (10 epochs, 9000 train + 1000 val, batch_size=32):
  Total: 8561.23s (142.69 min)
  Per epoch: 856.12s

                                                                         

## Overview

This document describes the complete pipeline for wireless channel prediction using graph neural networks, combining point cloud geometry embeddings (Point-MAE) with scene graph representations (GINE).

---

## Directory Structure

```
/home/hafeez/
├── Graph/                          # Main project directory
│   ├── data/
│   │   └── graph_dataset.py        # PyTorch Geometric dataset
│   ├── configs/
│   │   └── default.yaml            # Configuration file
│   ├── scripts/
│   │   └── extract.py              # Point-MAE embedding extraction
│   ├── train_genie.py              # GENIE training script (OPTIMIZED)
│   └── utils/
│       └── config.py               # Config loading utilities
│
├── Point-MAE/                      # Point cloud encoder
    ├── models/
    ├── cfgs/
    │   └── pretrain.yaml
    └── tools/
    
Point-MAE
/data/joshi/
|__modelnet40_trained__mae/
|   |__ckpt-last.pth

/data/hafeez/graphdata/
├── rooms/
│   ├── dataset.npy                 # 10,000 samples
│   ├── z_pc.npy                    # (61, 128) geometry embeddings
│   ├── scenes/                     # XML scene files
│   └── pointclouds/                # PLY mesh files
│
├── rooms_update/
│   ├── dataset.npy                 # 10,000 samples
│   ├── z_pc.npy                    # (61, 128) geometry embeddings
│   ├── scenes/
│   └── pointclouds/
│
└── roomsmoved/
    ├── dataset.npy                 # 100 samples
    ├── z_pc.npy                    # (157, 128) geometry embeddings
    ├── scenes/
    └── pointclouds/
```

---

## Data Structure

### dataset.npy Structure

**Format:** NumPy array with 5 elements

```python
dataset.npy = [
    tx_positions,    # Index 0: list of (N_samples, 3) - Transmitter positions
    rx_positions,    # Index 1: list of (N_samples, 3) - Receiver positions
    xml_paths,       # Index 2: list of N_samples strings - Scene XML file paths
    frequency,       # Index 3: float scalar - Carrier frequency (Hz)
    channel_data     # Index 4: list of (N_samples, 107) - Channel response
]
```

**Example for rooms_update:**
- Total samples: 10,000
- Scenes: 1 XML scene (reused with different TX/RX positions)
- Channel data size: 107 values per sample
- Sampling rate: 100 MHz

### z_pc.npy Structure

**Format:** NumPy array of geometry embeddings

```python
z_pc.npy shape: (N_objects, 128)
```

**Per dataset:**
- `rooms`: (61, 128) - 61 objects with 128-D embeddings
- `rooms_update`: (61, 128) - 61 objects with 128-D embeddings  
- `roomsmoved`: (157, 128) - 157 objects with 128-D embeddings

**Important:** Each dataset has a **different number of objects** because:
- Different scenes have different numbers of meshes
- Each mesh (.ply file) gets one 128-D embedding
- Embeddings are **reused across all samples** in that dataset

---

## Scene Graph Structure

### Nodes (58 total for rooms/rooms_update)

```
58 nodes = 56 object nodes + 1 TX node + 1 RX node
```

- **Object nodes (56):** Each `<shape>` in the XML file
- **TX node (1):** Transmitter position
- **RX node (1):** Receiver position

**Note:** The number of nodes varies by scene:
- rooms/rooms_update: 56 objects → 58 total nodes
- roomsmoved: More objects → different total

### Node Features (170 dimensions)

Each node feature vector is constructed as:

```python
x_node = [z_pc(128) | material(32) | centroid(3) | bbox(3) | node_type(3) | frequency(1)]
         └─────────┘ └──────────────────────────────────────────────────────────────────┘
          prepended              built in build_graph()
```

**Breakdown:**

| Feature | Dims | Description | Source |
|---------|------|-------------|--------|
| **z_pc** | 128 | 3D geometry from Point-MAE | Loaded from z_pc.npy |
| **material** | 32 | Material embedding (learned) | Material vocab in XML |
| **centroid** | 3 | 3D position (x, y, z) | Object/TX/RX position |
| **bbox** | 3 | Bounding box (w, h, d) | Currently placeholder |
| **node_type** | 3 | One-hot: [object, TX, RX] | Node category |
| **frequency** | 1 | Normalized RF frequency | Carrier frequency |
| **TOTAL** | **170** | Complete node representation | |

**Construction order:**
1. `build_graph()` creates 42-D features: material(32) + centroid(3) + bbox(3) + node_type(3) + freq(1)
2. `SceneGraphDataset.get()` prepends 128-D z_pc
3. Final: 128 + 42 = **170 dimensions**

### Edge Features (4 dimensions)

```python
edge_attr = [distance, dx, dy, dz]
```

- **distance (1):** Euclidean distance between node centroids
- **direction (3):** Unit direction vector (dx, dy, dz)

### Edge Construction

Edges are fully connected with physical meaning:

| Edge Type | Count (for 56 objects) | Purpose |
|-----------|------------------------|---------|
| Object ↔ Object | ~56 × 55 = 3,080 | Spatial relationships |
| TX → Object | 56 | Signal propagation from TX |
| Object → TX | 56 | Reverse edges |
| RX → Object | 56 | Signal reception at RX |
| Object → RX | 56 | Reverse edges |
| TX ↔ RX | 2 | Direct line-of-sight |
| **Total** | **~3,306 edges** | Full connectivity |

**Note:** Edge count varies with number of objects:
- rooms/rooms_update (56 objects): ~3,306 edges
- roomsmoved (more objects): proportionally more edges

---

## Pipeline Stages

### Stage 1: Point Cloud Embedding (Point-MAE)

**Script:** `scripts/extract.py`

**Input:**
- PLY mesh files in `{dataset}/pointclouds/*.ply`
- Point-MAE checkpoint: `/data/joshi/modelnet40_trained__mae/ckpt-last.pth`

**Process:**
1. Convert PLY meshes to .txt point clouds (x, y, z, nx, ny, nz)
2. Load pretrained Point-MAE model (384-D encoder)
3. For each point cloud:
   - Split into local groups (G=64, S=32 points)
   - Encode using MAE encoder
   - Average visible token features
   - Project 384-D → 128-D
4. Save all embeddings in order

**Output:**
- `{dataset}/z_pc.npy` with shape `(N_objects, 128)`

**Example:**
```bash
python scripts/extract.py --dataset rooms_update
# Output: /data/hafeez/graphdata/rooms_update/z_pc.npy (61, 128)
```

**Key characteristics:**
- One 128-D embedding per mesh object
- **Same embeddings reused for all samples** in the dataset
- Scene-agnostic: doesn't depend on TX/RX positions

---

### Stage 2: Graph Construction

**Code:** `data/graph_dataset.py`

**Per sample process:**

1. **Load sample data:**
   ```python
   sample = {
       'tx': (3,),      # TX position
       'rx': (3,),      # RX position
       'xml': path,     # Scene file
       'freq': float,   # Frequency
       'tab': (107,)    # Channel response (target)
   }
   ```

2. **Build graph from XML:**
   ```python
   # Parse XML for objects and materials
   # Create nodes: [objects (56) + TX (1) + RX (1)]
   # Create features: material(32) + centroid(3) + bbox(3) + node_type(3) + freq(1)
   # Build edges: fully connected with distance + direction
   ```

3. **Prepend geometry:**
   ```python
   num_nodes = 58  # For rooms/rooms_update
   z_pc = self.z_pc[:num_nodes]  # Take first 58 embeddings
   graph.x = torch.cat([z_pc, features], dim=1)  # (58, 128+42=170)
   ```

4. **Return PyG Data:**
   ```python
   Data(
       x=(58, 170),           # Node features
       edge_index=(2, 3306),  # Edge connectivity
       edge_attr=(3306, 4),   # Edge features
       y=(107,),              # Target channel response
       node_type=(58,),       # Node categories
       tx_index=56,           # TX node index
       rx_index=57            # RX node index
   )
   ```

**Important:** z_pc lookup is **position-based**, not scene-specific:
- Always uses `z_pc[:num_nodes]` 
- Same geometry for all samples (different TX/RX only)

---

### Stage 3: GENIE Training

**Script:** `train_genie.py` 

**Model Architecture:**

```
Input: PyG Data(x, edge_index, edge_attr)
  ↓
1. Edge Encoding
   edge_attr (4,) → EdgeEncoder → (192,)
  
2. Node Feature Processing
   x (170,) → Split: x_geo (128,) | x_other (42,)
   x_geo (128,) → GeoProj → (16,)  [compression!]
   Concatenate: (16,) + (42,) = (58,)
   → NodeEncoder → (192,)
  
3. Message Passing (3 layers)
   For each GenieLayer:
     h = LayerNorm(h + MessagePassing(h, edge_index, edge_attr))
   Output: h (N_nodes, 192)
  
4. Graph Pooling (pooling="txrx")
   obj_emb = mean_pool(h[node_type==0])  → (192,)  [objects]
   tx_emb  = mean_pool(h[node_type==1])  → (192,)  [TX]
   rx_emb  = mean_pool(h[node_type==2])  → (192,)  [RX]
   z_xml = concat([obj_emb, tx_emb, rx_emb])  → (576,)  ← This is z_xml!
  
5. Prediction Head
   z_xml (576,) → MLP(192) → ReLU → Linear → (107,)
  
Output: Channel prediction (107,)
```

**Key dimensions:**
- Input node features: 170 (128 geometry + 42 scene)
- Compressed to: 58 (16 geometry + 42 scene)
- Hidden dim: 192
- **z_xml dim: 576** (192 × 3 for txrx pooling)
- Output: 107 (channel response)
- Total params: 664,571

**Training configuration (from default.yaml):**

```yaml
model:
  hidden_dim: 192
  layers: 3
  pooling: "txrx"      # Separate obj/tx/rx pooling
  geo_dim: 16          # Compress 128→16

node_features:
  material:
    dim: 32            # Material embedding size
    vocab_size: 20     # Number of material types
  centroid: true       # Include 3D positions
  bbox: true           # Include bounding boxes
  node_type:
    dim: 3             # One-hot encoding
  frequency:
    enabled: true      # Include RF frequency
```

**Optimizations applied:**
- Batch size: 128 (vs original 1)
- Pin memory: enabled (fast CPU→GPU)
- Non-blocking transfers
- num_workers: 0 (for compatibility)
- extract_workers: 2 (for extraction only)

**Training command:**
```bash
# Quick test
python train_genie.py \
    --datasets rooms_update \
    --max_samples 100 \
    --epochs 10 \
    --device cuda

# Full training
python train_genie.py \
    --datasets rooms_update \
    --epochs 100 \
    --device cuda \
    --extract \
    --extract_workers 2
```

**Outputs:**
- `Genie.pt`: Final model weights
- `Genie_best.pt`: Best model (lowest val loss)
- `loss_curve.png`: Training/validation curves
- `z_xml.npy`: Embeddings (10000, 576) ← Scene-level embeddings

---

## z_xml: Graph-Level Embeddings

**What is z_xml?**
- Graph-level embedding extracted from GENIE model
- Captures entire scene in fixed 576-D vector
- Combines object, TX, and RX information

**Dimension breakdown:**
```
z_xml (576,) = [obj_embedding(192,) | tx_embedding(192,) | rx_embedding(192,)]
```

- **obj_embedding (192):** Aggregated information from all object nodes
- **tx_embedding (192):** Transmitter node embedding
- **rx_embedding (192):** Receiver node embedding

**How it's computed:**
1. Message passing layers process node features
2. Pool nodes by type (object/TX/RX)
3. Concatenate three 192-D embeddings
4. Result: 576-D scene representation

**Uses:**
- Input to downstream prediction tasks
- Scene similarity comparison
- Transfer learning to new tasks
- Compressed scene representation

**Example output:**
```
z_xml.npy shape: (10000, 576)
- 10,000 samples from rooms_update
- Each sample: one 576-D embedding
- Represents: scene geometry + TX/RX configuration
```

---

## Comparison: z_pc vs z_xml

| Property | z_pc | z_xml |
|----------|------|-------|
| **Source** | Point-MAE encoder | GENIE graph model |
| **Scope** | Per-object geometry | Per-scene (all nodes) |
| **Dimension** | 128 | 576 |
| **Input** | PLY mesh files | Scene graph + z_pc |
| **Reuse** | Same for all samples | Unique per sample |
| **Info** | 3D shape only | Geometry + materials + TX/RX |
| **Task** | Geometry encoding | Channel prediction |
| **Count** | N_objects (61, 128) | N_samples (10000, 576) |

**Key difference:**
- **z_pc**: Object-centric, geometry-only, reused across samples
- **z_xml**: Sample-specific, includes TX/RX positions, learned task-specific

---

## Dataset Statistics

### rooms
- Samples: 10,000
- Objects: 61
- z_pc shape: (61, 128)
- z_xml shape: (10000, 576)
- Nodes per graph: 58 (56 objects + TX + RX)
- Edges per graph: ~3,306

### rooms_update  
- Samples: 10,000
- Objects: 61
- z_pc shape: (61, 128)
- z_xml shape: (10000, 576)
- Nodes per graph: 58 (56 objects + TX + RX)
- Edges per graph: ~3,306
- **Same scene as rooms, different TX/RX positions**

### roomsmoved
- Samples: 100
- Objects: 157
- z_pc shape: (157, 128)
- z_xml shape: (100, 576) [when extracted]
- Nodes per graph: 159 (157 objects + TX + RX)
- Edges per graph: ~25,000+
- **Different scene with more objects**

---


## Quick Reference Commands

### Generate z_pc embeddings
```bash
python scripts/extract.py --dataset rooms_update
```

### Train GENIE (quick test)
```bash
python train_genie.py \
    --datasets rooms_update \
    --max_samples 100 \
    --epochs 5 \
    --profile \
    --device cuda
```

### Train GENIE (full, with extraction)
```bash
python train_genie.py \
    --datasets rooms_update \
    --epochs 100 \
    --device cuda \
    --extract \
    --extract_workers 2
```

### Train only (skip extraction)
```bash
python train_genie.py \
    --datasets rooms_update \
    --epochs 100 \
    --device cuda
```

### Extract z_xml only (from pretrained model)
```bash
python train_genie.py \
    --datasets rooms_update \
    --epochs 0 \
    --extract \
    --extract_workers 2 \
    --device cuda
```

---

