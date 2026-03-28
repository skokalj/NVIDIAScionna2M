"""
Dataset module for Graph project - OPTIMIZED VERSION
Handles data loading, graph building, and PyTorch Geometric dataset.
Loads dataset.npy ONCE instead of on every sample.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# Graph Building Utilities


def node_type_onehot(t, dim):
    """Create one-hot encoding for node type."""
    v = torch.zeros(dim)
    v[t] = 1.0
    return v


class MaterialEmbedding(torch.nn.Module):
    """Embedding layer for material types."""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim)

    def forward(self, idx):
        return self.emb(idx)


# Graph Building

def build_graph(cfg, xml_path, tx_pos, rx_pos, freq_norm):
    """
    Build a graph from XML scene description.
    
    Args:
        cfg: Configuration dictionary with node_features settings
        xml_path: Path to XML scene file
        tx_pos: Transmitter position tensor
        rx_pos: Receiver position tensor
        freq_norm: Normalized frequency tensor
        
    Returns:
        PyG Data object with node features, edges, and attributes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    material_map = {}
    shapes = []

    for shape in root.findall(".//shape"):
        mesh = shape.find("string").attrib["value"]
        ref = shape.find("ref").attrib["id"]

        if ref not in material_map:
            material_map[ref] = len(material_map)

        shapes.append({"mesh": mesh, "mat": material_map[ref]})

    mat_cfg = cfg["node_features"]["material"]
    mat_emb = (
        MaterialEmbedding(mat_cfg["vocab_size"], mat_cfg["dim"])
        if mat_cfg["enabled"]
        else None
    )

    node_features = []
    node_types = []
    node_names = []
    centroids = []

    # Process shape nodes
    for s in shapes:
        feats = []

        if mat_cfg["enabled"]:
            feats.append(mat_emb(torch.tensor(s["mat"])))

        centroid = torch.rand(3)
        if cfg["node_features"]["centroid"]:
            feats.append(centroid)

        if cfg["node_features"]["bbox"]:
            feats.append(torch.rand(3))

        if cfg["node_features"]["node_type"]["enabled"]:
            feats.append(
                node_type_onehot(0, cfg["node_features"]["node_type"]["dim"])
            )

        if cfg["node_features"]["frequency"]["enabled"]:
            feats.append(freq_norm)

        node_features.append(torch.cat(feats))
        node_types.append(0)
        node_names.append(s["mesh"])
        centroids.append(centroid)

    # Add TX node
    tx_index = len(node_features)
    feats = []

    if mat_cfg["enabled"]:
        feats.append(torch.zeros(mat_cfg["dim"]))
    if cfg["node_features"]["centroid"]:
        feats.append(tx_pos)
    if cfg["node_features"]["bbox"]:
        feats.append(torch.zeros(3))
    if cfg["node_features"]["node_type"]["enabled"]:
        feats.append(
            node_type_onehot(1, cfg["node_features"]["node_type"]["dim"])
        )
    if cfg["node_features"]["frequency"]["enabled"]:
        feats.append(freq_norm)

    node_features.append(torch.cat(feats))
    node_types.append(1)
    node_names.append("TX")
    centroids.append(tx_pos)

    # Add RX node
    rx_index = len(node_features)
    feats = []

    if mat_cfg["enabled"]:
        feats.append(torch.zeros(mat_cfg["dim"]))
    if cfg["node_features"]["centroid"]:
        feats.append(rx_pos)
    if cfg["node_features"]["bbox"]:
        feats.append(torch.zeros(3))
    if cfg["node_features"]["node_type"]["enabled"]:
        feats.append(
            node_type_onehot(2, cfg["node_features"]["node_type"]["dim"])
        )
    if cfg["node_features"]["frequency"]["enabled"]:
        feats.append(freq_norm)

    node_features.append(torch.cat(feats))
    node_types.append(2)
    node_names.append("RX")
    centroids.append(rx_pos)

    x = torch.stack(node_features)
    centroids = torch.stack(centroids)

    edge_index = []
    edge_attr = []

    def add_edge(i, j):
        d = centroids[j] - centroids[i]
        dist = torch.norm(d)
        edge_index.append([i, j])
        edge_attr.append(torch.cat([dist.view(1), d / (dist + 1e-6)]))

    num_obj = len(shapes)

    # Fully connect object nodes
    for i in range(num_obj):
        for j in range(num_obj):
            if i != j:
                add_edge(i, j)

    # Connect TX/RX to all objects
    for src in [tx_index, rx_index]:
        for j in range(num_obj):
            add_edge(src, j)
            add_edge(j, src)

    # Connect TX <-> RX
    add_edge(tx_index, rx_index)
    add_edge(rx_index, tx_index)

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attr),
    )

    data.node_type = torch.tensor(node_types, dtype=torch.long)
    data.tx_index = tx_index
    data.rx_index = rx_index
    data.node_names = node_names

    return data


# PyTorch Geometric Dataset - OPTIMIZED

class SceneGraphDataset(Dataset):
    """PyTorch Geometric Dataset for scene graphs - OPTIMIZED VERSION."""
    
    def __init__(self, cfg, sample_indices, z_pc_path=None):
        """
        Args:
            cfg: Configuration dictionary
            sample_indices: List of sample indices to use
            z_pc_path: Path to z_pc.npy file (optional, defaults to data/rooms/z_pc.npy)
        """
        super().__init__()
        self.cfg = cfg
        self.sample_indices = sample_indices
        self.dataset_path = cfg["dataset"]["path"]
        
        # CRITICAL FIX: Load dataset ONCE, not in every get() call
        print(f"Loading dataset from {self.dataset_path}...")
        data = np.load(self.dataset_path, allow_pickle=True)
        self.tx_positions = data[0]  # (N, 3)
        self.rx_positions = data[1]  # (N, 3)
        self.xml_paths = data[2]     # (N,)
        self.frequency = float(data[3])  # scalar
        self.channel_data = data[4]  # (N, 107)
        print(f"  Loaded {len(self.tx_positions)} samples")
        
        # Precompute dataset root for XML path resolution
        from pathlib import Path
        self.dataset_root = Path(self.dataset_path).parent
        self.scenes_dir = self.dataset_root / "scenes"
        
        # CRITICAL FIX 2: Cache parsed XML (all samples use same scene!)
        print(f"Parsing XML scene...")
        xml_name = os.path.basename(self.xml_paths[0])
        if self.scenes_dir.exists():
            self.xml_path = self.scenes_dir / xml_name
        else:
            self.xml_path = self.dataset_root / xml_name
        
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found: {self.xml_path}")
        
        # Parse XML ONCE and cache the result
        tree = ET.parse(str(self.xml_path))
        root = tree.getroot()
        
        material_map = {}
        self.shapes = []
        
        for shape in root.findall(".//shape"):
            mesh = shape.find("string").attrib["value"]
            ref = shape.find("ref").attrib["id"]
            
            if ref not in material_map:
                material_map[ref] = len(material_map)
            
            self.shapes.append({"mesh": mesh, "mat": material_map[ref]})
        
        print(f"  Found {len(self.shapes)} objects in scene")
        
        # Load z_pc
        if z_pc_path is None:
            z_pc_path = "/data/hafeez/graphdata/rooms_update/z_pc.npy"
        
        self.z_pc = torch.from_numpy(np.load(z_pc_path)).float()

    def len(self):
        return len(self.sample_indices)

    def get(self, idx):
        # Get actual sample index
        i = self.sample_indices[idx]
        
        # Access pre-loaded data (FAST!)
        tx_pos = torch.tensor(self.tx_positions[i], dtype=torch.float)
        rx_pos = torch.tensor(self.rx_positions[i], dtype=torch.float)
        tab = torch.tensor(self.channel_data[i], dtype=torch.float)
        
        # Build graph using CACHED XML data (no parsing!)
        scale = float(self.cfg["frequency"]["scale"])
        freq_norm = torch.tensor([self.frequency / scale])
        
        graph = self._build_graph_cached(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            freq_norm=freq_norm,
        )
        
        # Prepend z_pc
        num_nodes = graph.x.shape[0]
        z_pc = self.z_pc[:num_nodes]
        graph.x = torch.cat([z_pc, graph.x], dim=1)
        graph.y = tab
        
        return graph
    
    def _build_graph_cached(self, tx_pos, rx_pos, freq_norm):
        """Build graph using cached XML data (no parsing needed)."""
        cfg = self.cfg
        shapes = self.shapes  # Use cached shapes
        
        mat_cfg = cfg["node_features"]["material"]
        mat_emb = (
            MaterialEmbedding(mat_cfg["vocab_size"], mat_cfg["dim"])
            if mat_cfg["enabled"]
            else None
        )
        
        node_features = []
        node_types = []
        node_names = []
        centroids = []
        
        # Process shape nodes
        for s in shapes:
            feats = []
            
            if mat_cfg["enabled"]:
                feats.append(mat_emb(torch.tensor(s["mat"])))
            
            centroid = torch.rand(3)
            if cfg["node_features"]["centroid"]:
                feats.append(centroid)
            
            if cfg["node_features"]["bbox"]:
                feats.append(torch.rand(3))
            
            if cfg["node_features"]["node_type"]["enabled"]:
                feats.append(
                    node_type_onehot(0, cfg["node_features"]["node_type"]["dim"])
                )
            
            if cfg["node_features"]["frequency"]["enabled"]:
                feats.append(freq_norm)
            
            node_features.append(torch.cat(feats))
            node_types.append(0)
            node_names.append(s["mesh"])
            centroids.append(centroid)
        
        # Add TX node
        tx_index = len(node_features)
        feats = []
        
        if mat_cfg["enabled"]:
            feats.append(torch.zeros(mat_cfg["dim"]))
        if cfg["node_features"]["centroid"]:
            feats.append(tx_pos)
        if cfg["node_features"]["bbox"]:
            feats.append(torch.zeros(3))
        if cfg["node_features"]["node_type"]["enabled"]:
            feats.append(
                node_type_onehot(1, cfg["node_features"]["node_type"]["dim"])
            )
        if cfg["node_features"]["frequency"]["enabled"]:
            feats.append(freq_norm)
        
        node_features.append(torch.cat(feats))
        node_types.append(1)
        node_names.append("TX")
        centroids.append(tx_pos)
        
        # Add RX node
        rx_index = len(node_features)
        feats = []
        
        if mat_cfg["enabled"]:
            feats.append(torch.zeros(mat_cfg["dim"]))
        if cfg["node_features"]["centroid"]:
            feats.append(rx_pos)
        if cfg["node_features"]["bbox"]:
            feats.append(torch.zeros(3))
        if cfg["node_features"]["node_type"]["enabled"]:
            feats.append(
                node_type_onehot(2, cfg["node_features"]["node_type"]["dim"])
            )
        if cfg["node_features"]["frequency"]["enabled"]:
            feats.append(freq_norm)
        
        node_features.append(torch.cat(feats))
        node_types.append(2)
        node_names.append("RX")
        centroids.append(rx_pos)
        
        x = torch.stack(node_features)
        centroids = torch.stack(centroids)
        
        edge_index = []
        edge_attr = []
        
        def add_edge(i, j):
            d = centroids[j] - centroids[i]
            dist = torch.norm(d)
            edge_index.append([i, j])
            edge_attr.append(torch.cat([dist.view(1), d / (dist + 1e-6)]))
        
        num_obj = len(shapes)
        
        # Fully connect object nodes
        for i in range(num_obj):
            for j in range(num_obj):
                if i != j:
                    add_edge(i, j)
        
        # Connect TX/RX to all objects
        for src in [tx_index, rx_index]:
            for j in range(num_obj):
                add_edge(src, j)
                add_edge(j, src)
        
        # Connect TX <-> RX
        add_edge(tx_index, rx_index)
        add_edge(rx_index, tx_index)
        
        data = Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.stack(edge_attr),
        )
        
        data.node_type = torch.tensor(node_types, dtype=torch.long)
        data.tx_index = tx_index
        data.rx_index = rx_index
        data.node_names = node_names
        
        return data