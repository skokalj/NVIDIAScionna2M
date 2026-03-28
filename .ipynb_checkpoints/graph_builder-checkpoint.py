import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data


def node_type_onehot(t, dim):
    v = torch.zeros(dim)
    v[t] = 1.0
    return v


class MaterialEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim)

    def forward(self, idx):
        return self.emb(idx)


def build_graph(cfg, xml_path, tx_pos, rx_pos, freq_norm):
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
                node_type_onehot(
                    0, cfg["node_features"]["node_type"]["dim"]
                )
            )

        if cfg["node_features"]["frequency"]["enabled"]:
            feats.append(freq_norm)

        node_features.append(torch.cat(feats))
        node_types.append(0)
        node_names.append(s["mesh"])
        centroids.append(centroid)

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
            node_type_onehot(
                1, cfg["node_features"]["node_type"]["dim"]
            )
        )
    if cfg["node_features"]["frequency"]["enabled"]:
        feats.append(freq_norm)

    node_features.append(torch.cat(feats))
    node_types.append(1)
    node_names.append("TX")
    centroids.append(tx_pos)

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
            node_type_onehot(
                2, cfg["node_features"]["node_type"]["dim"]
            )
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

    for i in range(num_obj):
        for j in range(num_obj):
            if i != j:
                add_edge(i, j)

    for src in [tx_index, rx_index]:
        for j in range(num_obj):
            add_edge(src, j)
            add_edge(j, src)

    add_edge(tx_index, rx_index)
    add_edge(rx_index, tx_index)

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long)
        .t()
        .contiguous(),
        edge_attr=torch.stack(edge_attr),
    )

    data.node_type = torch.tensor(node_types, dtype=torch.long)
    data.tx_index = tx_index
    data.rx_index = rx_index
    data.node_names = node_names

    return data