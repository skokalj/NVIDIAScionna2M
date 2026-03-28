import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data
from pyvis.network import Network
import numpy as np

XML_FILE = "test1.xml"
HTML_OUT = "scene_graph.html"
DEVICE = "cpu"

EDGE_MODE = "full"   # "full" or "knn"
K_OBJ = 6
K_TR = 8

Z_PC_DIM = 64
MAT_EMB_DIM = 8
NODE_TYPE_DIM = 3
TX_POS = torch.tensor([1.0, 1.0, 1.5])
RX_POS = torch.tensor([4.0, 2.5, 1.5])

FREQ_DIM = 1
dataset = np.load("dataset.npy", allow_pickle=True)
FREQ = float(dataset[3])          # carrier frequency
FREQ_NORM = torch.tensor([FREQ / 1e9])  # shape (1,)

def node_type_onehot(t):
    v = torch.zeros(NODE_TYPE_DIM)
    v[t] = 1.0
    return v
    
def load_z_pc(mesh_file):
    torch.manual_seed(abs(hash(mesh_file)) % (2**31))
    z_pc = torch.randn(Z_PC_DIM)
    centroid = torch.rand(3)
    bbox = torch.rand(3)
    return z_pc, centroid, bbox


class MaterialEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(64, MAT_EMB_DIM)

    def forward(self, idx):
        return self.emb(idx)


def build_graph(xml_path, tx_pos, rx_pos, edge_mode="full", k_obj=6, k_tr=8):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    material_map = {}
    mat_counter = 0
    shapes = []

    for shape in root.findall(".//shape"):
        mesh = shape.find("string").attrib["value"]
        ref = shape.find("ref").attrib["id"]

        if ref not in material_map:
            material_map[ref] = mat_counter
            mat_counter += 1

        shapes.append({"mesh": mesh, "mat": material_map[ref]})

    mat_emb = MaterialEmbedding()

    node_features = []
    node_types = []
    node_names = []
    centroids = []

    # ---------- Object nodes ----------
    for s in shapes:
        z_pc, centroid, bbox = load_z_pc(s["mesh"])
        mat_vec = mat_emb(torch.tensor(s["mat"]))
        type_vec = node_type_onehot(0)
        feat = torch.cat([
    z_pc,
    mat_vec,
    centroid,
    bbox,
    type_vec,
    FREQ_NORM
])

        node_features.append(feat)
        node_types.append(0)
        node_names.append(s["mesh"])
        centroids.append(centroid)

    # ---------- TX node ----------
    tx_index = len(node_features)
    
    tx_type_vec = node_type_onehot(1)  # TX = type 1
    
    node_features.append(torch.cat([
    torch.zeros(Z_PC_DIM),
    torch.zeros(MAT_EMB_DIM),
    tx_pos,
    torch.zeros(3),
    node_type_onehot(1),
    FREQ_NORM
]))
    
    node_types.append(1)
    node_names.append("TX")
    centroids.append(tx_pos)
    
    # ---------- RX node ----------
    rx_index = len(node_features)
    
    rx_type_vec = node_type_onehot(2)  # RX = type 2
    
    node_features.append(torch.cat([
    torch.zeros(Z_PC_DIM),
    torch.zeros(MAT_EMB_DIM),
    rx_pos,
    torch.zeros(3),
    node_type_onehot(2),
    FREQ_NORM
]))
    
    node_types.append(2)
    node_names.append("RX")
    centroids.append(rx_pos)
    
    # ---------- Stack tensors ----------
    x = torch.stack(node_features).float()
    centroids = torch.stack(centroids)
    
    # ---------- Edge containers ----------
    edge_index = []
    edge_attr = []

    def add_edge(i, j):
        dvec = centroids[j] - centroids[i]
        dist = torch.norm(dvec)
        unit = dvec / (dist + 1e-6)
        edge_index.append([i, j])
        edge_attr.append(torch.cat([dist.view(1), unit]))

    num_obj = len(shapes)

    # ---------- Object ↔ Object ----------
    if edge_mode == "full":
        for i in range(num_obj):
            for j in range(num_obj):
                if i != j:
                    add_edge(i, j)

    elif edge_mode == "knn":
        for i in range(num_obj):
            dists = torch.norm(centroids[:num_obj] - centroids[i], dim=1)
            knn = torch.argsort(dists)[1:k_obj+1]
            for j in knn:
                add_edge(i, j)
                add_edge(j, i)

    # ---------- TX / RX ↔ Objects ----------
    for src in [tx_index, rx_index]:
        src_centroid = centroids[src]

        if edge_mode == "full":
            for j in range(num_obj):
                add_edge(src, j)
                add_edge(j, src)

        elif edge_mode == "knn":
            dists = torch.norm(centroids[:num_obj] - src_centroid, dim=1)
            knn = torch.argsort(dists)[:k_tr]
            for j in knn:
                add_edge(src, j)
                add_edge(j, src)

    #---------- TX ↔ RX ----------
    add_edge(tx_index, rx_index)
    add_edge(rx_index, tx_index)

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.stack(edge_attr).float()
    )

    data.node_type = torch.tensor(node_types)
    data.node_name = node_names
    data.tx_index = tx_index
    data.rx_index = rx_index

    return data


def visualize_graph(data, html_out):

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#0f0f0f",
        font_color="white",
        directed=True
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -2500,
          "springLength": 120,
          "springConstant": 0.02,
          "damping": 0.9
        }
      }
    }
    """)

    for i, name in enumerate(data.node_name):
        t = int(data.node_type[i])
        color, size = (
            ("red", 40) if t == 1 else
            ("lime", 40) if t == 2 else
            ("skyblue", 22)
        )
        net.add_node(i, label=name.split("/")[-1], color=color, size=size)

    ei = data.edge_index.numpy()
    ea = data.edge_attr.numpy()

    for k in range(ei.shape[1]):
        src, dst = int(ei[0, k]), int(ei[1, k])
        dist = float(ea[k, 0])
        net.add_edge(src, dst, title=f"dist={dist:.2f}")

    net.write_html(html_out, open_browser=False)
    print(f"Visualization saved → {html_out}")


if __name__ == "__main__":

    data = build_graph(
        XML_FILE,
        TX_POS,
        RX_POS,
        edge_mode=EDGE_MODE,
        k_obj=K_OBJ,
        k_tr=K_TR
    )

    print("Graph built")
    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Node feature dim:", data.x.shape[1])
    print("Edge feature dim:", data.edge_attr.shape[1])

    visualize_graph(data, HTML_OUT)