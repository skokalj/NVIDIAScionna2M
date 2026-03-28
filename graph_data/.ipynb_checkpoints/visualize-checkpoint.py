import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from graph_dataset import SceneGraphDataset

def draw_feature_column(ax, x, y, width, heights, labels, colors, text_pad=0.005):
    y0 = y
    for h, lab, col in zip(heights, labels, colors):
        ax.add_patch(plt.Rectangle((x, y0), width, h, facecolor=col, alpha=0.85))
        ax.text(
            x + width/2,
            y0 + h/2,
            lab,
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )
        y0 += h + text_pad

def visualize_graph_structured_features(data, out_png="scene_graph_structured.png"):
    G = to_networkx(data, edge_attrs=None, node_attrs=None)
    pos = nx.spring_layout(G, seed=42, k=1.6)

    fig, ax = plt.subplots(figsize=(15,15))

    # Node colors and sizes
    colors = ["lightblue" if t==0 else "red" if t==1 else "green" for t in data.node_type.tolist()]
    sizes = [600 if t==0 else 900 for t in data.node_type.tolist()]

    nx.draw(G, pos, node_color=colors, node_size=sizes, edge_color="gray", width=0.1, alpha=0.6, ax=ax)
    labels = {i: data.node_names[i] for i in range(data.num_nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

    # Pick example nodes: one tx, one rx, one other
    example_nodes = {}
    for i, t in enumerate(data.node_type.tolist()):
        if t == 0 and "tx" not in example_nodes:
            example_nodes["tx"] = i
        elif t == 1 and "rx" not in example_nodes:
            example_nodes["rx"] = i
        elif t not in (0,1) and "other" not in example_nodes:
            example_nodes["other"] = i
        if len(example_nodes) == 3:
            break

    seg_labels = ["z_pc (128)", "material (32)", "centroid (3)", "bbox (3)", "node_type (3)", "frequency (1)"]
    seg_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    seg_heights = [0.03]*6

    # Draw feature columns for example nodes
    for role, i in example_nodes.items():
        x0, y0 = pos[int(i)]
        col_offset_x = 0.1 if x0 < 0 else 0.12
        col_offset_y = 0.1

        # Node name
        ax.text(
            x0,
            y0 + col_offset_y,
            data.node_names[i],
            fontsize=12,
            ha="center",
            va="bottom",
            weight="bold"
        )

        draw_feature_column(
            ax,
            x0 + col_offset_x,
            y0 - col_offset_y,
            0.06,
            seg_heights,
            seg_labels,
            seg_colors
        )

    # Edge features for the first edge
    src,dst = data.edge_index[:,0]
    src,dst = int(src.item()), int(dst.item())
    x0,y0 = pos[src]
    x1,y1 = pos[dst]
    xm,ym = (x0+x1)/2, (y0+y1)/2

    edge_labels = ["distance","dx","dy","dz"]
    edge_heights = [0.015]*4
    edge_colors = ["orange","yellow","cyan","magenta"]

    edge_offset_x = 0.12
    edge_offset_y = 0.08

    ax.text(
        xm + edge_offset_x,
        ym + edge_offset_y,
        "edge features",
        fontsize=12,
        ha="left",
        va="bottom",
        weight="bold"
    )

    draw_feature_column(
        ax,
        xm + edge_offset_x + 0.05,
        ym - edge_offset_y,
        0.06,
        edge_heights,
        edge_labels,
        edge_colors
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved visualization to {out_png}")

if __name__=="__main__":
    cfg={
        "dataset":{"path":"/data/hafeez/graphdata/rooms_update/dataset.npy"},
        "node_features":{
            "material":{"enabled":True,"vocab_size":5,"dim":10},
            "centroid":True,
            "bbox":True,
            "node_type":{"enabled":True,"dim":3},
            "frequency":{"enabled":True}
        },
        "frequency":{"scale":1.0}
    }
    dataset = SceneGraphDataset(cfg, sample_indices=[0])
    graph = dataset[0]
    visualize_graph_structured_features(graph)