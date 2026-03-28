import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from graph_dataset import SceneGraphDataset

def draw_feature_table(ax, x, y, width, labels, colors, title=None, row_height=0.03, text_pad=0.005):
    
    y0 = y
    rows = labels.copy()
    colors = colors.copy()

    if title is not None:
        # Insert title at the top row
        rows.insert(0, title)
        colors.insert(0, "#CCCCCC")  

    for row_label, color in zip(rows, colors):
        # Draw rectangle
        ax.add_patch(plt.Rectangle((x, y0), width, row_height, facecolor=color, edgecolor="black"))
        # Draw text
        ax.text(
            x + width/2,
            y0 + row_height/2,
            row_label,
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )
        y0 -= row_height + text_pad  # move down for next row

def visualize_graph_structured_features(data, out_png="scene_graph_structured.png"):
   
    data.node_names = [name.replace("meshes/", "").replace(".ply", "") for name in data.node_names]
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
            if len(data.node_names[i]) <= 8:
                example_nodes["other"] = i 
        if len(example_nodes) == 3:
            break

    seg_labels = ["z_pc (128)", "material (32)", "centroid (3)", "bbox (3)", "node_type (3)", "frequency (1)"]
    seg_colors = ["#aec6cf", "#ffb347", "#b5e7a0", "#ff6961", "#c5b0d5", "#d7bfae"]

     # Define per-node offsets (x, y) relative to the node position
    node_offsets = {
    "tx": (-0.16, -0.06),      #it is other
    "rx": (0.048, 0.05),       #  it is tx
    "other": (0.041, 0.088)    # it is rx
}
    
    for role, i in example_nodes.items():
        x0, y0 = pos[int(i)]
        offset_x, offset_y = node_offsets[role]
    
        draw_feature_table(
            ax,
            x0 + offset_x,
            y0 + offset_y,
            width=0.18,
            labels=seg_labels,
            colors=seg_colors,
            title=data.node_names[i],
            row_height=0.03
        )

    # Edge features for the first edge
    src,dst = data.edge_index[:,0]
    src,dst = int(src.item()), int(dst.item())
    x0,y0 = pos[src]
    x1,y1 = pos[dst]
    xm,ym = (x0+x1)/2, (y0+y1)/2

    edge_labels = ["distance","dx","dy","dz"]
    edge_colors = ["#ffd27f", "#fff79a", "#b0e0e6", "#dda0dd"]

    draw_feature_table(
        ax,
        xm + 0.12,
        ym + 0.05,
        width=0.12,
        labels=edge_labels,
        colors=edge_colors,
        title="Edge attr",
        row_height=0.025
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