from torch_geometric.loader import DataLoader
from utils.config import load_config
from data.graph_dataset import SceneGraphDataset
from visualize import visualize_graph
import sys


cfg = load_config("configs/default.yaml")

indices = list(range(1))
dataset = SceneGraphDataset(cfg, indices)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in loader:
    print("Graphs:", batch.num_graphs)
    print("x:", batch.x.shape)
    print("y:", batch.y.shape)


graph = dataset[0]    
print(graph)
visualize_graph(graph)