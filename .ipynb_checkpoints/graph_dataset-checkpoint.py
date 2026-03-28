import torch
from torch_geometric.data import Dataset
from data.dataset_loader import load_sample
from data.graph_builder import build_graph
import numpy as np


class SceneGraphDataset(Dataset):
    def __init__(self, cfg, sample_indices):
        super().__init__()
        self.cfg = cfg
        self.sample_indices = sample_indices
        self.dataset_path = cfg["dataset"]["path"]

        self.z_pc = torch.from_numpy(
            np.load("data/rooms/z_pc.npy")
        ).float()

    def len(self):
        return len(self.sample_indices)

    def get(self, idx):
        i = self.sample_indices[idx]

        sample = load_sample(
            self.dataset_path,
            i,
            self.cfg["scene"]["xml_root"]
        )

        scale = float(self.cfg["frequency"]["scale"])
        freq_norm = torch.tensor([sample["freq"] / scale])

        graph = build_graph(
            cfg=self.cfg,
            xml_path=sample["xml"],
            tx_pos=sample["tx"],
            rx_pos=sample["rx"],
            freq_norm=freq_norm,
        )

        num_nodes = graph.x.shape[0]

        z_pc = self.z_pc[:num_nodes]

        graph.x = torch.cat(
            [z_pc, graph.x],
            dim=1
        )

        graph.y = sample["tab"]
        return graph