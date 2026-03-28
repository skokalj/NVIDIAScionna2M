import os
import os
GRAPH_ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
POINTCLOUD_DIR = os.path.join(GRAPH_ROOT, "data/rooms/pointclouds")
POINT_MAE_ROOT = "/home/hafeez/Point-MAE"
sys.path.insert(0, POINT_MAE_ROOT)

from models.build import MODELS
from utils.config import cfg_from_yaml_file


OUTPUT_FILE = os.path.join(GRAPH_ROOT, "data/rooms/z_pc.npy")

CKPT_PATH = "/home/nvidia/ckpt-epoch-300.pth"
CONFIG_PATH = "cfgs/pretrain.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GeometryProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(384, 64)
    def forward(self, x):
        return self.proj(x)

def load_pointmae():
    cwd = os.getcwd()
    os.chdir(POINT_MAE_ROOT)
    cfg = cfg_from_yaml_file(CONFIG_PATH)
    model = MODELS.build(cfg.model).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    os.chdir(cwd)
    return model

def load_pointcloud(path):
    pc = np.loadtxt(path, delimiter=",").astype(np.float32)
    pc = pc[:, :3]
    return torch.from_numpy(pc).unsqueeze(0).to(DEVICE)

def main():
    model = load_pointmae()
    projector = GeometryProjector().to(DEVICE)
    projector.eval()

    files = sorted(f for f in os.listdir(POINTCLOUD_DIR) if f.endswith(".txt"))
    z_pc_list = []

    with torch.no_grad():
        for fname in tqdm(files):
            pc = load_pointcloud(os.path.join(POINTCLOUD_DIR, fname))
            neighborhood, center = model.group_divider(pc)
            x_vis, _ = model.MAE_encoder(neighborhood, center)
            global_mean = x_vis.mean(dim=1)
            z_pc = projector(global_mean)
            z_pc_list.append(z_pc.cpu().numpy()[0])

    z_pc = np.stack(z_pc_list)
    np.save(OUTPUT_FILE, z_pc)

if __name__ == "__main__":
    main()