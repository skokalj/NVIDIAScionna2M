import os
import numpy as np
import torch

def load_sample(dataset_path, idx, xml_root):
    tx, rx, xml, freq, tab = np.load(dataset_path, allow_pickle=True)

    xml_name = os.path.basename(xml[idx])
    xml_path = os.path.join(xml_root, xml_name)

    return {
        "tx": torch.tensor(tx[idx], dtype=torch.float),
        "rx": torch.tensor(rx[idx], dtype=torch.float),
        "xml": xml_path,
        "freq": float(freq),
        "tab": torch.tensor(tab[idx], dtype=torch.float),
    }