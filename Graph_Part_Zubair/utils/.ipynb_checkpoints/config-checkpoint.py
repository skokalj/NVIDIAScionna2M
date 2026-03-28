import yaml

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ["dataset", "scene", "frequency", "edges", "node_features"]:
        if key not in cfg:
            raise ValueError(f"Missing config section: {key}")

    return cfg