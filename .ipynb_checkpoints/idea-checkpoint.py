import numpy as np

def deep_print(obj, name="root", indent=0, max_items=5):
    pad = "  " * indent
    print(f"{pad}{name}: type={type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{pad}  shape={obj.shape}, dtype={obj.dtype}")
        if obj.dtype == object:
            for i, v in enumerate(obj[:max_items]):
                deep_print(v, f"{name}[{i}]", indent + 1)
        else:
            print(f"{pad}  preview={obj.flat[:5]}")

    elif isinstance(obj, list):
        print(f"{pad}  len={len(obj)}")
        for i, v in enumerate(obj[:max_items]):
            deep_print(v, f"{name}[{i}]", indent + 1)

    elif isinstance(obj, dict):
        print(f"{pad}  keys={list(obj.keys())}")
        for k, v in obj.items():
            deep_print(v, f"{name}['{k}']", indent + 1)

    else:
        print(f"{pad}  value={obj}")

# Load dataset
data = np.load("dataset.npy", allow_pickle=True)

print("\n=== FULL DATASET STRUCTURE ===")
deep_print(data)