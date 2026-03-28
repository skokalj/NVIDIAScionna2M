
"""
Generate zrs1.npy from rooms_update dataset only.
"""

import numpy as np

# Load rooms_update dataset
with open('/data/hafeez/graphdata/rooms_update/dataset.npy', 'rb') as f:
    loaded_array = np.load(f, allow_pickle=True)

zrs = np.stack(loaded_array[4], axis=0)  # 10000 x 107

# Save to /data/
with open('/data/hafeez/zrs1.npy', 'wb') as f:
    np.save(f, zrs)

print(f"zrs shape: {zrs.shape}")
print(f"Saved to: /data/hafeez/zrs1.npy")