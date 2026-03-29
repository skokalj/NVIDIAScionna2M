"""
03_datasets.py - Dataset Classes for Point-MAE

This module provides dataset classes for:
- ShapeNet55: Pretraining dataset (52,470 shapes, 55 categories)
- ModelNet40: Classification dataset (12,311 shapes, 40 categories)
- ScanObjectNN: Real-world scanned objects dataset

Data Pipeline:
1. Load point cloud from file (.npy, .txt, .h5)
2. Random/FPS sampling to fixed number of points
3. Normalize to unit sphere
4. Apply data augmentation (during training)
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data_io import IO, pc_normalize, farthest_point_sample_numpy, random_sample


class ShapeNet(Dataset):
    """
    ShapeNet55 Dataset for pretraining.
    
    Directory structure:
    data/ShapeNet55-34/
    ├── shapenet_pc/
    │   ├── 02691156-xxx.npy  (8192 points each)
    │   └── ...
    └── ShapeNet-55/
        ├── train.txt
        └── test.txt
    
    Args:
        config: EasyDict with:
            - DATA_PATH: Path to ShapeNet-55 folder
            - PC_PATH: Path to shapenet_pc folder
            - N_POINTS: Points per file (8192)
            - subset: 'train' or 'test'
            - npoints: Points to sample (1024)
    """
    
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.whole = config.get('whole', False)
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        print(f'[DATASET] Sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')
        
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print(f'[DATASET] Open file {test_data_list_file}')
            lines = test_lines + lines
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        self.permutation = np.arange(self.npoints)
    
    def pc_norm(self, pc):
        """Normalize point cloud to unit sphere."""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def random_sample(self, pc, num):
        """Randomly sample points."""
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        
        # Sample to target number of points
        data = self.random_sample(data, self.sample_points_num)
        # Normalize
        data = self.pc_norm(data)
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        return sample['taxonomy_id'], sample['model_id'], data
    
    def __len__(self):
        return len(self.file_list)


class ModelNet(Dataset):
    """
    ModelNet40 Dataset for classification.
    
    Directory structure:
    data/ModelNet/modelnet40_normal_resampled/
    ├── modelnet40_shape_names.txt
    ├── modelnet40_train.txt
    ├── modelnet40_test.txt
    ├── airplane/
    │   ├── airplane_0001.txt  (N x 6: xyz + normals)
    │   └── ...
    └── ...
    
    First run: Processes .txt files and caches to .dat
    Subsequent runs: Loads from .dat cache
    
    Args:
        config: EasyDict with:
            - DATA_PATH: Path to modelnet40_normal_resampled
            - N_POINTS: Points to sample (8192)
            - USE_NORMALS: Whether to use normals (False for Point-MAE)
            - NUM_CATEGORY: 10 or 40
            - subset: 'train' or 'test'
    """
    
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.subset = config.subset
        split = config.subset
        
        # Load category names
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        
        # Load shape IDs
        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in 
                                  open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in 
                                 open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in 
                                  open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in 
                                 open(os.path.join(self.root, 'modelnet40_test.txt'))]
        
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], 
                          os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt')
                         for i in range(len(shape_ids[split]))]
        
        print(f'The size of {split} data is {len(self.datapath)}')
        
        # Cache file path
        if self.uniform:
            self.save_path = os.path.join(
                self.root, 
                f'modelnet{self.num_category}_{split}_{self.npoints}pts_fps.dat'
            )
        else:
            self.save_path = os.path.join(
                self.root,
                f'modelnet{self.num_category}_{split}_{self.npoints}pts.dat'
            )
        
        # Process or load cached data
        if self.process_data:
            if not os.path.exists(self.save_path):
                print(f'Processing data {self.save_path} (only running in the first time)...')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                    
                    if self.uniform:
                        point_set = farthest_point_sample_numpy(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]
                    
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print(f'Load processed data from {self.save_path}...')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
    
    def __len__(self):
        return len(self.datapath)
    
    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            
            if self.uniform:
                point_set = farthest_point_sample_numpy(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
        
        # Normalize XYZ
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        # Discard normals if not using them
        if not self.use_normals:
            point_set = point_set[:, 0:3]
        
        return point_set, label[0]
    
    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        
        return 'ModelNet', 'sample', (current_points, label)


class ScanObjectNN(Dataset):
    """
    ScanObjectNN Dataset for classification on real-world scans.
    
    Directory structure:
    data/ScanObjectNN/
    ├── main_split/
    │   ├── training_objectdataset_augmentedrot_scale75.h5
    │   └── test_objectdataset_augmentedrot_scale75.h5
    └── main_split_nobg/
        ├── training_objectdataset.h5
        └── test_objectdataset.h5
    
    Args:
        config: EasyDict with:
            - DATA_PATH: Path to h5 file
            - subset: 'train' or 'test'
            - N_POINTS: Points to sample
    """
    
    def __init__(self, config):
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        # Load h5 file
        import h5py
        h5_file = h5py.File(config.DATA_PATH, 'r')
        
        self.points = np.array(h5_file['data']).astype(np.float32)
        self.labels = np.array(h5_file['label']).astype(np.int64)
        
        h5_file.close()
        
        print(f'[ScanObjectNN] {self.subset}: {len(self.points)} samples loaded')
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, index):
        point_set = self.points[index][:self.npoints]
        label = self.labels[index]
        
        # Normalize
        point_set = pc_normalize(point_set)
        point_set = torch.from_numpy(point_set).float()
        
        return 'ScanObjectNN', 'sample', (point_set, label)


# Registry for datasets
DATASETS = {
    'ShapeNet': ShapeNet,
    'ModelNet': ModelNet,
    'ScanObjectNN': ScanObjectNN,
}


def build_dataset(config, default_args=None):
    """
    Build dataset from config.
    
    Args:
        config: Dataset config with _base_ and others
        default_args: Additional arguments to merge
        
    Returns:
        Dataset instance
    """
    # Merge base config with others
    from easydict import EasyDict
    
    if hasattr(config, '_base_'):
        base_config = config._base_
    else:
        base_config = config
    
    if default_args is not None:
        for key, val in default_args.items():
            base_config[key] = val
    
    dataset_name = base_config.NAME
    if dataset_name not in DATASETS:
        raise KeyError(f'{dataset_name} is not in the dataset registry')
    
    return DATASETS[dataset_name](base_config)


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset module...")
    
    # Create dummy config for testing
    from easydict import EasyDict
    
    # Test ShapeNet config
    shapenet_config = EasyDict({
        'DATA_PATH': 'data/ShapeNet55-34/ShapeNet-55',
        'PC_PATH': 'data/ShapeNet55-34/shapenet_pc',
        'N_POINTS': 8192,
        'subset': 'train',
        'npoints': 1024,
    })
    
    print(f"ShapeNet config created: {shapenet_config}")
    
    # Test ModelNet config
    modelnet_config = EasyDict({
        'DATA_PATH': 'data/ModelNet/modelnet40_normal_resampled',
        'N_POINTS': 8192,
        'USE_NORMALS': False,
        'NUM_CATEGORY': 40,
        'subset': 'train',
    })
    
    print(f"ModelNet config created: {modelnet_config}")
    print("\nDataset module loaded successfully!")
