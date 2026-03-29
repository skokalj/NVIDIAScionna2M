'''
CustomDataset.py - Dataset loader for custom PLY-converted data

This dataset loader is designed to work with data preprocessed by
tools/preprocess_custom_dataset.py. It follows the same structure as ModelNet
but with configurable dataset name prefix.
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


@DATASETS.register_module()
class CustomDataset(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.dataset_name = getattr(config, 'DATASET_NAME', 'custom')
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        # Load category names
        self.catfile = os.path.join(self.root, f'{self.dataset_name}_shape_names.txt')
        if not os.path.exists(self.catfile):
            raise FileNotFoundError(f"Category file not found: {self.catfile}")
        
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        
        # Verify number of categories
        if len(self.cat) != self.num_category:
            print_log(f'Warning: NUM_CATEGORY ({self.num_category}) does not match '
                     f'actual categories ({len(self.cat)}). Using {len(self.cat)}.', 
                     logger='CustomDataset')
            self.num_category = len(self.cat)

        # Load shape IDs for this split
        shape_ids = {}
        train_file = os.path.join(self.root, f'{self.dataset_name}_train.txt')
        test_file = os.path.join(self.root, f'{self.dataset_name}_test.txt')
        
        if os.path.exists(train_file):
            shape_ids['train'] = [line.rstrip() for line in open(train_file)]
        else:
            shape_ids['train'] = []
            
        if os.path.exists(test_file):
            shape_ids['test'] = [line.rstrip() for line in open(test_file)]
        else:
            shape_ids['test'] = []

        assert (split == 'train' or split == 'test'), f"Invalid split: {split}"
        
        # Build data paths
        shape_names = []
        for x in shape_ids[split]:
            # Extract class name from sample name (e.g., "table_0001" -> "table")
            parts = x.rsplit('_', 1)
            if len(parts) == 2:
                shape_names.append(parts[0])
            else:
                shape_names.append(x)
        
        self.datapath = [
            (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') 
            for i in range(len(shape_ids[split]))
        ]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='CustomDataset')

        # Set up cache path
        if self.uniform:
            self.save_path = os.path.join(
                self.root, 
                f'{self.dataset_name}_{split}_{self.npoints}pts_fps.dat'
            )
        else:
            self.save_path = os.path.join(
                self.root, 
                f'{self.dataset_name}_{split}_{self.npoints}pts.dat'
            )

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, 
                         logger='CustomDataset')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger='CustomDataset')
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
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
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
        return 'CustomDataset', 'sample', (current_points, label)
    
    def get_class_names(self):
        """Return list of class names."""
        return self.cat
    
    def get_num_classes(self):
        """Return number of classes."""
        return self.num_category
