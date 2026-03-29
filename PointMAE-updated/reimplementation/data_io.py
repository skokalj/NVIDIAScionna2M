"""
02_data_io.py - Data I/O Utilities for Point-MAE

This module handles loading point cloud data from various formats:
- .npy (NumPy binary)
- .txt (Text files, space or comma separated)
- .h5 (HDF5 format)

Point clouds are loaded as numpy arrays of shape (N, 3) or (N, 6).
Only XYZ coordinates are used by Point-MAE (no normals required).
"""

import os
import h5py
import numpy as np


class IO:
    """
    Point cloud I/O class supporting multiple file formats.
    
    Supported formats:
    - .npy: NumPy binary format
    - .txt: Text format (space or comma separated)
    - .h5: HDF5 format with 'data' key
    
    Usage:
        points = IO.get('path/to/pointcloud.npy')
    """
    
    @classmethod
    def get(cls, file_path):
        """
        Load point cloud from file.
        
        Args:
            file_path: Path to point cloud file
            
        Returns:
            numpy.ndarray: Point cloud of shape (N, 3) or (N, D)
        """
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)
    
    @classmethod
    def _read_npy(cls, file_path):
        """Load from NumPy binary format."""
        return np.load(file_path)
    
    @classmethod
    def _read_txt(cls, file_path):
        """
        Load from text format.
        Handles both space-separated and comma-separated files.
        """
        # Try comma-separated first (ModelNet format)
        try:
            return np.loadtxt(file_path, delimiter=',')
        except:
            # Fall back to space-separated
            return np.loadtxt(file_path)
    
    @classmethod
    def _read_h5(cls, file_path):
        """Load from HDF5 format."""
        with h5py.File(file_path, 'r') as f:
            return f['data'][()]
    
    @classmethod
    def save_npy(cls, file_path, data):
        """Save point cloud to NumPy format."""
        np.save(file_path, data)
    
    @classmethod
    def save_txt(cls, file_path, data, delimiter=' '):
        """Save point cloud to text format."""
        np.savetxt(file_path, data, delimiter=delimiter)


def pc_normalize(pc):
    """
    Normalize point cloud to unit sphere centered at origin.
    
    Args:
        pc: numpy.ndarray of shape (N, 3) or (N, D)
        
    Returns:
        Normalized point cloud
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample_numpy(point, npoint):
    """
    Farthest Point Sampling (FPS) in pure NumPy.
    Used for preprocessing when CUDA is not available.
    
    Args:
        point: numpy.ndarray of shape (N, D)
        npoint: Number of points to sample
        
    Returns:
        Sampled point cloud of shape (npoint, D)
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    point = point[centroids]
    return point


def random_sample(pc, num):
    """
    Randomly sample points from point cloud.
    
    Args:
        pc: numpy.ndarray of shape (N, D)
        num: Number of points to sample
        
    Returns:
        Sampled point cloud of shape (num, D)
    """
    permutation = np.random.permutation(pc.shape[0])
    pc = pc[permutation[:num]]
    return pc


if __name__ == '__main__':
    # Test IO functionality
    print("Testing IO module...")
    
    # Create test point cloud
    test_pc = np.random.randn(1024, 3).astype(np.float32)
    
    # Test save and load npy
    IO.save_npy('/tmp/test_pc.npy', test_pc)
    loaded_pc = IO.get('/tmp/test_pc.npy')
    assert np.allclose(test_pc, loaded_pc), "NPY save/load failed"
    print("NPY format: OK")
    
    # Test save and load txt
    IO.save_txt('/tmp/test_pc.txt', test_pc)
    loaded_pc = IO.get('/tmp/test_pc.txt')
    assert np.allclose(test_pc, loaded_pc, atol=1e-5), "TXT save/load failed"
    print("TXT format: OK")
    
    # Test normalization
    normalized = pc_normalize(test_pc)
    assert np.abs(np.mean(normalized, axis=0)).max() < 1e-5, "Normalization center failed"
    assert np.max(np.sqrt(np.sum(normalized**2, axis=1))) <= 1.0 + 1e-5, "Normalization scale failed"
    print("Normalization: OK")
    
    # Test FPS
    sampled = farthest_point_sample_numpy(test_pc, 256)
    assert sampled.shape == (256, 3), "FPS shape failed"
    print("FPS: OK")
    
    # Test random sample
    sampled = random_sample(test_pc, 512)
    assert sampled.shape == (512, 3), "Random sample shape failed"
    print("Random sample: OK")
    
    print("\nAll IO tests passed!")
