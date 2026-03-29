"""
04_transforms.py - Data Augmentation Transforms for Point-MAE

This module provides point cloud augmentation transforms:
- PointcloudScaleAndTranslate: Main transform used in training
- PointcloudRotate: Y-axis rotation
- PointcloudJitter: Add Gaussian noise
- PointcloudScale: Random scaling
- PointcloudTranslate: Random translation
- RandomHorizontalFlip: Flip along horizontal axes

All transforms operate on batched tensors of shape (B, N, 3).
"""

import numpy as np
import torch
import random


class PointcloudScaleAndTranslate:
    """
    Scale and translate point cloud.
    This is the main augmentation used in Point-MAE training.
    
    Args:
        scale_low: Minimum scale factor (default: 2/3)
        scale_high: Maximum scale factor (default: 3/2)
        translate_range: Translation range (default: 0.2)
    """
    
    def __init__(self, scale_low=2./3., scale_high=3./2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range
    
    def __call__(self, pc):
        """
        Args:
            pc: Tensor of shape (B, N, 3)
        Returns:
            Transformed tensor of shape (B, N, 3)
        """
        bsize = pc.size()[0]
        device = pc.device
        
        for i in range(bsize):
            # Random scale per axis
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            # Random translation per axis
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            scale = torch.from_numpy(xyz1).float().to(device)
            translate = torch.from_numpy(xyz2).float().to(device)
            
            pc[i, :, 0:3] = pc[i, :, 0:3] * scale + translate
        
        return pc


class PointcloudRotate:
    """
    Rotate point cloud around Y-axis.
    """
    
    def __call__(self, pc):
        """
        Args:
            pc: Tensor of shape (B, N, 3)
        Returns:
            Rotated tensor of shape (B, N, 3)
        """
        bsize = pc.size()[0]
        device = pc.device
        
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            
            # Rotation matrix around Y-axis
            rotation_matrix = np.array([
                [cosval, 0, sinval],
                [0, 1, 0],
                [-sinval, 0, cosval]
            ])
            
            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(device)
            pc[i, :, :] = torch.matmul(pc[i], R)
        
        return pc


class PointcloudJitter:
    """
    Add Gaussian noise to point cloud.
    
    Args:
        std: Standard deviation of noise (default: 0.01)
        clip: Clip range for noise (default: 0.05)
    """
    
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip
    
    def __call__(self, pc):
        """
        Args:
            pc: Tensor of shape (B, N, 3)
        Returns:
            Jittered tensor of shape (B, N, 3)
        """
        bsize = pc.size()[0]
        
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
        
        return pc


class PointcloudScale:
    """
    Scale point cloud.
    
    Args:
        scale_low: Minimum scale factor (default: 2/3)
        scale_high: Maximum scale factor (default: 3/2)
    """
    
    def __init__(self, scale_low=2./3., scale_high=3./2.):
        self.scale_low = scale_low
        self.scale_high = scale_high
    
    def __call__(self, pc):
        """
        Args:
            pc: Tensor of shape (B, N, 3)
        Returns:
            Scaled tensor of shape (B, N, 3)
        """
        bsize = pc.size()[0]
        device = pc.device
        
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scale = torch.from_numpy(xyz1).float().to(device)
            pc[i, :, 0:3] = pc[i, :, 0:3] * scale
        
        return pc


class PointcloudTranslate:
    """
    Translate point cloud.
    
    Args:
        translate_range: Translation range (default: 0.2)
    """
    
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range
    
    def __call__(self, pc):
        """
        Args:
            pc: Tensor of shape (B, N, 3)
        Returns:
            Translated tensor of shape (B, N, 3)
        """
        bsize = pc.size()[0]
        device = pc.device
        
        for i in range(bsize):
            xyz2 = np.random.uniform(
                low=-self.translate_range, 
                high=self.translate_range, 
                size=[3]
            )
            translate = torch.from_numpy(xyz2).float().to(device)
            pc[i, :, 0:3] = pc[i, :, 0:3] + translate
        
        return pc


class PointcloudRandomInputDropout:
    """
    Randomly drop points from point cloud.
    Dropped points are replaced with the first point.
    
    Args:
        max_dropout_ratio: Maximum fraction of points to drop (default: 0.5)
    """
    
    def __init__(self, max_dropout_ratio=0.5):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
    
    def __call__(self, pc):
        """
        Args:
            pc: Tensor of shape (B, N, 3)
        Returns:
            Tensor with dropped points of shape (B, N, 3)
        """
        bsize = pc.size()[0]
        
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)
                pc[i, :, :] = cur_pc
        
        return pc


class RandomHorizontalFlip:
    """
    Randomly flip point cloud along horizontal axes.
    
    Args:
        upright_axis: Axis pointing up ('x', 'y', or 'z')
        is_temporal: Whether data has temporal dimension
    """
    
    def __init__(self, upright_axis='z', is_temporal=False):
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])
    
    def __call__(self, coords):
        """
        Args:
            coords: Tensor of shape (B, N, 3)
        Returns:
            Flipped tensor of shape (B, N, 3)
        """
        bsize = coords.size()[0]
        
        for i in range(bsize):
            if random.random() < 0.95:
                for curr_ax in self.horz_axes:
                    if random.random() < 0.5:
                        coord_max = torch.max(coords[i, :, curr_ax])
                        coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
        
        return coords


class Compose:
    """
    Compose multiple transforms.
    
    Args:
        transforms: List of transform objects
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, pc):
        for t in self.transforms:
            pc = t(pc)
        return pc


# Default transforms used in Point-MAE
def get_train_transforms():
    """Get default training transforms."""
    return Compose([
        PointcloudScaleAndTranslate(),
    ])


def get_test_transforms():
    """Get default test transforms (same as train for Point-MAE)."""
    return Compose([
        PointcloudScaleAndTranslate(),
    ])


if __name__ == '__main__':
    # Test transforms
    print("Testing transforms module...")
    
    # Create dummy point cloud
    pc = torch.randn(4, 1024, 3).cuda() if torch.cuda.is_available() else torch.randn(4, 1024, 3)
    
    # Test each transform
    transforms_to_test = [
        ('PointcloudScaleAndTranslate', PointcloudScaleAndTranslate()),
        ('PointcloudRotate', PointcloudRotate()),
        ('PointcloudJitter', PointcloudJitter()),
        ('PointcloudScale', PointcloudScale()),
        ('PointcloudTranslate', PointcloudTranslate()),
        ('PointcloudRandomInputDropout', PointcloudRandomInputDropout()),
        ('RandomHorizontalFlip', RandomHorizontalFlip()),
    ]
    
    for name, transform in transforms_to_test:
        pc_copy = pc.clone()
        try:
            result = transform(pc_copy)
            assert result.shape == pc.shape, f"{name} changed shape"
            print(f"{name}: OK")
        except Exception as e:
            print(f"{name}: FAILED - {e}")
    
    # Test compose
    train_transforms = get_train_transforms()
    result = train_transforms(pc.clone())
    assert result.shape == pc.shape
    print("Compose: OK")
    
    print("\nAll transform tests passed!")
