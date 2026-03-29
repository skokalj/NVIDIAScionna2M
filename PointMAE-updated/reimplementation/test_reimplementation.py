#!/usr/bin/env python
"""
test_reimplementation.py - Test Script for Point-MAE Reimplementation

This script verifies that all components work correctly:
1. Data I/O utilities
2. Transforms
3. Loss functions
4. Model forward pass (Point_MAE and PointTransformer)
5. Configuration loading

Run with: python test_reimplementation.py
"""

import os
import sys
import torch
import numpy as np

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_data_io():
    """Test data I/O utilities."""
    print("\n" + "="*60)
    print("Testing Data I/O...")
    print("="*60)
    
    from data_io import IO, pc_normalize, farthest_point_sample_numpy, random_sample
    
    # Create test point cloud
    test_pc = np.random.randn(1024, 3).astype(np.float32)
    
    # Test save and load npy
    IO.save_npy('/tmp/test_pc.npy', test_pc)
    loaded_pc = IO.get('/tmp/test_pc.npy')
    assert np.allclose(test_pc, loaded_pc), "NPY save/load failed"
    print("  NPY format: OK")
    
    # Test normalization
    normalized = pc_normalize(test_pc)
    assert np.abs(np.mean(normalized, axis=0)).max() < 1e-5, "Normalization center failed"
    print("  Normalization: OK")
    
    # Test FPS
    sampled = farthest_point_sample_numpy(test_pc, 256)
    assert sampled.shape == (256, 3), "FPS shape failed"
    print("  FPS (numpy): OK")
    
    # Test random sample
    sampled = random_sample(test_pc, 512)
    assert sampled.shape == (512, 3), "Random sample shape failed"
    print("  Random sample: OK")
    
    print("Data I/O: ALL TESTS PASSED")
    return True


def test_transforms():
    """Test data augmentation transforms."""
    print("\n" + "="*60)
    print("Testing Transforms...")
    print("="*60)
    
    from transforms import (
        PointcloudScaleAndTranslate,
        PointcloudRotate,
        PointcloudJitter,
        PointcloudScale,
        PointcloudTranslate,
        Compose,
        get_train_transforms
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pc = torch.randn(4, 1024, 3).to(device)
    
    transforms_to_test = [
        ('PointcloudScaleAndTranslate', PointcloudScaleAndTranslate()),
        ('PointcloudRotate', PointcloudRotate()),
        ('PointcloudJitter', PointcloudJitter()),
        ('PointcloudScale', PointcloudScale()),
        ('PointcloudTranslate', PointcloudTranslate()),
    ]
    
    for name, transform in transforms_to_test:
        pc_copy = pc.clone()
        result = transform(pc_copy)
        assert result.shape == pc.shape, f"{name} changed shape"
        print(f"  {name}: OK")
    
    # Test compose
    train_transforms = get_train_transforms()
    result = train_transforms(pc.clone())
    assert result.shape == pc.shape
    print("  Compose: OK")
    
    print("Transforms: ALL TESTS PASSED")
    return True


def test_losses():
    """Test loss functions."""
    print("\n" + "="*60)
    print("Testing Loss Functions...")
    print("="*60)
    
    from losses import ChamferDistanceL1, ChamferDistanceL2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    xyz1 = torch.randn(4, 32, 3).to(device)
    xyz2 = torch.randn(4, 32, 3).to(device)
    
    # Test ChamferDistanceL2
    cd_l2 = ChamferDistanceL2().to(device)
    loss_l2 = cd_l2(xyz1, xyz2)
    print(f"  ChamferDistanceL2: {loss_l2.item():.4f}")
    
    # Test ChamferDistanceL1
    cd_l1 = ChamferDistanceL1().to(device)
    loss_l1 = cd_l1(xyz1, xyz2)
    print(f"  ChamferDistanceL1: {loss_l1.item():.4f}")
    
    # Test with identical point clouds (should be ~0)
    loss_same = cd_l2(xyz1, xyz1)
    assert loss_same.item() < 1e-5, "Same point cloud should have ~0 loss"
    print(f"  Same point cloud loss: {loss_same.item():.6f} (should be ~0)")
    
    # Test gradient flow
    xyz1_grad = xyz1.clone().requires_grad_(True)
    loss = cd_l2(xyz1_grad, xyz2)
    loss.backward()
    assert xyz1_grad.grad is not None, "Gradients should flow"
    print("  Gradient flow: OK")
    
    print("Losses: ALL TESTS PASSED")
    return True


def test_model():
    """Test model forward pass."""
    print("\n" + "="*60)
    print("Testing Models...")
    print("="*60)
    
    from model import Point_MAE, PointTransformer, build_model, HAS_POINTNET2, HAS_KNN_CUDA
    from easydict import EasyDict
    
    print(f"  HAS_POINTNET2: {HAS_POINTNET2}")
    print(f"  HAS_KNN_CUDA: {HAS_KNN_CUDA}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test Point_MAE
    print("\n  Testing Point_MAE (pretraining model)...")
    config = EasyDict({
        'NAME': 'Point_MAE',
        'group_size': 32,
        'num_group': 64,
        'loss': 'cdl2',
        'transformer_config': EasyDict({
            'mask_ratio': 0.6,
            'mask_type': 'rand',
            'trans_dim': 384,
            'encoder_dims': 384,
            'depth': 12,
            'drop_path_rate': 0.1,
            'num_heads': 6,
            'decoder_depth': 4,
            'decoder_num_heads': 6,
        }),
    })
    
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params / 1e6:.2f}M")
    
    x = torch.randn(2, 1024, 3).to(device)
    loss = model(x)
    print(f"    Input: {x.shape}, Loss: {loss.item():.4f}")
    print("    Point_MAE: OK")
    
    # Test PointTransformer
    print("\n  Testing PointTransformer (finetuning model)...")
    ft_config = EasyDict({
        'NAME': 'PointTransformer',
        'trans_dim': 384,
        'depth': 12,
        'drop_path_rate': 0.1,
        'cls_dim': 40,
        'num_heads': 6,
        'group_size': 32,
        'num_group': 64,
        'encoder_dims': 384,
    })
    
    ft_model = build_model(ft_config).to(device)
    ft_params = sum(p.numel() for p in ft_model.parameters())
    print(f"    Parameters: {ft_params / 1e6:.2f}M")
    
    logits = ft_model(x)
    print(f"    Input: {x.shape}, Output: {logits.shape}")
    print("    PointTransformer: OK")
    
    print("\nModels: ALL TESTS PASSED")
    return True


def test_config():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("Testing Configuration...")
    print("="*60)
    
    from config import cfg_from_yaml_file
    
    # Test pretrain config
    config = cfg_from_yaml_file('cfgs/pretrain.yaml')
    assert config.model.NAME == 'Point_MAE'
    assert config.total_bs == 128
    assert config.max_epoch == 300
    print("  Pretrain config: OK")
    
    # Test finetune config
    config = cfg_from_yaml_file('cfgs/finetune_modelnet.yaml')
    assert config.model.NAME == 'PointTransformer'
    assert config.model.cls_dim == 40
    print("  Finetune config: OK")
    
    print("Configuration: ALL TESTS PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Point-MAE Reimplementation Test Suite")
    print("#"*60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    all_passed = True
    
    try:
        all_passed &= test_data_io()
    except Exception as e:
        print(f"Data I/O tests FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_transforms()
    except Exception as e:
        print(f"Transforms tests FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_losses()
    except Exception as e:
        print(f"Losses tests FAILED: {e}")
        all_passed = False
    
    try:
        all_passed &= test_model()
    except Exception as e:
        print(f"Model tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_config()
    except Exception as e:
        print(f"Config tests FAILED: {e}")
        all_passed = False
    
    print("\n" + "#"*60)
    if all_passed:
        print("# ALL TESTS PASSED!")
        print("#")
        print("# The reimplementation is ready for training.")
        print("# ")
        print("# To start pretraining (8 GPUs):")
        print("#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \\")
        print("#       --nproc_per_node=8 main.py --config cfgs/pretrain.yaml \\")
        print("#       --launcher pytorch --exp_name pretrain_8gpu")
    else:
        print("# SOME TESTS FAILED!")
        print("# Please check the errors above.")
    print("#"*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
