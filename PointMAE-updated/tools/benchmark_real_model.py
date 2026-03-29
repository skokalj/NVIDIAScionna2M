#!/usr/bin/env python
"""
Point-MAE REAL Model Benchmarking Tool

This script benchmarks the ACTUAL Point-MAE architecture (PointTransformer) 
from models/Point_MAE.py - NOT a simplified version.

It measures:
1. Encoder-only forward pass timing (single point cloud)
2. Full classification inference timing
3. Component-wise breakdown (FPS+KNN, Encoder, Transformer, Classification Head)
4. Model architecture visualization

REQUIREMENTS (install in order):
    # Set compiler environment
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    export CFLAGS="-I/usr/local/linux-headers-5.4/include"
    export CPPFLAGS="-I/usr/local/linux-headers-5.4/include"
    
    # Install Chamfer Distance
    cd extensions/chamfer_dist && python setup.py install --user
    
    # Install PointNet++ ops
    pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
    
    # Install KNN CUDA
    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

USAGE:
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
        --config cfgs/finetune_modelnet.yaml \
        --ckpts <checkpoint_path> \
        --n_points 8192 \
        --num_runs 100

Author: Benchmark tool for Point-MAE paper
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the actual Point-MAE codebase
from utils.config import cfg_from_yaml_file
from models.Point_MAE import PointTransformer


# ============================================================================
# Model Architecture Visualization
# ============================================================================

def get_model_tree(model, prefix='', is_last=True, max_depth=10, current_depth=0):
    """Generate a tree-like visualization of model architecture"""
    if current_depth > max_depth:
        return []
    
    lines = []
    connector = '└── ' if is_last else '├── '
    
    module_name = model.__class__.__name__
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get shape info for common layers
    shape_info = ''
    if isinstance(model, nn.Linear):
        shape_info = f' [{model.in_features} → {model.out_features}]'
    elif isinstance(model, nn.Conv1d):
        shape_info = f' [{model.in_channels} → {model.out_channels}, k={model.kernel_size[0]}]'
    elif isinstance(model, nn.BatchNorm1d):
        shape_info = f' [{model.num_features}]'
    elif isinstance(model, nn.LayerNorm):
        shape_info = f' [{model.normalized_shape}]'
    elif isinstance(model, nn.Dropout):
        shape_info = f' [p={model.p}]'
    
    param_info = f' (params: {total_params:,})' if total_params > 0 else ''
    lines.append(f'{prefix}{connector}{module_name}{shape_info}{param_info}')
    
    children = list(model.named_children())
    extension = '    ' if is_last else '│   '
    
    for i, (name, child) in enumerate(children):
        is_last_child = (i == len(children) - 1)
        child_prefix = prefix + extension
        child_connector = '└── ' if is_last_child else '├── '
        child_lines = get_model_tree(child, child_prefix, is_last_child, max_depth, current_depth + 1)
        
        if child_lines:
            first_line = child_lines[0]
            parts = first_line.split(connector if is_last_child else '├── ')
            if len(parts) > 1:
                child_lines[0] = f'{child_prefix}{child_connector}({name}): {parts[1]}'
        
        lines.extend(child_lines)
    
    return lines


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# Timing Utilities
# ============================================================================

def warmup_gpu(model, input_tensor, num_warmup=50):
    """Warmup GPU to ensure stable timing"""
    print(f"  Warming up GPU ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()


def benchmark_forward_pass(model, input_tensor, num_runs=100, description="Forward pass"):
    """Benchmark forward pass with CUDA synchronization for accurate timing"""
    times = []
    
    print(f"  Benchmarking {description} ({num_runs} runs)...")
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(input_tensor)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 20 == 0:
                print(f"    Run {i+1}/{num_runs}: {times[-1]:.2f} ms")
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'throughput_per_sec': float(1000 / np.mean(times)),
        'all_times_ms': [float(t) for t in times]
    }


def benchmark_encoder_only(model, input_tensor, num_runs=100):
    """Benchmark encoder-only forward pass (without classification head)"""
    times = []
    
    print(f"  Benchmarking encoder-only ({num_runs} runs)...")
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Run only the encoder part
            neighborhood, center = model.group_divider(input_tensor)
            group_input_tokens = model.encoder(neighborhood)
            
            cls_tokens = model.cls_token.expand(group_input_tokens.size(0), -1, -1)
            cls_pos = model.cls_pos.expand(group_input_tokens.size(0), -1, -1)
            
            pos = model.pos_embed(center)
            
            x = torch.cat((cls_tokens, group_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)
            
            x = model.blocks(x, pos)
            x = model.norm(x)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'throughput_per_sec': float(1000 / np.mean(times)),
        'all_times_ms': [float(t) for t in times]
    }


def benchmark_components(model, input_tensor, num_runs=100):
    """Benchmark individual components of the model"""
    results = {}
    
    print(f"  Benchmarking individual components ({num_runs} runs each)...")
    
    # 1. Benchmark FPS + KNN (Group Divider)
    print("    - FPS + KNN Grouping...")
    fps_knn_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            neighborhood, center = model.group_divider(input_tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            fps_knn_times.append((end - start) * 1000)
    
    results['fps_knn_grouping'] = {
        'mean_ms': float(np.mean(fps_knn_times)),
        'std_ms': float(np.std(fps_knn_times)),
        'min_ms': float(np.min(fps_knn_times)),
        'max_ms': float(np.max(fps_knn_times)),
        'description': 'Farthest Point Sampling + K-Nearest Neighbors (CUDA)'
    }
    
    # Get neighborhood and center for subsequent benchmarks
    neighborhood, center = model.group_divider(input_tensor)
    
    # 2. Benchmark Point Encoder (Mini-PointNet)
    print("    - Point Encoder (Mini-PointNet)...")
    encoder_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            group_input_tokens = model.encoder(neighborhood)
            torch.cuda.synchronize()
            end = time.perf_counter()
            encoder_times.append((end - start) * 1000)
    
    results['point_encoder'] = {
        'mean_ms': float(np.mean(encoder_times)),
        'std_ms': float(np.std(encoder_times)),
        'min_ms': float(np.min(encoder_times)),
        'max_ms': float(np.max(encoder_times)),
        'description': 'Mini-PointNet encoder for group features'
    }
    
    # Get tokens for subsequent benchmarks
    group_input_tokens = model.encoder(neighborhood)
    
    # 3. Benchmark Positional Embedding
    print("    - Positional Embedding...")
    pos_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            pos = model.pos_embed(center)
            torch.cuda.synchronize()
            end = time.perf_counter()
            pos_times.append((end - start) * 1000)
    
    results['positional_embedding'] = {
        'mean_ms': float(np.mean(pos_times)),
        'std_ms': float(np.std(pos_times)),
        'min_ms': float(np.min(pos_times)),
        'max_ms': float(np.max(pos_times)),
        'description': 'Positional embedding from center coordinates'
    }
    
    # Prepare for transformer benchmark
    cls_tokens = model.cls_token.expand(group_input_tokens.size(0), -1, -1)
    cls_pos = model.cls_pos.expand(group_input_tokens.size(0), -1, -1)
    pos = model.pos_embed(center)
    x = torch.cat((cls_tokens, group_input_tokens), dim=1)
    pos_full = torch.cat((cls_pos, pos), dim=1)
    
    # 4. Benchmark Transformer Blocks
    print("    - Transformer Blocks...")
    transformer_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            x_out = model.blocks(x, pos_full)
            x_out = model.norm(x_out)
            torch.cuda.synchronize()
            end = time.perf_counter()
            transformer_times.append((end - start) * 1000)
    
    results['transformer_blocks'] = {
        'mean_ms': float(np.mean(transformer_times)),
        'std_ms': float(np.std(transformer_times)),
        'min_ms': float(np.min(transformer_times)),
        'max_ms': float(np.max(transformer_times)),
        'description': f'Transformer encoder ({model.depth} layers, {model.num_heads} heads)'
    }
    
    # Get transformer output for classification head benchmark
    x_out = model.blocks(x, pos_full)
    x_out = model.norm(x_out)
    concat_f = torch.cat([x_out[:, 0], x_out[:, 1:].max(1)[0]], dim=-1)
    
    # 5. Benchmark Classification Head
    print("    - Classification Head...")
    cls_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model.cls_head_finetune(concat_f)
            torch.cuda.synchronize()
            end = time.perf_counter()
            cls_times.append((end - start) * 1000)
    
    results['classification_head'] = {
        'mean_ms': float(np.mean(cls_times)),
        'std_ms': float(np.std(cls_times)),
        'min_ms': float(np.min(cls_times)),
        'max_ms': float(np.max(cls_times)),
        'description': 'MLP classification head (768 → 256 → 256 → num_classes)'
    }
    
    return results


# ============================================================================
# Main Benchmark Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark REAL Point-MAE Model on GPU')
    parser.add_argument('--config', type=str, default='cfgs/finetune_modelnet.yaml', help='Config file path')
    parser.add_argument('--ckpts', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--n_points', type=int, default=8192, help='Number of input points')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--num_warmup', type=int, default=50, help='Number of warmup runs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("=" * 80)
    print("POINT-MAE REAL MODEL BENCHMARK (GPU)")
    print("=" * 80)
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(args.gpu)
    gpu_memory = torch.cuda.get_device_properties(args.gpu).total_memory / 1e9
    
    print(f"\nGPU Information:")
    print(f"  - Device: {gpu_name}")
    print(f"  - Memory: {gpu_memory:.1f} GB")
    print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"  - PyTorch Version: {torch.__version__}")
    
    print(f"\nBenchmark Configuration:")
    print(f"  - Input points: {args.n_points}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Benchmark runs: {args.num_runs}")
    print(f"  - Warmup runs: {args.num_warmup}")
    
    # Load config
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    config = cfg_from_yaml_file(args.config)
    
    # Create model config
    class ModelConfig:
        def __init__(self, cfg):
            self.trans_dim = cfg.model.trans_dim
            self.depth = cfg.model.depth
            self.drop_path_rate = cfg.model.drop_path_rate
            self.cls_dim = cfg.model.cls_dim
            self.num_heads = cfg.model.num_heads
            self.group_size = cfg.model.group_size
            self.num_group = cfg.model.num_group
            self.encoder_dims = cfg.model.encoder_dims
            self.input_channel = getattr(cfg.model, 'input_channel', 6)
    
    model_config = ModelConfig(config)
    
    print(f"\nModel Configuration:")
    print(f"  - Transformer dim: {model_config.trans_dim}")
    print(f"  - Depth: {model_config.depth}")
    print(f"  - Num heads: {model_config.num_heads}")
    print(f"  - Num groups: {model_config.num_group}")
    print(f"  - Group size: {model_config.group_size}")
    print(f"  - Encoder dims: {model_config.encoder_dims}")
    print(f"  - Input channels: {model_config.input_channel}")
    print(f"  - Num classes: {model_config.cls_dim}")
    
    # Build model
    model = PointTransformer(model_config)
    
    # Load checkpoint if provided
    if args.ckpts and os.path.exists(args.ckpts):
        print(f"\nLoading checkpoint: {args.ckpts}")
        model.load_model_from_ckpt(args.ckpts)
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  - Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  - Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Generate model architecture tree
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    
    model_tree = get_model_tree(model, max_depth=6)
    model_tree_str = "PointTransformer (REAL Architecture)\n" + "\n".join(model_tree[1:])
    print(model_tree_str[:3000] + "\n... (truncated for display)")
    
    # Save full architecture to file
    arch_file = os.path.join(args.output_dir, f'real_model_architecture_{timestamp}.txt')
    with open(arch_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POINT-MAE REAL MODEL ARCHITECTURE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"PyTorch: {torch.__version__}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Input points: {args.n_points}\n")
        f.write(f"  - Transformer dim: {model_config.trans_dim}\n")
        f.write(f"  - Depth: {model_config.depth}\n")
        f.write(f"  - Num heads: {model_config.num_heads}\n")
        f.write(f"  - Num groups: {model_config.num_group}\n")
        f.write(f"  - Group size: {model_config.group_size}\n")
        f.write(f"  - Encoder dims: {model_config.encoder_dims}\n")
        f.write(f"  - Input channels: {model_config.input_channel}\n")
        f.write(f"  - Num classes: {model_config.cls_dim}\n\n")
        f.write(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)\n\n")
        f.write("=" * 80 + "\n")
        f.write("LAYER TREE\n")
        f.write("=" * 80 + "\n\n")
        f.write(model_tree_str + "\n")
    
    print(f"\n✓ Architecture saved to: {arch_file}")
    
    # Create test input
    print("\n" + "=" * 80)
    print("BENCHMARKING")
    print("=" * 80)
    
    # Input: B x N x C (batch_size x n_points x input_channels)
    input_channels = model_config.input_channel
    test_input = torch.randn(args.batch_size, args.n_points, input_channels).to(device)
    print(f"\nInput shape: {test_input.shape} (B x N x C)")
    
    # Warmup
    warmup_gpu(model, test_input, args.num_warmup)
    
    # Benchmark full forward pass (classification)
    print("\n--- Full Classification Inference ---")
    full_results = benchmark_forward_pass(model, test_input, args.num_runs, "full classification")
    
    # Benchmark encoder only
    print("\n--- Encoder-Only Forward Pass ---")
    encoder_results = benchmark_encoder_only(model, test_input, args.num_runs)
    
    # Benchmark individual components
    print("\n--- Component-wise Breakdown ---")
    component_results = benchmark_components(model, test_input, args.num_runs)
    
    # Print results
    print("\n" + "=" * 80)
    print("TIMING RESULTS")
    print("=" * 80)
    
    print("\n--- Encoder Forward Pass (1 point cloud) ---")
    print(f"  Mean:       {encoder_results['mean_ms']:.3f} ms")
    print(f"  Std:        {encoder_results['std_ms']:.3f} ms")
    print(f"  Min:        {encoder_results['min_ms']:.3f} ms")
    print(f"  Max:        {encoder_results['max_ms']:.3f} ms")
    print(f"  Median:     {encoder_results['median_ms']:.3f} ms")
    print(f"  P95:        {encoder_results['p95_ms']:.3f} ms")
    print(f"  P99:        {encoder_results['p99_ms']:.3f} ms")
    print(f"  Throughput: {encoder_results['throughput_per_sec']:.1f} samples/sec")
    
    print("\n--- Full Classification Inference (1 point cloud) ---")
    print(f"  Mean:       {full_results['mean_ms']:.3f} ms")
    print(f"  Std:        {full_results['std_ms']:.3f} ms")
    print(f"  Min:        {full_results['min_ms']:.3f} ms")
    print(f"  Max:        {full_results['max_ms']:.3f} ms")
    print(f"  Median:     {full_results['median_ms']:.3f} ms")
    print(f"  P95:        {full_results['p95_ms']:.3f} ms")
    print(f"  P99:        {full_results['p99_ms']:.3f} ms")
    print(f"  Throughput: {full_results['throughput_per_sec']:.1f} samples/sec")
    
    print("\n--- Component Breakdown ---")
    total_component_time = 0
    for name, data in component_results.items():
        print(f"  {name}:")
        print(f"    Mean: {data['mean_ms']:.3f} ms ± {data['std_ms']:.3f} ms")
        print(f"    Description: {data['description']}")
        total_component_time += data['mean_ms']
    
    print(f"\n  Total (sum): {total_component_time:.3f} ms")
    
    # Calculate percentages
    print("\n--- Time Distribution (% of encoder time) ---")
    encoder_time = encoder_results['mean_ms']
    for name, data in component_results.items():
        if name != 'classification_head':
            pct = (data['mean_ms'] / encoder_time) * 100
            print(f"  {name}: {pct:.1f}%")
    
    # Save results to JSON
    results = {
        'timestamp': timestamp,
        'gpu_info': {
            'name': gpu_name,
            'memory_gb': gpu_memory,
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__
        },
        'configuration': {
            'n_points': args.n_points,
            'batch_size': args.batch_size,
            'trans_dim': model_config.trans_dim,
            'depth': model_config.depth,
            'num_heads': model_config.num_heads,
            'num_group': model_config.num_group,
            'group_size': model_config.group_size,
            'encoder_dims': model_config.encoder_dims,
            'input_channel': model_config.input_channel,
            'cls_dim': model_config.cls_dim,
            'num_runs': args.num_runs,
            'num_warmup': args.num_warmup
        },
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_millions': total_params / 1e6
        },
        'encoder_timing': {k: v for k, v in encoder_results.items() if k != 'all_times_ms'},
        'classification_timing': {k: v for k, v in full_results.items() if k != 'all_times_ms'},
        'component_timing': component_results,
        'raw_encoder_times_ms': encoder_results['all_times_ms'],
        'raw_classification_times_ms': full_results['all_times_ms']
    }
    
    results_file = os.path.join(args.output_dir, f'real_benchmark_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate paper-ready summary
    paper_summary = f"""# Point-MAE Inference Time Analysis (REAL Model)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Hardware Configuration
- **GPU**: {gpu_name}
- **GPU Memory**: {gpu_memory:.1f} GB
- **CUDA Version**: {torch.version.cuda}
- **PyTorch Version**: {torch.__version__}

## Model Configuration
- **Input**: {args.n_points} points × {model_config.input_channel} channels per point cloud
- **Architecture**: Point-MAE Transformer with {model_config.depth} layers
- **Embedding dimension**: {model_config.trans_dim}
- **Attention heads**: {model_config.num_heads}
- **Number of groups**: {model_config.num_group} (group size: {model_config.group_size})
- **Total parameters**: {total_params:,} ({total_params/1e6:.2f}M)

## Encoder Forward Pass (Single Point Cloud, Batch Size = {args.batch_size})

| Metric | Value |
|--------|-------|
| Mean latency | {encoder_results['mean_ms']:.2f} ± {encoder_results['std_ms']:.2f} ms |
| Median latency | {encoder_results['median_ms']:.2f} ms |
| Min latency | {encoder_results['min_ms']:.2f} ms |
| Max latency | {encoder_results['max_ms']:.2f} ms |
| 95th percentile | {encoder_results['p95_ms']:.2f} ms |
| 99th percentile | {encoder_results['p99_ms']:.2f} ms |
| Throughput | {encoder_results['throughput_per_sec']:.1f} samples/sec |

## Full Classification Inference (Single Point Cloud, Batch Size = {args.batch_size})

| Metric | Value |
|--------|-------|
| Mean latency | {full_results['mean_ms']:.2f} ± {full_results['std_ms']:.2f} ms |
| Median latency | {full_results['median_ms']:.2f} ms |
| Min latency | {full_results['min_ms']:.2f} ms |
| Max latency | {full_results['max_ms']:.2f} ms |
| 95th percentile | {full_results['p95_ms']:.2f} ms |
| 99th percentile | {full_results['p99_ms']:.2f} ms |
| Throughput | {full_results['throughput_per_sec']:.1f} samples/sec |

## Component-wise Latency Breakdown

| Component | Latency (ms) | % of Encoder |
|-----------|-------------|--------------|
| FPS + KNN Grouping | {component_results['fps_knn_grouping']['mean_ms']:.2f} ± {component_results['fps_knn_grouping']['std_ms']:.2f} | {(component_results['fps_knn_grouping']['mean_ms']/encoder_time)*100:.1f}% |
| Point Encoder | {component_results['point_encoder']['mean_ms']:.2f} ± {component_results['point_encoder']['std_ms']:.2f} | {(component_results['point_encoder']['mean_ms']/encoder_time)*100:.1f}% |
| Positional Embedding | {component_results['positional_embedding']['mean_ms']:.2f} ± {component_results['positional_embedding']['std_ms']:.2f} | {(component_results['positional_embedding']['mean_ms']/encoder_time)*100:.1f}% |
| Transformer Blocks | {component_results['transformer_blocks']['mean_ms']:.2f} ± {component_results['transformer_blocks']['std_ms']:.2f} | {(component_results['transformer_blocks']['mean_ms']/encoder_time)*100:.1f}% |
| Classification Head | {component_results['classification_head']['mean_ms']:.2f} ± {component_results['classification_head']['std_ms']:.2f} | N/A |

## Experimental Setup
- **Device**: {gpu_name}
- **Framework**: PyTorch {torch.__version__}
- **Batch size**: {args.batch_size}
- **Number of runs**: {args.num_runs}
- **Warmup runs**: {args.num_warmup}
- **Timing method**: CUDA synchronized (torch.cuda.synchronize())

## Methodology
- **Mean (±std)**: Average latency across {args.num_runs} runs with standard deviation
- **Throughput**: 1000 / mean_latency (samples per second)
- **P95/P99**: 95th and 99th percentile latencies
- **CUDA Sync**: All timings use torch.cuda.synchronize() for accurate GPU timing
"""
    
    summary_file = os.path.join(args.output_dir, f'real_paper_summary_{timestamp}.md')
    with open(summary_file, 'w') as f:
        f.write(paper_summary)
    
    print(f"✓ Paper summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {arch_file}")
    print(f"  2. {results_file}")
    print(f"  3. {summary_file}")


if __name__ == '__main__':
    main()
