"""
06_losses.py - Loss Functions for Point-MAE

This module provides Chamfer Distance loss functions:
- ChamferDistanceL1: L1 Chamfer Distance (sqrt of squared distances)
- ChamferDistanceL2: L2 Chamfer Distance (squared distances)

Chamfer Distance measures the similarity between two point clouds:
CD(P, Q) = (1/|P|) * sum_{p in P} min_{q in Q} ||p - q||^2
         + (1/|Q|) * sum_{q in Q} min_{p in P} ||p - q||^2

This implementation is pure PyTorch (no CUDA extensions required).
For faster training, you can use the CUDA extension from the original repo.
"""

import torch
import torch.nn as nn


def chamfer_distance(xyz1, xyz2):
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        xyz1: (B, N, 3) first point cloud
        xyz2: (B, M, 3) second point cloud
        
    Returns:
        dist1: (B, N) distance from each point in xyz1 to nearest in xyz2
        dist2: (B, M) distance from each point in xyz2 to nearest in xyz1
    """
    # xyz1: B x N x 3
    # xyz2: B x M x 3
    
    # Compute pairwise squared distances
    # diff: B x N x M x 3
    diff = xyz1.unsqueeze(2) - xyz2.unsqueeze(1)
    dist_matrix = torch.sum(diff ** 2, dim=-1)  # B x N x M
    
    # For each point in xyz1, find nearest in xyz2
    dist1, _ = torch.min(dist_matrix, dim=2)  # B x N
    
    # For each point in xyz2, find nearest in xyz1
    dist2, _ = torch.min(dist_matrix, dim=1)  # B x M
    
    return dist1, dist2


class ChamferDistanceL2(nn.Module):
    """
    Chamfer Distance L2 (squared distances).
    
    CD_L2(P, Q) = mean(min_q ||p-q||^2) + mean(min_p ||p-q||^2)
    
    Args:
        ignore_zeros: If True and batch_size=1, ignore zero points
    """
    
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros
    
    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1: (B, N, 3) predicted point cloud
            xyz2: (B, M, 3) ground truth point cloud
            
        Returns:
            Scalar loss value
        """
        batch_size = xyz1.size(0)
        
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
        
        dist1, dist2 = chamfer_distance(xyz1, xyz2)
        
        return torch.mean(dist1) + torch.mean(dist2)


class ChamferDistanceL2_split(nn.Module):
    """
    Chamfer Distance L2 returning both directions separately.
    """
    
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros
    
    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
        
        dist1, dist2 = chamfer_distance(xyz1, xyz2)
        
        return torch.mean(dist1), torch.mean(dist2)


class ChamferDistanceL1(nn.Module):
    """
    Chamfer Distance L1 (sqrt of squared distances).
    
    CD_L1(P, Q) = (mean(sqrt(min_q ||p-q||^2)) + mean(sqrt(min_p ||p-q||^2))) / 2
    
    Args:
        ignore_zeros: If True and batch_size=1, ignore zero points
    """
    
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros
    
    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1: (B, N, 3) predicted point cloud
            xyz2: (B, M, 3) ground truth point cloud
            
        Returns:
            Scalar loss value
        """
        batch_size = xyz1.size(0)
        
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
        
        dist1, dist2 = chamfer_distance(xyz1, xyz2)
        
        # Take sqrt for L1
        dist1 = torch.sqrt(dist1 + 1e-8)  # Add epsilon for numerical stability
        dist2 = torch.sqrt(dist2 + 1e-8)
        
        return (torch.mean(dist1) + torch.mean(dist2)) / 2


class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (Wasserstein) Loss.
    
    Note: This is a simplified approximation using Hungarian matching.
    For exact EMD, use the CUDA extension from the original repo.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, xyz1, xyz2):
        """
        Approximate EMD using greedy matching.
        
        Args:
            xyz1: (B, N, 3) predicted point cloud
            xyz2: (B, N, 3) ground truth point cloud (must have same N)
            
        Returns:
            Scalar loss value
        """
        batch_size = xyz1.size(0)
        num_points = xyz1.size(1)
        
        assert xyz1.size(1) == xyz2.size(1), "EMD requires same number of points"
        
        total_loss = 0
        
        for b in range(batch_size):
            # Compute pairwise distances
            diff = xyz1[b].unsqueeze(1) - xyz2[b].unsqueeze(0)  # N x N x 3
            dist_matrix = torch.sum(diff ** 2, dim=-1)  # N x N
            
            # Greedy matching (approximation)
            matched_dist = 0
            used = torch.zeros(num_points, dtype=torch.bool, device=xyz1.device)
            
            for i in range(num_points):
                # Find nearest unmatched point
                dists = dist_matrix[i].clone()
                dists[used] = float('inf')
                j = torch.argmin(dists)
                matched_dist += dists[j]
                used[j] = True
            
            total_loss += matched_dist / num_points
        
        return total_loss / batch_size


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test point clouds
    xyz1 = torch.randn(4, 32, 3).to(device)
    xyz2 = torch.randn(4, 32, 3).to(device)
    
    # Test ChamferDistanceL2
    cd_l2 = ChamferDistanceL2().to(device)
    loss_l2 = cd_l2(xyz1, xyz2)
    print(f"ChamferDistanceL2: {loss_l2.item():.4f}")
    
    # Test ChamferDistanceL1
    cd_l1 = ChamferDistanceL1().to(device)
    loss_l1 = cd_l1(xyz1, xyz2)
    print(f"ChamferDistanceL1: {loss_l1.item():.4f}")
    
    # Test with identical point clouds (should be ~0)
    loss_same = cd_l2(xyz1, xyz1)
    print(f"ChamferDistanceL2 (same): {loss_same.item():.6f}")
    assert loss_same.item() < 1e-5, "Same point cloud should have ~0 loss"
    
    # Test gradient flow
    xyz1.requires_grad = True
    loss = cd_l2(xyz1, xyz2)
    loss.backward()
    assert xyz1.grad is not None, "Gradients should flow"
    print("Gradient flow: OK")
    
    # Test ChamferDistanceL2_split
    cd_l2_split = ChamferDistanceL2_split().to(device)
    d1, d2 = cd_l2_split(xyz1.detach(), xyz2)
    print(f"ChamferDistanceL2_split: d1={d1.item():.4f}, d2={d2.item():.4f}")
    
    print("\nAll loss function tests passed!")
