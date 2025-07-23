import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

def normalize_point_cloud(pc, return_stats=False):
    """将点云归一化到单位球"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / (m + 1e-8)
    if return_stats:
        return pc, centroid, m
    return pc

def estimate_normals(points: torch.Tensor, k: int = 16) -> torch.Tensor:
    """使用PCA/SVD方法估计法向量 (支持FP16)"""
    B, N, _ = points.shape
    device = points.device
    
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 获取KNN邻居
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=k+1, largest=False).indices[:, :, 1:]  # 排除自身
        
        knn_points = torch.gather(
            points.unsqueeze(1).expand(-1, N, -1, -1), 
            dim=2, 
            index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        
        # 计算中心点
        centroids = knn_points.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
        
        # 中心化点云
        centered = knn_points - centroids
        
        # 计算协方差矩阵 (向量化)
        cov_matrix = torch.einsum('bnik,bnil->bnkl', centered, centered) / (k - 1)
        
        # 使用SVD分解求最小特征值对应的特征向量
        _, _, V = torch.svd(cov_matrix)
        normals = V[:, :, :, 2]  # 最小特征值对应的特征向量
        
        # 法线方向一致性约束
        ref_normal = normals[:, :1].detach()  # 参考法线
        dot_product = (normals * ref_normal).sum(dim=-1, keepdim=True)
        sign = torch.sign(dot_product)
        normals = normals * sign
    
    return normals

def compute_curvature(points: torch.Tensor, normals: torch.Tensor, k: int = 8) -> torch.Tensor:
    """基于法向量变化计算曲率 (支持FP16)"""
    B, N, _ = points.shape
    
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 获取KNN邻居
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=k+1, largest=False).indices[:, :, 1:]  # 排除自身
        
        knn_normals = torch.gather(
            normals.unsqueeze(1).expand(-1, N, -1, -1), 
            dim=2, 
            index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        
        # 计算法向量差异
        normal_diff = 1 - (knn_normals * normals.unsqueeze(2)).sum(dim=-1)
        
        # 计算平均差异作为曲率估计
        curvature = normal_diff.mean(dim=-1, keepdim=True)
    
    return curvature

def geometric_augmentation(points: torch.Tensor, labels=None, 
                          rotation_range=(-180, 180), 
                          jitter_std=0.01, 
                          scale_range=(0.8, 1.2),
                          shift_std=0.1):
    """
    点云几何增强 (支持FP16)
    输入:
        points: [B, N, 3] 点云坐标
        labels: [B, N] 可选标签
    输出:
        augmented_points: 增强后的点云
    """
    B, N, _ = points.shape
    device = points.device
    
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 随机旋转
        angles = torch.deg2rad(torch.FloatTensor(B, 3).uniform_(*rotation_range).to(device))
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        
        # 构建旋转矩阵
        Rx = torch.stack([
            torch.ones(B, device=device),
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            torch.zeros(B, device=device),
            cos_a[:, 0], -sin_a[:, 0],
            torch.zeros(B, device=device),
            sin_a[:, 0], cos_a[:, 0]
        ], dim=1).reshape(B, 3, 3)
        
        Ry = torch.stack([
            cos_a[:, 1], torch.zeros(B, device=device), sin_a[:, 1],
            torch.zeros(B, device=device), torch.ones(B, device=device), torch.zeros(B, device=device),
            -sin_a[:, 1], torch.zeros(B, device=device), cos_a[:, 1]
        ], dim=1).reshape(B, 3, 3)
        
        Rz = torch.stack([
            cos_a[:, 2], -sin_a[:, 2], torch.zeros(B, device=device),
            sin_a[:, 2], cos_a[:, 2], torch.zeros(B, device=device),
            torch.zeros(B, device=device), torch.zeros(B, device=device), torch.ones(B, device=device)
        ], dim=1).reshape(B, 3, 3)
        
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
        rotated = torch.bmm(points, R)
        
        # 随机缩放
        scales = torch.rand(B, 1, 3, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        scaled = rotated * scales
        
        # 随机平移
        shifts = torch.randn(B, 1, 3, device=device) * shift_std
        shifted = scaled + shifts
        
        # 随机抖动
        jitter = torch.randn_like(shifted) * jitter_std
        augmented_points = shifted + jitter
    
    if labels is not None:
        return augmented_points, labels
    return augmented_points

def farthest_point_sample(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """优化的最远点采样算法 (支持FP16)"""
    B, N, _ = points.shape
    device = points.device
    
    # 初始化采样索引
    indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    
    # 随机选择起始点
    start_idx = torch.randint(0, N, (B,), device=device)
    indices[:, 0] = start_idx
    
    # 初始化距离矩阵
    dists = torch.ones(B, N, device=device, dtype=points.dtype) * 1e10
    
    # 预计算点对点距离
    point_to_point = torch.cdist(points, points, p=2)  # [B, N, N]
    
    for i in range(1, n_samples):
        # 获取当前采样点
        current_points = points[torch.arange(B), indices[:, i-1], :]
        
        # 计算到当前点的距离
        cur_dists = point_to_point[torch.arange(B), indices[:, i-1]]  # [B, N]
        
        # 更新最小距离
        dists = torch.min(dists, cur_dists)
        
        # 选择距离最大的点
        indices[:, i] = torch.argmax(dists, dim=-1)
    
    return indices

def knn_search(points: torch.Tensor, k: int, radius: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    高效的K近邻搜索，支持FP16和半径限制
    输入:
        points: [B, N, 3] 点云坐标
        k: 邻居数量
        radius: 可选搜索半径
    输出:
        dists: [B, N, k] 邻居距离
        indices: [B, N, k] 邻居索引
    """
    B, N, _ = points.shape
    
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 计算距离矩阵 (使用展开优化大矩阵)
        points_norm = (points ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        pairwise_dist = points_norm + points_norm.transpose(1, 2) - 2 * torch.bmm(points, points.transpose(1, 2))
        pairwise_dist = pairwise_dist.clamp(min=0)  # 防止数值误差
        
        # 应用半径限制
        if radius is not None:
            mask = pairwise_dist > radius ** 2
            pairwise_dist[mask] = float('inf')
        
        # 获取topk邻居 (排除自身)
        dists, indices = torch.topk(pairwise_dist, k=k+1, dim=-1, largest=False, sorted=True)
        dists, indices = dists[:, :, 1:], indices[:, :, 1:]  # 排除第一个点(自身)
        dists = torch.sqrt(dists.clamp(min=1e-8))  # 数值稳定性
    
    return dists, indices

def compute_density(points: torch.Tensor, radius: float = 0.1, k: int = 16) -> torch.Tensor:
    """计算点云局部密度 (支持FP16)"""
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        dists, _ = knn_search(points, k=k, radius=radius)
        density = (dists < radius).sum(dim=-1, dtype=torch.float)
    return density

def adaptive_knn_search(points: torch.Tensor, min_k: int = 8, max_k: int = 32, radius: float = 0.1) -> torch.Tensor:
    """
    自适应KNN搜索 (基于局部密度)
    输入:
        points: [B, N, 3] 点云坐标
        min_k: 最小邻居数
        max_k: 最大邻居数
        radius: 密度计算半径
    输出:
        knn_idx: [B, N, k] 邻居索引 (k自适应变化)
    """
    B, N, _ = points.shape
    
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 计算点密度
        density = compute_density(points, radius, max_k)
        
        # 动态确定每个点的k值
        k_values = torch.clamp(
            density * (max_k - min_k) / (density.max(dim=1, keepdim=True)[0] + min_k),
            min_k, max_k
        ).long()  # [B, N]
        
        # 执行KNN搜索 (使用最大k值)
        _, knn_idx_full = knn_search(points, max_k)
        
        # 截断到实际k值
        knn_idx = torch.zeros(B, N, max_k, dtype=torch.long, device=points.device)
        for b in range(B):
            for i in range(N):
                k = k_values[b, i].item()
                knn_idx[b, i, :k] = knn_idx_full[b, i, :k]
    
    return knn_idx, k_values