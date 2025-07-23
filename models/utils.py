import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

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
        
        # 获取topk邻居
        dists, indices = torch.topk(pairwise_dist, k=k, dim=-1, largest=False, sorted=True)
        dists = torch.sqrt(dists.clamp(min=1e-8))  # 数值稳定性
    
    return dists, indices

def farthest_point_sample(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    优化的最远点采样算法 (支持FP16)
    输入:
        points: [B, N, 3] 点云坐标
        n_samples: 采样点数
    输出:
        indices: [B, n_samples] 采样点索引
    """
    B, N, _ = points.shape
    device = points.device
    
    # 初始化采样索引
    indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    
    # 随机选择起始点
    start_idx = torch.randint(0, N, (B,), device=device)
    indices[:, 0] = start_idx
    
    # 初始化距离矩阵
    dists = torch.ones(B, N, device=device, dtype=points.dtype) * 1e10
    
    with torch.no_grad():
        for i in range(1, n_samples):
            # 获取当前采样点
            cur_point = points[torch.arange(B), indices[:, i-1], :]
            
            # 计算到当前点的距离
            cur_dist = torch.norm(points - cur_point.unsqueeze(1), dim=-1)
            
            # 更新最小距离
            dists = torch.min(dists, cur_dist)
            
            # 选择距离最大的点
            indices[:, i] = torch.argmax(dists, dim=-1)
    
    return indices

def estimate_normals(points: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    使用PCA/SVD方法估计法向量 (支持FP16)
    输入:
        points: [B, N, 3] 点云坐标
        k: 邻居数量
    输出:
        normals: [B, N, 3] 法向量
    """
    B, N, _ = points.shape
    
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 获取KNN邻居
        _, knn_idx = knn_search(points, k=k)
        knn_points = points.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        
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
    """
    基于法向量变化计算曲率 (支持FP16)
    输入:
        points: [B, N, 3] 点云坐标
        normals: [B, N, 3] 法向量
        k: 邻居数量
    输出:
        curvature: [B, N, 1] 曲率值
    """
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        # 获取KNN邻居
        _, knn_idx = knn_search(points, k=k)
        knn_normals = normals.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        
        # 计算法向量差异
        normal_diff = 1 - (knn_normals * normals.unsqueeze(2)).sum(dim=-1)
        
        # 计算平均差异作为曲率估计
        curvature = normal_diff.mean(dim=-1, keepdim=True)
    
    return curvature

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
        dists, _ = knn_search(points, k=max_k, radius=radius)
        density = (dists < radius).sum(dim=-1, dtype=torch.float)  # [B, N]
        
        # 动态确定每个点的k值
        k_values = torch.clamp(
            density * (max_k - min_k) / (density.max(dim=1, keepdim=True)[0] + min_k),
            min_k, max_k
        ).long()  # [B, N]
        
        # 创建索引矩阵
        knn_idx = torch.zeros(B, N, max_k, dtype=torch.long, device=points.device)
        
        # 填充有效索引
        for b in range(B):
            for i in range(N):
                k = k_values[b, i].item()
                _, idx = torch.topk(dists[b, i], k=k, largest=False)
                knn_idx[b, i, :k] = idx
    
    return knn_idx, k_values

def compute_feature_gradient(features: torch.Tensor, points: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    计算特征在点云空间中的梯度 (支持FP16)
    输入:
        features: [B, N, D] 点特征
        points: [B, N, 3] 点坐标
        k: 邻居数量
    输出:
        gradients: [B, N, D] 特征梯度
    """
    B, N, D = features.shape
    
    with torch.cuda.amp.autocast(enabled=features.dtype==torch.float16):
        # 获取KNN邻居
        _, knn_idx = knn_search(points, k=k)
        knn_points = points.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        knn_features = features.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        
        # 计算相对位置
        rel_pos = knn_points - points.unsqueeze(2)  # [B, N, k, 3]
        
        # 计算特征差异
        feature_diff = knn_features - features.unsqueeze(2)  # [B, N, k, D]
        
        # 构建线性方程组 (最小二乘解)
        A = rel_pos  # [B, N, k, 3]
        b = feature_diff  # [B, N, k, D]
        
        # 求解梯度 (伪逆法)
        A_t = A.transpose(-1, -2)  # [B, N, 3, k]
        ATA = torch.matmul(A_t, A)  # [B, N, 3, 3]
        ATb = torch.matmul(A_t, b)  # [B, N, 3, D]
        
        # 添加正则项防止奇异矩阵
        I = torch.eye(3, device=points.device).view(1, 1, 3, 3)
        gradients = torch.matmul(torch.inverse(ATA + 1e-6 * I), ATb)  # [B, N, 3, D]
        
        # 取梯度范数
        gradient_norm = torch.norm(gradients, dim=2)  # [B, N, D]
    
    return gradient_norm

def batch_indexing(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    高效批索引操作 (支持任意维度)
    输入:
        data: [B, N, ...] 任意张量
        indices: [B, M] 索引矩阵
    输出:
        indexed_data: [B, M, ...] 索引后的数据
    """
    B, M = indices.shape
    batch_idx = torch.arange(B, device=data.device).view(B, 1).expand(-1, M)
    return data[batch_idx, indices]

def compute_density(points: torch.Tensor, radius: float = 0.1, k: int = 16) -> torch.Tensor:
    """
    计算点云局部密度 (支持FP16)
    输入:
        points: [B, N, 3] 点云坐标
        radius: 密度计算半径
        k: 最大邻居数
    输出:
        density: [B, N] 局部密度值
    """
    with torch.cuda.amp.autocast(enabled=points.dtype==torch.float16):
        dists, _ = knn_search(points, k=k, radius=radius)
        density = (dists < radius).sum(dim=-1, dtype=torch.float)
    return density

def pointnet2_sampler(points: torch.Tensor, features: torch.Tensor, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PointNet++风格的最远点采样和特征聚合
    输入:
        points: [B, N, 3] 点云坐标
        features: [B, N, D] 点特征
        n_samples: 采样点数
    输出:
        sampled_points: [B, n_samples, 3] 采样点
        sampled_features: [B, n_samples, D] 聚合特征
    """
    # 最远点采样
    fps_idx = farthest_point_sample(points, n_samples)
    sampled_points = batch_indexing(points, fps_idx)
    
    # 为每个采样点聚合邻居特征
    dists, knn_idx = knn_search(points, k=32)
    knn_features = batch_indexing(features, knn_idx)
    
    # 特征聚合 (最大池化)
    sampled_features = knn_features.max(dim=2)[0]
    
    return sampled_points, sampled_features

def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """
    点云归一化 (零中心化和单位缩放)
    输入:
        points: [B, N, 3] 点云坐标
    输出:
        normalized_points: [B, N, 3] 归一化后的点云
    """
    # 计算中心
    centroid = points.mean(dim=1, keepdim=True)
    
    # 中心化
    centered = points - centroid
    
    # 计算最大距离
    max_dist = centered.norm(dim=-1).max(dim=1, keepdim=True)[0]
    
    # 缩放
    normalized = centered / (max_dist.unsqueeze(-1) * 0.9)
    
    return normalized

def random_rotate_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """
    点云随机旋转增强 (支持FP16)
    输入:
        points: [B, N, 3] 点云坐标
    输出:
        rotated_points: [B, N, 3] 旋转后的点云
    """
    B = points.shape[0]
    device = points.device
    
    # 生成随机旋转矩阵
    angles = torch.rand(B, 3, device=device) * 2 * np.pi
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    
    # 构建旋转矩阵 (绕Z轴、Y轴、X轴)
    ones = torch.ones(B, device=device)
    zeros = torch.zeros(B, device=device)
    
    # Z轴旋转
    Rz = torch.stack([
        cos_a[:, 0], -sin_a[:, 0], zeros,
        sin_a[:, 0], cos_a[:, 0], zeros,
        zeros, zeros, ones
    ], dim=1).view(B, 3, 3)
    
    # Y轴旋转
    Ry = torch.stack([
        cos_a[:, 1], zeros, sin_a[:, 1],
        zeros, ones, zeros,
        -sin_a[:, 1], zeros, cos_a[:, 1]
    ], dim=1).view(B, 3, 3)
    
    # X轴旋转
    Rx = torch.stack([
        ones, zeros, zeros,
        zeros, cos_a[:, 2], -sin_a[:, 2],
        zeros, sin_a[:, 2], cos_a[:, 2]
    ], dim=1).view(B, 3, 3)
    
    # 组合旋转矩阵
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    
    # 应用旋转
    rotated = torch.bmm(points, R)
    
    return rotated

def point_dropout(points: torch.Tensor, features: torch.Tensor, dropout_rate: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    点云随机丢弃增强
    输入:
        points: [B, N, 3] 点云坐标
        features: [B, N, D] 点特征
        dropout_rate: 丢弃比例
    输出:
        new_points: [B, M, 3] 丢弃后的点云 (M ≈ N*(1-dropout_rate))
        new_features: [B, M, D] 丢弃后的特征
    """
    B, N, _ = points.shape
    device = points.device
    
    # 生成随机掩码
    mask = torch.rand(B, N, device=device) > dropout_rate
    
    # 应用掩码
    new_points = []
    new_features = []
    for i in range(B):
        new_points.append(points[i, mask[i]])
        new_features.append(features[i, mask[i]])
    
    # 填充到相同长度 (为了批处理)
    max_len = max([p.shape[0] for p in new_points])
    
    padded_points = torch.zeros(B, max_len, 3, device=device)
    padded_features = torch.zeros(B, max_len, features.shape[-1], device=device, dtype=features.dtype)
    
    for i in range(B):
        L = new_points[i].shape[0]
        padded_points[i, :L] = new_points[i]
        padded_features[i, :L] = new_features[i]
    
    return padded_points, padded_features