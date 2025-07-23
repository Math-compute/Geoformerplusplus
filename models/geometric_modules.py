import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

class GeometricFeatureEncoder(nn.Module):
    """几何特征提取与融合模块"""
    def __init__(self, in_channels, embed_dim=64):
        super().__init__()
        # 几何特征处理流
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
        # 特征融合门控机制
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # 语义特征投影
        self.sem_mlp = nn.Linear(in_channels, embed_dim)
        
    def forward(self, points, semantic_features):
        """
        输入:
            points: [B, N, 3] 点云坐标
            semantic_features: [B, N, C] 语义特征
            
        输出:
            fused_features: [B, N, embed_dim] 融合后的特征
        """
        B, N, _ = points.shape
        
        # 1. 计算法向量 (使用PCA方法)
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=16, largest=False).indices
        knn_points = points.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        
        centroids = knn_points.mean(dim=2)
        cov_matrix = torch.einsum('bnik,bnil->bnkl', knn_points - centroids.unsqueeze(2), 
                                 knn_points - centroids.unsqueeze(2))
        _, _, V = torch.svd(cov_matrix)
        normals = V[:, :, :, 2]  # 最小特征值对应的特征向量
        
        # 2. 计算曲率 (基于法向量变化)
        knn_normals = normals.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        curvature = 1 - (knn_normals * normals.unsqueeze(2)).sum(dim=-1).mean(dim=-1)
        curvature = curvature.unsqueeze(-1)  # [B, N, 1]
        
        # 3. 几何特征组合
        geo_features = torch.cat([normals, curvature], dim=-1)  # [B, N, 4]
        
        # 4. 特征融合
        geo_embed = self.geo_mlp(geo_features)  # [B, N, embed_dim]
        sem_embed = self.sem_mlp(semantic_features)  # [B, N, embed_dim]
        
        # 门控融合: G = σ(W·[G; S]), F_fused = G ⊙ S + (1-G) ⊙ G
        gate = self.gate_mlp(torch.cat([geo_embed, sem_embed], dim=-1))
        fused_features = gate * sem_embed + (1 - gate) * geo_embed
        
        return fused_features


class DifferentialGeometryOperator(nn.Module):
    """微分几何算子，用于边界增强"""
    def __init__(self, feature_dim):
        super().__init__()
        self.boundary_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, points):
        """
        输入:
            features: [B, N, D] 特征
            points: [B, N, 3] 点坐标
            
        输出:
            boundary_weights: [B, N, 1] 边界权重
            enhanced_features: [B, N, D] 增强后的特征
        """
        # 计算特征梯度
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=8, largest=False).indices
        knn_feats = features.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, features.size(-1)))
        
        # 特征差异 (梯度近似)
        feat_diff = knn_feats - features.unsqueeze(2)
        feat_grad = torch.norm(feat_diff, dim=-1).mean(dim=2)  # [B, N]
        
        # 边界概率
        boundary_prob = self.boundary_net(features)  # [B, N, 1]
        
        # 特征增强: F' = F + α·tanh(β·‖∇F‖)·B
        enhancement = torch.tanh(5.0 * feat_grad.unsqueeze(-1)) * boundary_prob
        enhanced_features = features + 0.3 * enhancement
        
        return boundary_prob, enhanced_features