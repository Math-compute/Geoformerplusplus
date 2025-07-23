import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricConsistencyPretrain(nn.Module):
    """几何一致的自监督预训练模块"""
    def __init__(self, encoder, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # 解码器用于重建
        self.decoder = nn.Sequential(
            nn.Linear(encoder.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 重建法向量(3) + 曲率(1)
        )
        
    def forward(self, points):
        B, N, _ = points.shape
        
        # 1. 生成几何感知的掩码
        mask = self.geometric_mask(points)
        
        # 2. 编码可见点
        visible_points = points[~mask].view(B, -1, 3)
        features = self.encoder(visible_points)
        
        # 3. 重建被掩码的点
        reconstructed = self.decoder(features)
        
        # 4. 计算重建损失
        loss = self.compute_loss(points, mask, reconstructed)
        return loss
    
    def geometric_mask(self, points):
        """优先掩码高曲率区域"""
        # 简化曲率估计 (实际实现应使用更精确的方法)
        dists = torch.cdist(points, points)
        knn_dists = dists.topk(k=8, largest=False).values
        curvature = knn_dists.std(dim=-1)  # [B, N]
        
        # 高曲率区域有更高概率被掩码
        mask_prob = torch.sigmoid(5.0 * (curvature - 0.2)) * self.mask_ratio
        mask = torch.rand_like(curvature) < mask_prob
        return mask
    
    def compute_loss(self, points, mask, reconstructed):
        # 提取真实几何属性
        normals = self.estimate_normals(points)  # [B, N, 3]
        curvature = self.estimate_curvature(points, normals)  # [B, N, 1]
        gt_geo = torch.cat([normals, curvature], dim=-1)  # [B, N, 4]
        
        # 位置重建损失
        pos_loss = F.mse_loss(reconstructed[..., :3], points[mask].unsqueeze(0))
        
        # 法线一致性损失
        rec_normals = reconstructed[..., :3]
        gt_normals = normals[mask].unsqueeze(0)
        normal_loss = 1 - F.cosine_similarity(rec_normals, gt_normals).mean()
        
        # 曲率重建损失
        curvature_loss = F.mse_loss(reconstructed[..., 3:], curvature[mask].unsqueeze(0))
        
        return pos_loss + 0.5 * normal_loss + 0.2 * curvature_loss
    
    def estimate_normals(self, points):
        # 简化的法向量估计 (实际应使用PCA)
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=8, largest=False).indices
        knn_points = points.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        centroids = knn_points.mean(dim=2)
        return F.normalize(points - centroids, dim=-1)
    
    def estimate_curvature(self, points, normals):
        # 简化的曲率估计
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=8, largest=False).indices
        knn_normals = normals.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        normal_diff = 1 - (knn_normals * normals.unsqueeze(2)).sum(dim=-1)
        return normal_diff.mean(dim=-1, keepdim=True)