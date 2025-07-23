import torch
import torch.nn as nn
import numpy as np

class RobustNormalEstimator(nn.Module):
    """鲁棒的法线估计器，支持学习权重"""
    def __init__(self, k=16, num_iters=3, eps=1e-6):
        super().__init__()
        self.k = k
        self.num_iters = num_iters
        self.eps = eps
        self.weight_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, points):
        """
        输入:
            points: [B, N, 3] 点云坐标
        输出:
            normals: [B, N, 3] 估计的法线
        """
        B, N, _ = points.shape
        
        # 计算点对点距离
        dists = torch.cdist(points, points)
        
        # 获取K近邻
        _, knn_idx = torch.topk(dists, k=self.k+1, dim=-1, largest=False)
        knn_idx = knn_idx[:, :, 1:]  # 排除自身
        
        knn_points = torch.gather(
            points.unsqueeze(1).expand(-1, N, -1, -1), 
            dim=2, 
            index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        
        # 初始法线估计
        normals = self._weighted_pca(points, knn_points)
        
        # 迭代优化
        for _ in range(self.num_iters):
            # 计算权重
            diff = knn_points - points.unsqueeze(2)
            weights = self.weight_mlp(diff).squeeze(-1)  # [B, N, k]
            
            # 加权PCA
            normals = self._weighted_pca(points, knn_points, weights)
        
        return normals
    
    def _weighted_pca(self, points, knn_points, weights=None):
        """加权PCA法线估计"""
        B, N, K, _ = knn_points.shape
        
        # 计算中心点
        centroids = points.unsqueeze(2)  # [B, N, 1, 3]
        
        # 中心化点云
        centered = knn_points - centroids
        
        # 应用权重
        if weights is not None:
            centered = centered * weights.unsqueeze(-1)
        
        # 计算协方差矩阵
        cov_matrix = torch.einsum('bnik,bnil->bnkl', centered, centered) / (K - 1)
        
        # 使用SVD分解求最小特征值对应的特征向量
        _, _, V = torch.svd(cov_matrix)
        normals = V[:, :, :, 2]  # 最小特征值对应的特征向量
        
        # 法线方向一致性约束
        ref_normal = normals[:, :1].detach()  # 参考法线
        dot_product = (normals * ref_normal).sum(dim=-1, keepdim=True)
        sign = torch.sign(dot_product)
        normals = normals * sign
        
        return normals

def estimate_normals_jet(points, k=16):
    """使用Jet曲面拟合估计法线"""
    from sklearn.neighbors import NearestNeighbors
    from scipy.linalg import eigh
    
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
    _, indices = nbrs.kneighbors(points)
    indices = indices[:, 1:]  # 排除自身
    
    normals = np.zeros_like(points)
    for i in range(points.shape[0]):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov((neighbors - centroid).T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = eigh(cov_matrix)
        
        # 最小特征值对应的特征向量即为法线
        normals[i] = eigenvectors[:, 0]
    
    return normals