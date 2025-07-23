import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGuidedUpsampling(nn.Module):
    """语义引导的特征上采样"""
    def __init__(self, in_dim, out_dim, num_classes):
        super().__init__()
        self.semantic_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        self.feature_proj = nn.Linear(in_dim, out_dim)
        self.upsample_net = nn.Sequential(
            nn.Linear(in_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, src_points, tgt_points, src_features):
        """
        输入:
            src_points: [B, M, 3] 源点云坐标
            tgt_points: [B, N, 3] 目标点云坐标
            src_features: [B, M, C] 源特征
            
        输出:
            upsampled_features: [B, N, out_dim] 上采样后的特征
        """
        # 1. 预测语义logits
        semantic_logits = self.semantic_head(src_features)  # [B, M, num_classes]
        
        # 2. 计算点对点距离
        dists = torch.cdist(tgt_points, src_points)  # [B, N, M]
        knn_idx = dists.topk(k=3, largest=False).indices  # [B, N, 3]
        
        # 3. 收集源特征和语义
        knn_features = src_features.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, src_features.size(-1)))
        knn_semantic = semantic_logits.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, semantic_logits.size(-1)))
        
        # 4. 语义引导的特征融合
        weights = F.softmax(-dists.gather(2, knn_idx), dim=-1)  # [B, N, 3]
        
        # 语义加权
        semantic_weights = F.softmax(knn_semantic, dim=-1)  # [B, N, 3, num_classes]
        semantic_weights = semantic_weights.mean(dim=2)  # [B, N, num_classes]
        
        # 特征融合
        fused_features = torch.einsum('b n k d, b n k -> b n d', knn_features, weights)
        fused_semantic = torch.einsum('b n k c, b n k -> b n c', knn_semantic, weights)
        
        # 5. 语义引导的上采样
        combined = torch.cat([fused_features, semantic_weights], dim=-1)
        upsampled = self.upsample_net(combined)
        
        return upsampled, semantic_logits


class BoundaryAwareRefinement(nn.Module):
    """边界感知的特征优化"""
    def __init__(self, feature_dim):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim//2, 1),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Conv1d(feature_dim//2, 1, 1),
            nn.Sigmoid()
        )
        
        self.feature_refine = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim*2, 1),
            nn.BatchNorm1d(feature_dim*2),
            nn.ReLU(),
            nn.Conv1d(feature_dim*2, feature_dim, 1)
        )
        
    def forward(self, features, semantic_logits):
        """
        输入:
            features: [B, N, D] 特征
            semantic_logits: [B, N, C] 语义预测
            
        输出:
            refined_features: [B, N, D] 优化后的特征
        """
        # 计算边界概率
        entropy = -torch.sum(F.softmax(semantic_logits, dim=-1) * 
                         F.log_softmax(semantic_logits, dim=-1), dim=-1)
        boundary_prob = self.boundary_conv(features.transpose(1, 2)).squeeze(1)
        boundary_mask = (boundary_prob + 0.3 * entropy) > 0.5
        
        # 边界区域特征增强
        features_t = features.transpose(1, 2)
        refined = self.feature_refine(features_t).transpose(1, 2)
        
        # 仅在边界区域应用增强
        output = torch.where(boundary_mask.unsqueeze(-1), 
                           features + 0.5 * refined, 
                           features)
        return output