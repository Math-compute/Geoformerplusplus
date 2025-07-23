import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class GeometricAttention(nn.Module):
    """几何感知的多头注意力机制"""
    def __init__(self, dim, heads=8, k=16):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.k = k
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.geo_proj = nn.Linear(4, heads)  # 几何特征到注意力头
        
        # 位置编码
        self.pos_enc = nn.Sequential(
            nn.Linear(3, dim // heads),
            nn.ReLU()
        )
        
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, points, geo_features):
        """
        输入:
            x: [B, N, D] 特征
            points: [B, N, 3] 点坐标
            geo_features: [B, N, 4] 几何特征 (法向量+曲率)
        """
        B, N, D, H = *x.shape, self.heads
        
        # 1. 获取查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=H), qkv)
        
        # 2. 寻找k近邻
        dists = torch.cdist(points, points)
        knn_idx = dists.topk(k=self.k, largest=False).indices  # [B, N, k]
        
        # 3. 几何引导的注意力
        geo_weights = self.geo_proj(geo_features)  # [B, N, H]
        geo_weights = rearrange(geo_weights, 'b n h -> b h n 1')
        
        # 4. 位置编码
        rel_pos = points.unsqueeze(2) - points.unsqueeze(1)  # [B, N, N, 3]
        rel_pos = rel_pos.gather(2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, -1, 3))  # [B, N, k, 3]
        pos_enc = self.pos_enc(rel_pos)  # [B, N, k, d]
        pos_enc = rearrange(pos_enc, 'b n k (h d) -> b h n k d', h=H)
        
        # 5. 注意力计算 (加入几何相似性约束)
        k_gather = k.gather(2, knn_idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, D//H))
        v_gather = v.gather(2, knn_idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, -1, D//H))
        
        dots = torch.einsum('b h i d, b h i j d -> b h i j', q, k_gather) * self.scale
        dots += torch.einsum('b h i d, b h i j d -> b h i j', q, pos_enc) * 0.5
        
        # 几何约束: A_ij = softmax(Q·K/√d + λ·cos∠(n_i, n_j))
        knn_geo = geo_features.gather(1, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 4))
        normal_sim = F.cosine_similarity(
            geo_features[:, :, :3].unsqueeze(2),
            knn_geo[:, :, :, :3],
            dim=-1
        )  # [B, N, k]
        normal_sim = rearrange(normal_sim, 'b i j -> b () i j')
        dots += 0.3 * normal_sim * geo_weights
        
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 6. 聚合值
        out = torch.einsum('b h i j, b h i j d -> b h i d', attn, v_gather + pos_enc)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class MultiScaleGeoFormer(nn.Module):
    """多尺度几何Transformer模块"""
    def __init__(self, dim, scales=[16, 32, 64], heads=8):
        super().__init__()
        self.scales = scales
        self.attention_heads = nn.ModuleList([
            GeometricAttention(dim, heads=heads, k=k) for k in scales
        ])
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(dim * len(scales), dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x, points, geo_features):
        attended = []
        for attn_head in self.attention_heads:
            attended.append(attn_head(x, points, geo_features))
        
        fused = torch.cat(attended, dim=-1)
        return self.feature_fusion(fused)