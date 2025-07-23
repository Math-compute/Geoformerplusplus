import torch
import torch.nn as nn
from . import geometric_modules, transformer_modules, decoder_modules

class GeoFormerPP(nn.Module):
    """几何感知的点云分割模型"""
    def __init__(self, in_channels=6, num_classes=20, embed_dim=256, depths=[2, 4, 2]):
        super().__init__()
        
        # 输入嵌入层
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim//2)
        )
        
        # 几何特征编码器
        self.geo_encoder = geometric_modules.GeometricFeatureEncoder(
            in_channels=embed_dim//2,
            embed_dim=embed_dim//2
        )
        
        # 多尺度Transformer编码器
        self.encoder_layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.ModuleList([
                transformer_modules.MultiScaleGeoFormer(
                    dim=embed_dim,
                    scales=[16, 32, 64] if i < 2 else [32, 64, 128],
                    heads=8
                ) for _ in range(depths[i])
            ])
            self.encoder_layers.append(layer)
            
            # 下采样
            if i < len(depths) - 1:
                self.encoder_layers.append(
                    nn.Sequential(
                        nn.Linear(embed_dim, embed_dim*2),
                        nn.ReLU()
                    )
                )
                embed_dim *= 2
        
        # 解码器层
        self.decoder_layers = nn.ModuleList()
        for i in range(len(depths)-2, -1, -1):
            # 上采样
            self.decoder_layers.append(
                decoder_modules.SemanticGuidedUpsampling(
                    in_dim=embed_dim,
                    out_dim=embed_dim//2,
                    num_classes=num_classes
                )
            )
            embed_dim //= 2
            
            # 特征精炼
            self.decoder_layers.append(
                decoder_modules.BoundaryAwareRefinement(embed_dim)
            )
        
        # 输出层
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        # 微分几何算子
        self.diff_geo = geometric_modules.DifferentialGeometryOperator(embed_dim)
    
    def forward(self, x, points):
        """
        输入:
            x: [B, N, C] 点特征 (坐标+颜色等)
            points: [B, N, 3] 点坐标
        """
        # 初始特征提取
        features = self.input_proj(x)  # [B, N, embed_dim//2]
        
        # 几何特征融合
        geo_fused = self.geo_encoder(points, features)  # [B, N, embed_dim//2]
        features = torch.cat([features, geo_fused], dim=-1)  # [B, N, embed_dim]
        
        # 几何属性估计
        normals = self.geo_encoder.estimate_normals(points)
        curvature = self.geo_encoder.estimate_curvature(points, normals)
        geo_features = torch.cat([normals, curvature], dim=-1)  # [B, N, 4]
        
        # 编码器路径
        encoder_features = []
        current_points = points
        for layer_group in self.encoder_layers:
            if isinstance(layer_group, nn.ModuleList):
                for block in layer_group:
                    features = block(features, current_points, geo_features)
                encoder_features.append(features)
            else:
                features = layer_group(features)
                # 下采样点云
                current_points = self.fps(current_points, features.size(1))
                geo_features = geo_features.gather(1, self.fps_indices)
        
        # 解码器路径
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, decoder_modules.SemanticGuidedUpsampling):
                # 上采样
                features, sem_logits = layer(
                    src_points=current_points,
                    tgt_points=encoder_features.pop()[0],  # 上一级点
                    src_features=features
                )
                current_points = encoder_features.pop()[1]  # 恢复点
            else:
                # 边界增强
                features = layer(features, sem_logits)
                
                # 应用微分几何算子
                _, features = self.diff_geo(features, current_points)
        
        # 最终预测
        seg_logits = self.seg_head(features)
        return seg_logits
    
    def fps(self, points, n_samples):
        """最远点采样"""
        # 简化实现 (实际应使用CUDA优化版本)
        B, N, _ = points.shape
        device = points.device
        
        indices = torch.zeros(B, n_samples, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        
        # 随机选择起始点
        start_idx = torch.randint(0, N, (B,), device=device)
        batch_indices = torch.arange(B, device=device)
        
        for i in range(n_samples):
            indices[:, i] = start_idx
            cur_point = points[batch_indices, start_idx, :]
            
            # 更新距离
            dist_to_center = torch.norm(points - cur_point.unsqueeze(1), dim=-1)
            distance = torch.min(distance, dist_to_center)
            
            # 选择下一个点
            start_idx = torch.argmax(distance, dim=-1)
        
        self.fps_indices = indices
        return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))