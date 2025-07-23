import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_class_iou(preds, labels, num_classes):
    """计算每个类别的IoU"""
    iou_list = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            iou_list.append(float('nan'))  # 避免除以0
        else:
            iou_list.append(intersection / union)
    return np.array(iou_list)

def compute_boundary_iou(preds, labels, boundary_mask, num_classes):
    """计算边界区域的IoU"""
    boundary_preds = preds[boundary_mask]
    boundary_labels = labels[boundary_mask]
    return compute_class_iou(boundary_preds, boundary_labels, num_classes)

def get_boundary_mask(labels, kernel_size=3, ignore_index=-100):
    """
    获取边界区域掩码
    输入:
        labels: [B, N] 标签
        kernel_size: 边界检测核大小
    """
    boundary_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    for b in range(labels.size(0)):
        # 获取当前批次的标签
        current_labels = labels[b]
        
        # 创建扩展的标签矩阵用于边界检测
        padded_labels = torch.nn.functional.pad(
            current_labels.unsqueeze(0).unsqueeze(0), 
            (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2),
            mode='replicate'
        ).squeeze()
        
        # 检测边界点
        for i in range(current_labels.size(0)):
            center_label = current_labels[i]
            if center_label == ignore_index:
                continue
                
            # 检查邻域
            neighborhood = padded_labels[i:i+kernel_size]
            different_labels = (neighborhood != center_label).any()
            
            if different_labels:
                boundary_mask[b, i] = True
    
    return boundary_mask

def geometric_augmentation(points, labels=None, rotation_std=0.1, jitter_std=0.01, scale_range=(0.9, 1.1)):
    """点云几何增强"""
    B, N, C = points.shape
    
    # 随机旋转
    angles = torch.randn(B, 3, device=points.device) * rotation_std
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    
    # 构建旋转矩阵
    Rx = torch.zeros(B, 3, 3, device=points.device)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_a[:, 0]
    Rx[:, 1, 2] = -sin_a[:, 0]
    Rx[:, 2, 1] = sin_a[:, 0]
    Rx[:, 2, 2] = cos_a[:, 0]
    
    Ry = torch.zeros(B, 3, 3, device=points.device)
    Ry[:, 0, 0] = cos_a[:, 1]
    Ry[:, 0, 2] = sin_a[:, 1]
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_a[:, 1]
    Ry[:, 2, 2] = cos_a[:, 1]
    
    Rz = torch.zeros(B, 3, 3, device=points.device)
    Rz[:, 0, 0] = cos_a[:, 2]
    Rz[:, 0, 1] = -sin_a[:, 2]
    Rz[:, 1, 0] = sin_a[:, 2]
    Rz[:, 1, 1] = cos_a[:, 2]
    Rz[:, 2, 2] = 1
    
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    augmented_points = torch.bmm(points, R)
    
    # 随机缩放
    scales = torch.rand(B, 1, 3, device=points.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    augmented_points = augmented_points * scales
    
    # 随机抖动
    jitter = torch.randn_like(augmented_points) * jitter_std
    augmented_points += jitter
    
    if labels is not None:
        return augmented_points, labels
    return augmented_points

def compute_geometric_consistency_loss(original_points, reconstructed_points, 
                                      original_normals, reconstructed_normals, 
                                      mask, lambda_normal=0.5):
    """
    计算几何一致性损失
    """
    # 位置重建损失
    pos_loss = F.mse_loss(reconstructed_points[mask], original_points[mask])
    
    # 法线一致性损失
    normal_sim = 1 - F.cosine_similarity(reconstructed_normals[mask], original_normals[mask], dim=-1)
    normal_loss = normal_sim.mean()
    
    return pos_loss + lambda_normal * normal_loss