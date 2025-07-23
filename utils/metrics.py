import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff

def compute_overall_accuracy(pred, target):
    """计算总体准确率"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    correct = (pred == target).sum()
    total = target.size
    return correct / total

def compute_mean_class_accuracy(pred, target, num_classes):
    """计算每个类别的平均准确率"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        # 真正属于该类别的样本
        target_mask = (target == c)
        if np.sum(target_mask) == 0:
            continue
        
        # 预测正确的样本
        correct_mask = (pred == c) & target_mask
        
        # 计算该类别的准确率
        class_acc[c] = np.sum(correct_mask) / np.sum(target_mask)
    
    # 返回不包括空类的平均准确率
    valid_classes = np.sum(target == np.arange(num_classes)[:, None], axis=1) > 0
    return np.mean(class_acc[valid_classes])

def compute_iou(pred, target, num_classes):
    """计算每个类别的IoU和mIoU"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    iou = np.zeros(num_classes)
    for c in range(num_classes):
        # 真正属于该类别的样本
        target_mask = (target == c)
        
        # 预测为该类别的样本
        pred_mask = (pred == c)
        
        # 计算交集和并集
        intersection = np.sum(target_mask & pred_mask)
        union = np.sum(target_mask | pred_mask)
        
        # 计算IoU
        if union == 0:
            iou[c] = 1.0 if intersection == 0 else 0.0
        else:
            iou[c] = intersection / union
    
    # 返回每个类别的IoU和mIoU
    valid_classes = np.sum(target == np.arange(num_classes)[:, None], axis=1) > 0
    return iou, np.mean(iou[valid_classes])

def compute_boundary_iou(pred, target, boundary_mask, num_classes):
    """
    计算边界区域的IoU
    输入:
        pred: [N] 预测标签
        target: [N] 真实标签
        boundary_mask: [N] 边界点掩码
        num_classes: 类别数
    输出:
        boundary_iou: 边界区域的整体IoU
        class_boundary_iou: 每个类别的边界IoU
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(boundary_mask, torch.Tensor):
        boundary_mask = boundary_mask.cpu().numpy()
    
    # 只考虑边界点
    boundary_pred = pred[boundary_mask]
    boundary_target = target[boundary_mask]
    
    # 计算边界区域的整体IoU
    overall_iou = compute_iou(boundary_pred, boundary_target, num_classes)[1]
    
    # 计算每个类别的边界IoU
    class_iou = np.zeros(num_classes)
    for c in range(num_classes):
        # 真正属于该类别的边界点
        target_mask = (boundary_target == c)
        if np.sum(target_mask) == 0:
            continue
        
        # 预测为该类别的边界点
        pred_mask = (boundary_pred == c)
        
        # 计算交集和并集
        intersection = np.sum(target_mask & pred_mask)
        union = np.sum(target_mask | pred_mask)
        
        # 计算IoU
        if union == 0:
            class_iou[c] = 1.0 if intersection == 0 else 0.0
        else:
            class_iou[c] = intersection / union
    
    return overall_iou, class_iou

def compute_hausdorff_distance(pred, target, points, num_classes):
    """
    计算Hausdorff距离（形状匹配度量）
    输入:
        pred: [B, N] 预测标签
        target: [B, N] 真实标签
        points: [B, N, 3] 点云坐标
        num_classes: 类别数
    输出:
        hausdorff_dist: [num_classes] 每个类别的Hausdorff距离
    """
    hausdorff_dist = np.zeros(num_classes)
    
    for c in range(num_classes):
        dists = []
        for b in range(pred.shape[0]):
            # 获取预测和真实的类别c点集
            pred_points = points[b][pred[b] == c]
            target_points = points[b][target[b] == c]
            
            if len(pred_points) == 0 or len(target_points) == 0:
                continue
            
            # 计算双向Hausdorff距离
            dist1 = directed_hausdorff(pred_points, target_points)[0]
            dist2 = directed_hausdorff(target_points, pred_points)[0]
            dists.append(max(dist1, dist2))
        
        if len(dists) > 0:
            hausdorff_dist[c] = np.mean(dists)
    
    return hausdorff_dist

def compute_normal_consistency(pred_normals, gt_normals, mask=None):
    """
    计算法线一致性（角度差异）
    输入:
        pred_normals: [B, N, 3] 预测法线
        gt_normals: [B, N, 3] 真实法线
        mask: [B, N] 可选掩码
    输出:
        mean_angle_diff: 平均角度差异（度）
    """
    # 归一化法线
    pred_normals = pred_normals / torch.norm(pred_normals, dim=-1, keepdim=True)
    gt_normals = gt_normals / torch.norm(gt_normals, dim=-1, keepdim=True)
    
    # 计算点积（余弦值）
    cos_sim = torch.sum(pred_normals * gt_normals, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    # 计算角度差异（弧度）
    angle_diff = torch.acos(cos_sim)
    
    # 应用掩码
    if mask is not None:
        angle_diff = angle_diff[mask]
    
    # 转换为角度并返回平均值
    return torch.rad2deg(angle_diff).mean().item()

def evaluate_segmentation(pred, target, num_classes, points=None, 
                         pred_normals=None, gt_normals=None, boundary_mask=None):
    """增强的分割结果评估函数"""
    results = {}
    
    # 基础指标
    results['accuracy'] = compute_overall_accuracy(pred, target)
    results['mean_class_accuracy'] = compute_mean_class_accuracy(pred, target, num_classes)
    results['iou'], results['miou'] = compute_iou(pred, target, num_classes)
    
    # 边界指标
    if boundary_mask is not None:
        results['boundary_iou'], results['class_boundary_iou'] = compute_boundary_iou(
            pred, target, boundary_mask, num_classes
        )
    
    # 几何指标
    if points is not None:
        results['hausdorff_distance'] = compute_hausdorff_distance(
            pred, target, points, num_classes
        )
    
    if pred_normals is not None and gt_normals is not None:
        results['normal_consistency'] = compute_normal_consistency(
            pred_normals, gt_normals
        )
    
    return results