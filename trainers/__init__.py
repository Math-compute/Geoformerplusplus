# 基础训练器
from .base_trainer import BaseTrainer

# 监督学习和自监督学习训练器
from .supervised_trainer import SupervisedTrainer
from .ssl_trainer import SSLTrainer

# 训练工具函数
from .trainer_utils import (
    compute_class_iou,
    compute_boundary_iou,
    get_boundary_mask,
    geometric_augmentation,
    compute_geometric_consistency_loss
)

__all__ = [
    # 训练器类
    'BaseTrainer',
    'SupervisedTrainer',
    'SSLTrainer',
    
    # 评估和计算工具
    'compute_class_iou',
    'compute_boundary_iou',
    'get_boundary_mask',
    
    # 数据增强和损失计算
    'geometric_augmentation',
    'compute_geometric_consistency_loss'
]