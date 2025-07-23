import os
import torch
from torch.utils.data import Dataset
import numpy as np
from abc import ABC, abstractmethod

class BasePointCloudDataset(Dataset, ABC):
    """点云语义分割数据集基类"""
    
    def __init__(self, root_dir, split='train', transform=None, 
                 num_points=None, ignore_label=255, augment=False):
        """
        参数:
            root_dir: 数据集根目录
            split: 'train', 'val' 或 'test'
            transform: 自定义变换函数
            num_points: 每个样本的最大点数
            ignore_label: 忽略的标签值
            augment: 是否应用数据增强
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_points = num_points
        self.ignore_label = ignore_label
        self.augment = augment
        
        # 获取数据文件列表
        self.data_list = self._get_data_list()
        
        # 加载类信息
        self.class_names = self._get_class_names()
        self.class_weights = self._get_class_weights()
        self.class_colors = self._get_class_colors()
        
        if not self.data_list:
            raise RuntimeError(f"在 {root_dir} 中未找到 {split} 分割的数据")

    @abstractmethod
    def _get_data_list(self):
        """获取数据文件列表，由子类实现"""
        pass
    
    @abstractmethod
    def _load_data(self, data_path):
        """加载点云和标签数据，由子类实现"""
        pass
    
    def _get_class_names(self):
        """获取类别名称，可被子类覆盖"""
        return None
    
    def _get_class_weights(self):
        """获取类别权重，可被子类覆盖"""
        return None
    
    def _get_class_colors(self):
        """获取类别颜色映射，可被子类覆盖"""
        return None
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # 加载数据
        points, labels = self._load_data(self.data_list[idx])
        
        # 转换为numpy
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        # 采样到固定点数
        if self.num_points and len(points) != self.num_points:
            if len(points) >= self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
            else:
                # 不足则部分重复采样
                indices = np.random.choice(len(points), self.num_points - len(points), replace=True)
                indices = np.concatenate([np.arange(len(points)), indices])
            
            points = points[indices]
            labels = labels[indices] if labels is not None else None
        
        # 数据增强 (仅在训练时)
        if self.split == 'train' and self.augment:
            points, labels = self._apply_augmentation(points, labels)
        
        # 应用变换
        if self.transform:
            points, labels = self.transform(points, labels)
        
        # 转换为张量
        points = torch.from_numpy(points).float()
        if labels is not None:
            labels = torch.from_numpy(labels).long()
        
        return {
            'points': points,
            'labels': labels,
            'file_path': self.data_list[idx]
        }
    
    def _apply_augmentation(self, points, labels):
        """应用基本数据增强"""
        # 随机旋转
        angles = np.random.uniform(0, 2*np.pi, 3)
        rotation_matrix = self._create_rotation_matrix(angles)
        points[:, :3] = points[:, :3] @ rotation_matrix
        
        # 随机缩放 (0.8-1.2)
        scale = np.random.uniform(0.8, 1.2, 3)
        points[:, :3] *= scale
        
        # 随机平移 (-0.1m to 0.1m)
        translation = np.random.uniform(-0.1, 0.1, 3)
        points[:, :3] += translation
        
        # 颜色抖动 (RGB通道)
        if points.shape[1] > 3:
            color_jitter = np.random.uniform(0.9, 1.1, 3)
            points[:, 3:6] = np.clip(points[:, 3:6] * color_jitter, 0, 255)
        
        # 随机丢弃点 (0-5%)
        if np.random.rand() > 0.5:
            drop_ratio = np.random.uniform(0, 0.05)
            mask = np.random.rand(len(points)) > drop_ratio
            points = points[mask]
            labels = labels[mask] if labels is not None else None
        
        return points, labels
    
    def _create_rotation_matrix(self, angles):
        """创建旋转矩阵"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    def get_class_names(self):
        """获取类别名称"""
        return self.class_names
    
    def get_class_weights(self):
        """获取类别权重"""
        return self.class_weights
    
    def get_class_colors(self):
        """获取类别颜色映射"""
        return self.class_colors
    
    def get_num_classes(self):
        """获取类别数量"""
        return len(self.class_names) if self.class_names else 0