import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from .base_dataset import BasePointCloudDataset

class SemanticKITTIDataset(BasePointCloudDataset):
    """SemanticKITTI点云语义分割数据集"""
    
    CLASS_NAMES = [
        'unlabeled', 'outlier', 'car', 'bicycle', 'bus', 'motorcycle', 'on-rails',
        'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
        'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation',
        'trunk', 'terrain', 'pole', 'traffic-sign'
    ]
    
    LEARNING_MAP = {
        0: 255,  1: 255,  10: 0,   11: 1,   13: 4,   15: 2,   16: 4,   18: 3,
        20: 4,   30: 5,   31: 6,   32: 7,   40: 8,   44: 9,   48: 10,  49: 11,
        50: 12,  51: 13,  52: 14,  60: 15,  70: 16,  71: 17,  72: 18,  80: 19,
        81: 20,  99: 255, 252: 0,  256: 1,  253: 4,  254: 2,  255: 5,  257: 6,
        258: 7,  259: 8
    }
    
    CLASS_WEIGHTS = np.array([
        0.0000, 0.3337, 1.5066, 2.3590, 1.0899, 2.0573, 2.7727, 3.1348,
        3.3342, 0.2352, 1.1024, 0.6731, 1.0536, 0.4503, 1.0521, 0.5023,
        1.8305, 0.6724, 1.3117, 1.2025, 0.0000
    ])
    
    CLASS_COLORS = np.array([
        [0, 0, 0], [100, 150, 245], [100, 230, 245], [30, 60, 150], [80, 30, 180],
        [100, 80, 250], [255, 30, 30], [255, 40, 200], [150, 30, 90], [255, 0, 255],
        [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0], [255, 120, 50],
        [0, 175, 0], [135, 60, 0], [150, 240, 80], [255, 240, 150], [255, 0, 0],
        [0, 0, 255], [255, 255, 0], [0, 255, 255]
    ]) / 255.0

    def __init__(self, root_dir, split='train', transform=None,
                 num_points=65536, ignore_label=255, augment=False):
        super().__init__(root_dir, split, transform, num_points, ignore_label, augment)

    def _get_class_names(self):
        return self.CLASS_NAMES
    
    def _get_class_weights(self):
        return torch.from_numpy(self.CLASS_WEIGHTS).float()
    
    def _get_class_colors(self):
        return self.CLASS_COLORS

    def _get_data_list(self):
        """获取数据文件列表"""
        data_list = []
        
        # 序列划分
        if self.split == 'train':
            sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val':
            sequences = ['08']
        elif self.split == 'test':
            sequences = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise ValueError(f"无效分割: {self.split}")

        for seq in sequences:
            velodyne_dir = os.path.join(self.root_dir, 'sequences', seq, 'velodyne')
            labels_dir = os.path.join(self.root_dir, 'sequences', seq, 'labels')
            
            if not os.path.exists(velodyne_dir):
                continue
            
            # 获取所有bin文件
            velodyne_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
            
            for velodyne_file in velodyne_files:
                velodyne_path = os.path.join(velodyne_dir, velodyne_file)
                label_path = os.path.join(labels_dir, velodyne_file.replace('.bin', '.label'))
                
                # 测试集没有标签
                if self.split == 'test' or not os.path.exists(label_path):
                    data_list.append((velodyne_path, None))
                else:
                    data_list.append((velodyne_path, label_path))
        
        return data_list

    def _load_data(self, data_path):
        """加载点云和标签数据"""
        velodyne_path, label_path = data_path
        
        # 加载点云 (x,y,z,intensity)
        points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
        
        # 添加反射率作为额外特征
        intensity = points[:, 3].copy()
        points = np.hstack([points[:, :3], intensity[:, None]])
        
        # 加载标签（如果有）
        if label_path and os.path.exists(label_path):
            labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            # 提取语义标签（前16位）
            labels = labels & 0xFFFF
            # 应用学习映射
            labels = np.vectorize(self.LEARNING_MAP.get)(labels).astype(np.int64)
        else:
            labels = np.ones(points.shape[0], dtype=np.int64) * self.ignore_label
        
        return points, labels

    def _apply_augmentation(self, points, labels):
        """应用SemanticKITTI特定的数据增强"""
        points, labels = super()._apply_augmentation(points, labels)
        
        # SemanticKITTI特有的增强：随机地面扰动
        if np.random.rand() > 0.5:
            points = self._ground_perturbation(points)
        
        return points, labels
    
    def _ground_perturbation(self, points):
        """应用地面扰动增强"""
        # 识别地面点 (z < 0.2m)
        ground_mask = points[:, 2] < 0.2
        
        # 应用随机高度偏移
        height_shift = np.random.uniform(-0.05, 0.05)
        points[ground_mask, 2] += height_shift
        
        # 应用随机噪声
        noise = np.random.uniform(-0.02, 0.02, (np.sum(ground_mask), 3))
        points[ground_mask, :3] += noise
        
        return points