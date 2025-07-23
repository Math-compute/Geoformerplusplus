import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from .base_dataset import BasePointCloudDataset

class ScanNetDataset(BasePointCloudDataset):
    """ScanNet点云语义分割数据集"""
    
    CLASS_NAMES = [
        'unlabeled', 'wall', 'floor', 'cabinet', 'bed', 'chair', 
        'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 
        'counter', 'blinds', 'desk', 'shelves', 'curtain', 
        'dresser', 'pillow', 'mirror', 'floor-mat'
    ]
    
    CLASS_WEIGHTS = np.array([
        0.0000, 0.3173, 0.2030, 2.2151, 2.6558, 0.7543, 2.9400, 1.1476, 
        1.8331, 1.9613, 2.1857, 2.3517, 1.5456, 3.2015, 2.0224, 2.3440, 
        1.9870, 2.6152, 3.1072, 2.5010, 3.0000
    ])
    
    # 忽略unlabeled类
    CLASS_WEIGHTS[0] = 0.0
    
    CLASS_COLORS = np.array([
        [0, 0, 0], [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
        [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213],
        [148, 103, 189], [196, 156, 148], [23, 190, 207], [247, 182, 210], [219, 219, 141],
        [255, 127, 14], [158, 218, 229], [44, 160, 44], [112, 128, 144], [227, 119, 194],
        [82, 84, 163]
    ]) / 255.0

    def __init__(self, root_dir, split='train', transform=None,
                 num_points=40960, ignore_label=255, augment=False):
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
        
        # 根据分割类型选择场景列表文件
        if self.split == 'train':
            scene_list_file = os.path.join(self.root_dir, 'scannetv2_train.txt')
        elif self.split == 'val':
            scene_list_file = os.path.join(self.root_dir, 'scannetv2_val.txt')
        elif self.split == 'test':
            scene_list_file = os.path.join(self.root_dir, 'scannetv2_test.txt')
        else:
            raise ValueError(f"无效分割: {self.split}")
        
        if not os.path.exists(scene_list_file):
            return []
        
        # 读取场景列表
        with open(scene_list_file, 'r') as f:
            scenes = [line.strip() for line in f.readlines()]
        
        # 构建数据路径
        for scene in scenes:
            point_file = os.path.join(self.root_dir, 'points', f"{scene}.npy")
            
            if self.split == 'test':
                # 测试集没有标签
                data_list.append((point_file, None))
            else:
                label_file = os.path.join(self.root_dir, 'labels', f"{scene}.npy")
                if os.path.exists(point_file) and os.path.exists(label_file):
                    data_list.append((point_file, label_file))
        
        return data_list

    def _load_data(self, data_path):
        """加载点云和标签数据"""
        point_file, label_file = data_path
        
        # 加载点云
        points = np.load(point_file)  # [N, 6] (x,y,z,r,g,b)
        
        # 归一化颜色
        if points.shape[1] > 3:
            points[:, 3:6] = points[:, 3:6] / 255.0 - 0.5
        
        # 加载标签（如果有）
        if label_file and os.path.exists(label_file):
            labels = np.load(label_file).astype(np.int64)
            # 应用学习映射（如果有）
            if hasattr(self, 'learning_map'):
                labels = np.vectorize(self.learning_map.get)(labels).astype(np.int64)
        else:
            labels = np.ones(points.shape[0], dtype=np.int64) * self.ignore_label
        
        return points, labels

    def _apply_augmentation(self, points, labels):
        """应用ScanNet特定的数据增强"""
        points, labels = super()._apply_augmentation(points, labels)
        
        # ScanNet特有的增强：随机平面切割
        if np.random.rand() > 0.5:
            points, labels = self._random_plane_cut(points, labels)
        
        return points, labels
    
    def _random_plane_cut(self, points, labels):
        """随机平面切割增强"""
        # 随机选择切割轴
        axis = np.random.choice(3)
        min_val = np.min(points[:, axis])
        max_val = np.max(points[:, axis])
        
        # 随机选择切割位置
        cut_pos = np.random.uniform(min_val + 0.1, max_val - 0.1)
        
        # 随机选择保留哪一侧
        if np.random.rand() > 0.5:
            mask = points[:, axis] > cut_pos
        else:
            mask = points[:, axis] < cut_pos
        
        return points[mask], labels[mask] if labels is not None else None