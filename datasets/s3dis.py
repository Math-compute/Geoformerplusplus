import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from .base_dataset import BasePointCloudDataset

class S3DISDataset(BasePointCloudDataset):
    """Stanford Large-Scale 3D Indoor Spaces Dataset"""
    
    CLASS_NAMES = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    
    CLASS_WEIGHTS = np.array([
        0.2248, 0.1505, 0.1496, 1.3416, 1.8112, 1.9687, 2.0175,
        0.6159, 0.5499, 2.1945, 1.2898, 1.9072, 0.5574
    ])
    
    CLASS_COLORS = np.array([
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
        [0, 128, 128], [220, 190, 255], [170, 110, 40]
    ]) / 255.0

    def __init__(self, root_dir, split='train', transform=None,
                 num_points=4096, block_size=1.0, stride=0.5,
                 ignore_label=255, test_area=5, augment=False):
        """
        参数:
            root_dir: 数据集根目录
            split: 'train' 或 'val'
            num_points: 每个样本的最大点数
            block_size: 分块大小(米)
            stride: 分块步长(米)
            ignore_label: 忽略的标签值
            test_area: 用作验证的区域编号
            augment: 是否应用数据增强
        """
        self.block_size = block_size
        self.stride = stride
        self.test_area = test_area
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
        
        # 区域划分
        areas = [f"Area_{i}" for i in range(1, 7)]
        if self.split == 'train':
            areas = [area for area in areas if area != f"Area_{self.test_area}"]
        elif self.split == 'val':
            areas = [f"Area_{self.test_area}"]
        else:
            raise ValueError(f"无效分割: {self.split}")

        # 扫描所有房间文件
        for area in areas:
            area_path = os.path.join(self.root_dir, area)
            if not os.path.exists(area_path):
                continue
                
            # 查找所有点云文件
            room_files = glob.glob(os.path.join(area_path, "*.npy"))
            for file_path in room_files:
                if "_label" in os.path.basename(file_path):
                    continue
                    
                # 对应的标签文件
                label_path = file_path.replace(".npy", "_label.npy")
                if not os.path.exists(label_path):
                    continue
                    
                data_list.append((file_path, label_path))
        
        return data_list

    def _load_data(self, data_path):
        """加载点云和标签数据"""
        point_file, label_file = data_path
        points = np.load(point_file)  # [N, 6] (x,y,z,r,g,b)
        labels = np.load(label_file)  # [N]
        
        # 确保标签在有效范围内
        labels = np.clip(labels, 0, len(self.CLASS_NAMES) - 1)
        
        return points, labels

    def _apply_augmentation(self, points, labels):
        """应用S3DIS特定的数据增强"""
        points, labels = super()._apply_augmentation(points, labels)
        
        # S3DIS特有的增强：弹性变形
        if np.random.rand() > 0.7:
            points = self._elastic_deformation(points)
        
        return points, labels
    
    def _elastic_deformation(self, points, granularity=0.2, magnitude=0.4):
        """应用弹性变形增强"""
        coords = points[:, :3].copy()
        
        # 创建位移场
        grid_size = int(np.ceil(np.ptp(coords, axis=0).max() / granularity))
        displacement = np.random.randn(grid_size + 3, grid_size + 3, grid_size + 3, 3) * magnitude
        
        # 应用平滑
        for i in range(3):
            displacement[..., i] = self._gaussian_filter(displacement[..., i], sigma=1.0)
        
        # 计算每个点的位移
        grid_coords = ((coords - coords.min(axis=0)) / granularity).astype(int)
        grid_coords = np.clip(grid_coords, 0, grid_size)
        
        # 三线性插值
        for i in range(len(coords)):
            x, y, z = grid_coords[i]
            coords[i] += displacement[x, y, z]
        
        points[:, :3] = coords
        return points
    
    def _gaussian_filter(self, grid, sigma=1.0):
        """应用高斯模糊"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(grid, sigma=sigma)