import numpy as np
import torch

class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, points, labels=None):
        for transform in self.transforms:
            points, labels = transform(points, labels)
        return points, labels


class RandomRotation:
    """随机旋转点云"""
    
    def __init__(self, max_angle=45, axis='z'):
        """
        参数:
            max_angle: 最大旋转角度(度)
            axis: 旋转轴 ('x', 'y', 'z' 或 'all')
        """
        self.max_angle = max_angle
        self.axis = axis
    
    def __call__(self, points, labels=None):
        if self.axis == 'all':
            # 随机选择旋转轴
            axis = np.random.choice(['x', 'y', 'z'])
        else:
            axis = self.axis
        
        # 转换为弧度
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        angle = np.radians(angle)
        
        # 创建旋转矩阵
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        else:  # 'z'
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        
        # 旋转坐标
        coords = points[:, :3]
        points[:, :3] = coords @ rotation_matrix
        
        return points, labels


class RandomScale:
    """随机缩放点云"""
    
    def __init__(self, scale_range=(0.95, 1.05), anisotropic=False):
        """
        参数:
            scale_range: 缩放范围 (min, max)
            anisotropic: 是否各向异性缩放
        """
        self.scale_range = scale_range
        self.anisotropic = anisotropic
    
    def __call__(self, points, labels=None):
        if self.anisotropic:
            # 每个轴独立缩放
            scales = np.random.uniform(self.scale_range[0], self.scale_range[1], 3)
        else:
            # 所有轴统一缩放
            scales = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        points[:, :3] *= scales
        return points, labels


class RandomShift:
    """随机平移点云"""
    
    def __init__(self, shift_range=0.1):
        """
        参数:
            shift_range: 平移范围 (米)
        """
        self.shift_range = shift_range
    
    def __call__(self, points, labels=None):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        points[:, :3] += shift
        return points, labels


class ColorJitter:
    """颜色抖动"""
    
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        参数:
            brightness: 亮度抖动范围
            contrast: 对比度抖动范围
            saturation: 饱和度抖动范围
            hue: 色调抖动范围
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, points, labels=None):
        if points.shape[1] < 6:  # 没有颜色信息
            return points, labels
        
        # 提取RGB
        rgb = points[:, 3:6].copy()
        
        # 转换为0-1范围
        rgb = rgb / 255.0
        
        # 应用颜色变换
        if self.brightness > 0:
            brightness = np.random.uniform(1 - self.brightness, 1 + self.brightness)
            rgb = rgb * brightness
        
        if self.contrast > 0:
            contrast = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            rgb = (rgb - 0.5) * contrast + 0.5
        
        if self.saturation > 0:
            saturation = np.random.uniform(1 - self.saturation, 1 + self.saturation)
            # 转换为HSV空间
            hsv = self._rgb_to_hsv(rgb)
            hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0, 1)
            rgb = self._hsv_to_rgb(hsv)
        
        if self.hue > 0:
            hue = np.random.uniform(-self.hue, self.hue)
            hsv = self._rgb_to_hsv(rgb)
            hsv[:, 0] = (hsv[:, 0] + hue) % 1.0
            rgb = self._hsv_to_rgb(hsv)
        
        # 限制在0-1范围
        rgb = np.clip(rgb, 0, 1)
        
        # 转换回0-255范围
        points[:, 3:6] = rgb * 255.0
        
        return points, labels
    
    def _rgb_to_hsv(self, rgb):
        """RGB转HSV"""
        hsv = np.zeros_like(rgb)
        max_val = rgb.max(axis=1)
        min_val = rgb.min(axis=1)
        diff = max_val - min_val
        
        # 计算色调
        with np.errstate(divide='ignore', invalid='ignore'):
            r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            
            # 红色通道最大
            mask = (max_val == r) & (diff != 0)
            hsv[mask, 0] = (g[mask] - b[mask]) / diff[mask] % 6.0
            
            # 绿色通道最大
            mask = (max_val == g) & (diff != 0)
            hsv[mask, 0] = (b[mask] - r[mask]) / diff[mask] + 2.0
            
            # 蓝色通道最大
            mask = (max_val == b) & (diff != 0)
            hsv[mask, 0] = (r[mask] - g[mask]) / diff[mask] + 4.0
            
            # 归一化色调
            hsv[:, 0] = (hsv[:, 0] / 6.0) % 1.0
            
            # 计算饱和度
            hsv[:, 1] = np.where(max_val > 0, diff / max_val, 0)
            
            # 计算明度
            hsv[:, 2] = max_val
        
        return hsv
    
    def _hsv_to_rgb(self, hsv):
        """HSV转RGB"""
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h = (h * 6.0) % 6.0
        i = np.floor(h).astype(int)
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        rgb = np.zeros_like(hsv)
        
        mask = (i == 0)
        rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
        
        mask = (i == 1)
        rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
        
        mask = (i == 2)
        rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
        
        mask = (i == 3)
        rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
        
        mask = (i == 4)
        rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
        
        mask = (i >= 5)
        rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)
        
        return rgb


class Normalize:
    """点云归一化"""
    
    def __init__(self, scale=1.0, center=True, color_scale=255.0):
        """
        参数:
            scale: 缩放因子
            center: 是否居中
            color_scale: 颜色缩放因子
        """
        self.scale = scale
        self.center = center
        self.color_scale = color_scale
    
    def __call__(self, points, labels=None):
        # 归一化坐标
        if self.center:
            centroid = np.mean(points[:, :3], axis=0)
            points[:, :3] -= centroid
        
        # 归一化尺度
        if self.scale != 1.0:
            max_dist = np.max(np.sqrt(np.sum(points[:, :3]**2, axis=1)))
            if max_dist > 0:
                points[:, :3] *= self.scale / max_dist
        
        # 归一化颜色
        if points.shape[1] > 3:
            points[:, 3:] /= self.color_scale
        
        return points, labels


class PointCloudToTensor:
    """将点云转换为张量"""
    
    def __call__(self, points, labels=None):
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).float()
        
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels).long()
        
        return points, labels