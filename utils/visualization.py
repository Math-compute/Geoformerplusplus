import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import torch

def visualize_point_cloud(points, colors=None, title="Point Cloud", figsize=(10, 10)):
    """使用matplotlib可视化点云"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=10, alpha=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=10, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置坐标轴比例相等
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

def visualize_point_cloud_with_labels(points, labels, num_classes=None, title="Point Cloud with Labels"):
    """可视化带标签的点云"""
    if num_classes is None:
        num_classes = np.max(labels) + 1
    
    # 为每个类别生成不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    # 为每个点分配对应的颜色
    point_colors = colors[labels]
    
    # 可视化点云
    visualize_point_cloud(points, point_colors, title)

def visualize_point_cloud_open3d(points, colors=None, normals=None, window_name="Point Cloud"):
    """使用Open3D可视化点云"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def visualize_segmentation_result(points, pred_labels, gt_labels=None, num_classes=None):
    """可视化分割结果"""
    if num_classes is None:
        num_classes = max(np.max(pred_labels), np.max(gt_labels)) + 1
    
    # 为每个类别生成不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    # 可视化预测结果
    pred_colors = colors[pred_labels]
    visualize_point_cloud_open3d(points, pred_colors, "Prediction")
    
    # 如果有真实标签，也可视化
    if gt_labels is not None:
        gt_colors = colors[gt_labels]
        visualize_point_cloud_open3d(points, gt_colors, "Ground Truth")

def visualize_geometry_features(points, features, cmap='viridis', title="Geometry Features"):
    """可视化几何特征（如曲率）"""
    # 归一化特征
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    # 处理多通道特征
    if features.ndim > 1 and features.shape[1] > 1:
        # 使用主成分分析 (PCA) 降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        features = pca.fit_transform(features)
    
    # 归一化到0-1范围
    features = (features - features.min()) / (features.max() - features.min() + 1e-8)
    
    # 应用颜色映射
    cmap = plt.get_cmap(cmap)
    colors = cmap(features)[:, 0]
    
    visualize_point_cloud(points, colors, title)

def visualize_attention_map(points, attention_weights, title="Attention Map"):
    """可视化注意力权重图"""
    # 归一化注意力权重
    attention_weights = (attention_weights - attention_weights.min()) / \
                       (attention_weights.max() - attention_weights.min() + 1e-8)
    
    # 应用热力图颜色映射
    cmap = plt.get_cmap('hot')
    colors = cmap(attention_weights)[:, 0]
    
    visualize_point_cloud(points, colors, title)

def visualize_point_cloud_sequence(points_sequence, labels_sequence=None, interval=0.5):
    """可视化点云序列（如动态点云）"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, points in enumerate(points_sequence):
        ax.clear()
        
        if labels_sequence is not None:
            labels = labels_sequence[i]
            num_classes = np.max(labels) + 1
            colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
            point_colors = colors[labels]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=10)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)
        
        ax.set_title(f"Frame {i+1}/{len(points_sequence)}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.pause(interval)
    
    plt.show()