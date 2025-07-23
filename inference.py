import argparse
import os
import torch
import numpy as np
import open3d as o3d
from models import GeoFormerSeg
from utils import setup_logger, set_seed, normalize_point_cloud
from utils.visualization import visualize_segmentation_result, visualize_geometry_features
import yaml

def main():
    parser = argparse.ArgumentParser(description='GeoFormer点云分割推理')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input_path', type=str, required=True, help='输入点云路径')
    parser.add_argument('--output_dir', type=str, default='./output/inference', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--num_points', type=int, default=4096, help='输入点云点数')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--save_features', action='store_true', help='保存几何特征')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    logger = setup_logger('GeoFormer_inference', args.output_dir, 0)
    logger.info(f'使用检查点: {args.checkpoint}')
    logger.info(f'输入点云: {args.input_path}')

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']

    # 加载点云
    if args.input_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(args.input_path)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = None
    elif args.input_path.endswith('.xyz') or args.input_path.endswith('.txt'):
        data = np.loadtxt(args.input_path)
        points = data[:, :3]
        if data.shape[1] >= 6:  # 包含颜色信息
            colors = data[:, 3:6]
        else:
            colors = None
    else:
        raise ValueError(f"不支持的文件格式: {args.input_path}")
    
    logger.info(f"加载点云，点数: {len(points)}")

    # 保存原始点云
    input_file_name = os.path.splitext(os.path.basename(args.input_path))[0]
    np.savetxt(os.path.join(args.output_dir, f"{input_file_name}_original.xyz"), points)
    
    # 预处理点云
    if len(points) > args.num_points:
        # 采样到指定点数
        indices = np.random.choice(len(points), args.num_points, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    elif len(points) < args.num_points:
        # 不足则重复采样
        indices = np.random.choice(len(points), args.num_points, replace=True)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    # 归一化
    points_normalized, centroid, scale = normalize_point_cloud(points, return_stats=True)
    
    # 转换为张量
    points_tensor = torch.from_numpy(points_normalized).float().unsqueeze(0).to(args.device)

    # 创建模型
    model = GeoFormerSeg(
        num_classes=model_cfg['num_classes'],
        num_layers=model_cfg.get('num_layers', 4),
        embed_dim=model_cfg.get('embed_dim', 256),
        num_heads=model_cfg.get('num_heads', 8),
        dropout=model_cfg.get('dropout', 0.5)
    ).to(args.device)

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # 处理分布式训练保存的模型
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    logger.info(f"成功加载模型权重，轮次: {checkpoint.get('epoch', 'unknown')}")

    # 设置为评估模式
    model.eval()

    # 推理
    with torch.no_grad():
        outputs = model(points_tensor)
        _, preds = torch.max(outputs, 1)
        
        # 获取几何特征（如果模型支持）
        if hasattr(model, 'get_geometry_features') and args.save_features:
            geometry_features = model.get_geometry_features()
            normals = geometry_features.get('normals', None)
            curvature = geometry_features.get('curvature', None)
        else:
            normals = None
            curvature = None
    
    # 获取预测结果
    predictions = preds.cpu().numpy()[0]
    
    # 保存结果
    output_points = np.hstack([points, predictions.reshape(-1, 1)])
    
    if normals is not None:
        normals = normals.cpu().numpy()[0]
        output_points = np.hstack([output_points, normals])
    
    if curvature is not None:
        curvature = curvature.cpu().numpy()[0]
        output_points = np.hstack([output_points, curvature.reshape(-1, 1)])
    
    # 保存带预测的点云
    output_file = os.path.join(args.output_dir, f"{input_file_name}_prediction.xyz")
    header = "x y z label"
    if normals is not None:
        header += " nx ny nz"
    if curvature is not None:
        header += " curvature"
    
    np.savetxt(output_file, output_points, fmt='%.6f', header=header)
    logger.info(f"预测结果已保存到: {output_file}")
    
    # 保存归一化信息（用于后续处理）
    with open(os.path.join(args.output_dir, "normalization.txt"), 'w') as f:
        f.write(f"Centroid: {centroid}\n")
        f.write(f"Scale: {scale}\n")

    # 可视化
    if args.visualize:
        # 可视化分割结果
        visualize_segmentation_result(points, predictions, gt_labels=None, num_classes=model_cfg['num_classes'])
        
        # 可视化几何特征
        if normals is not None:
            visualize_geometry_features(points, normals, title="法线特征")
        
        if curvature is not None:
            visualize_geometry_features(points, curvature, title="曲率特征")

if __name__ == '__main__':
    main()