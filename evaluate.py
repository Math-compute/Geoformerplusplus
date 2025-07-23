import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import PointCloudDataset
from models import GeoFormerSeg
from utils import setup_logger, set_seed, evaluate_segmentation
from tqdm import tqdm
import yaml
import json

def main():
    parser = argparse.ArgumentParser(description='GeoFormer模型评估')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./output/eval', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='使用的设备')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载工作线程数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--save_results', action='store_true', help='保存评估结果')
    parser.add_argument('--compute_boundary_metrics', action='store_true', help='计算边界指标')
    parser.add_argument('--compute_geometry_metrics', action='store_true', help='计算几何指标')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    logger = setup_logger('GeoFormer_eval', args.output_dir, 0)
    logger.info(f'使用检查点: {args.checkpoint}')

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据集
    test_dataset = PointCloudDataset(
        root=config['data']['root_dir'],
        split='test',
        num_points=config['data']['num_points'],
        augment=False,
        with_labels=True
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 创建模型
    model = GeoFormerSeg(
        num_classes=config['model']['num_classes'],
        num_layers=config['model'].get('num_layers', 4),
        embed_dim=config['model'].get('embed_dim', 256),
        num_heads=config['model'].get('num_heads', 8),
        dropout=config['model'].get('dropout', 0.5)
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

    # 存储所有预测和标签
    all_preds = []
    all_labels = []
    all_points = []
    all_boundary_masks = []
    all_normals = []

    # 评估
    with torch.no_grad():
        for data in tqdm(test_loader, desc='评估'):
            points = data['points'].to(args.device)
            labels = data['labels'].to(args.device)
            
            # 前向传播
            outputs = model(points)
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            # 存储结果
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_points.append(points.cpu().numpy())
            
            # 如果需要计算边界指标
            if args.compute_boundary_metrics:
                boundary_mask = model.get_boundary_mask(labels)  # 假设模型有这个方法
                all_boundary_masks.append(boundary_mask.cpu().numpy())
            
            # 如果需要计算几何指标
            if args.compute_geometry_metrics:
                normals = model.get_normals()  # 假设模型有这个方法
                all_normals.append(normals.cpu().numpy())
    
    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_points = np.concatenate(all_points)
    
    if args.compute_boundary_metrics:
        all_boundary_masks = np.concatenate(all_boundary_masks)
    else:
        all_boundary_masks = None
    
    if args.compute_geometry_metrics:
        all_normals = np.concatenate(all_normals)
    else:
        all_normals = None

    # 计算评估指标
    metrics = evaluate_segmentation(
        all_preds, 
        all_labels, 
        config['model']['num_classes'],
        points=all_points if args.compute_geometry_metrics else None,
        pred_normals=all_normals if args.compute_geometry_metrics else None,
        boundary_mask=all_boundary_masks if args.compute_boundary_metrics else None
    )
    
    # 打印评估结果
    logger.info(f"总体准确率: {metrics['accuracy']:.4f}")
    logger.info(f"平均类别准确率: {metrics['mean_class_accuracy']:.4f}")
    logger.info(f"mIoU: {metrics['miou']:.4f}")
    
    if 'boundary_iou' in metrics:
        logger.info(f"边界区域IoU: {metrics['boundary_iou']:.4f}")
    
    if 'normal_consistency' in metrics:
        logger.info(f"法线一致性(度): {metrics['normal_consistency']:.2f}")
    
    # 保存详细结果
    if args.save_results:
        # 保存预测结果
        np.save(os.path.join(args.output_dir, 'predictions.npy'), all_preds)
        np.save(os.path.join(args.output_dir, 'labels.npy'), all_labels)
        np.save(os.path.join(args.output_dir, 'points.npy'), all_points)
        
        if all_boundary_masks is not None:
            np.save(os.path.join(args.output_dir, 'boundary_masks.npy'), all_boundary_masks)
        
        if all_normals is not None:
            np.save(os.path.join(args.output_dir, 'normals.npy'), all_normals)
        
        # 保存指标为JSON
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"结果已保存到: {args.output_dir}")

if __name__ == '__main__':
    main()