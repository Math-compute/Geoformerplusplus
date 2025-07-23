import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from datasets import PointCloudDataset
from models import GeoFormerSeg
from trainers import SupervisedTrainer
from utils import setup_logger, set_seed, init_distributed_mode, save_training_config
import yaml

def main():
    parser = argparse.ArgumentParser(description='GeoFormer监督训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--pretrained', type=str, default='', help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default='./output/train', help='输出目录')
    parser.add_argument('--dist_url', default='env://', help='分布式训练URL')
    parser.add_argument('--use_amp', action='store_true', help='启用混合精度训练')
    parser.add_argument('--world_size', default=1, type=int, help='分布式训练节点数')
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    args = parser.parse_args()

    # 初始化分布式训练
    if args.distributed:
        init_distributed_mode(args)
    
    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    is_main_process = not args.distributed or (args.distributed and args.rank == 0)
    logger = setup_logger('GeoFormer_train', args.output_dir, 0, distributed_rank=args.rank)
    
    if is_main_process:
        logger.info(f'使用配置文件: {args.config}')
        logger.info(f'输出目录: {args.output_dir}')
        logger.info(f'使用混合精度: {args.use_amp}')
        logger.info(f'分布式训练: {args.distributed}, 当前Rank: {args.rank}')

    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 保存训练配置
    if is_main_process:
        save_training_config(config, args.output_dir)
    
    # 适配嵌套结构
    data_cfg = config['data']
    train_cfg = config['train']
    model_cfg = config['model'] if 'model' in config else {}

    # 创建数据集
    train_dataset = PointCloudDataset(
        root=data_cfg['root_dir'],
        split='train',
        num_points=data_cfg['num_points'],
        augment=True,
        with_labels=True
    )
    val_dataset = PointCloudDataset(
        root=data_cfg['root_dir'],
        split='val',
        num_points=data_cfg['num_points'],
        augment=False,
        with_labels=True
    )

    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 2),
        pin_memory=True,
        sampler=val_sampler
    )

    # 创建模型
    model = GeoFormerSeg(
        num_classes=model_cfg.get('num_classes', 13),
        num_layers=model_cfg.get('num_layers', 4),
        embed_dim=model_cfg.get('embed_dim', 256),
        num_heads=model_cfg.get('num_heads', 8),
        dropout=model_cfg.get('dropout', 0.5),
        boundary_weight=model_cfg.get('boundary_weight', 1.0)
    ).cuda()

    # 分布式训练包装
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )

    # 创建训练器
    trainer = SupervisedTrainer(
        model=model,
        config=config,
        data_loader={
            'train_loader': train_loader,
            'val_loader': val_loader
        },
        logger=logger,
        use_amp=args.use_amp
    )

    # 恢复训练或加载预训练模型
    if args.resume:
        trainer._resume_checkpoint(args.resume)
    elif args.pretrained:
        # 从预训练模型加载权重
        pretrained_dict = torch.load(args.pretrained, map_location='cuda')
        
        # 处理分布式模型前缀
        if args.distributed and 'module.' not in list(pretrained_dict.keys())[0]:
            pretrained_dict = {f'module.{k}': v for k, v in pretrained_dict.items()}
        elif not args.distributed and 'module.' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        if is_main_process:
            logger.info(f"成功加载预训练模型: {args.pretrained}")

    # 开始训练
    trainer.train()

    # 清理分布式训练
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()