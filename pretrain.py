import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from datasets import PointCloudDataset
from models import GeoFormer
from trainers import SSLTrainer
from utils import setup_logger, set_seed, init_distributed_mode, save_training_config
import yaml

def main():
    parser = argparse.ArgumentParser(description='GeoFormer自监督预训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='./output/pretrain', help='输出目录')
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
    logger = setup_logger('GeoFormer_pretrain', args.output_dir, 0, distributed_rank=args.rank)
    
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
    
    # 创建数据集
    train_dataset = PointCloudDataset(
        root=config['data']['root_dir'],
        split='train',
        num_points=config['data']['num_points'],
        augment=True
    )
    val_dataset = PointCloudDataset(
        root=config['data']['root_dir'],
        split='val',
        num_points=config['data']['num_points'],
        augment=False
    )

    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(train_sampler is None),
        num_workers=config['train'].get('num_workers', 4),
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train'].get('num_workers', 2),
        pin_memory=True,
        sampler=val_sampler
    )

    # 创建模型
    model = GeoFormer(
        num_layers=config['model'].get('num_layers', 6),
        embed_dim=config['model'].get('embed_dim', 256),
        num_heads=config['model'].get('num_heads', 8),
        dropout=config['model'].get('dropout', 0.1)
    ).cuda()

    # 分布式训练包装
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )

    # 创建训练器
    trainer = SSLTrainer(
        model=model,
        config=config,
        data_loader={
            'train_loader': train_loader,
            'val_loader': val_loader
        },
        logger=logger,
        use_amp=args.use_amp
    )

    # 恢复训练
    if args.resume:
        trainer._resume_checkpoint(args.resume)

    # 开始预训练
    trainer.train()

    # 清理分布式训练
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()