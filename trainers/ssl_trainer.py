import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from .trainer_utils import geometric_augmentation, compute_geometric_consistency_loss
from utils import AverageMeter, ProgressMeter
import time
import numpy as np

class SSLTrainer(BaseTrainer):
    """几何一致的自监督预训练训练器"""
    
    def __init__(self, model, config, data_loader, logger=None):
        super().__init__(model, config, data_loader, logger)
        
        # 掩码比例
        self.mask_ratio = config.get('mask_ratio', 0.6)
        
        # 几何一致性权重
        self.lambda_normal = config.get('lambda_normal', 0.5)
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.config.optimizer.type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay
            )
        elif self.config.optimizer.type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                momentum=self.config.optimizer.momentum,
                weight_decay=self.config.optimizer.weight_decay,
                nesterov=self.config.optimizer.nesterov
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.optimizer.type}")
    
    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        if self.config.lr_scheduler.type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr_scheduler.min_lr
            )
        elif self.config.lr_scheduler.type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_scheduler.step_size,
                gamma=self.config.lr_scheduler.gamma
            )
        else:
            return None
    
    def _train_epoch(self, epoch):
        """训练一个轮次"""
        self.model.train()
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.data_loader['train_loader']),
            [batch_time, data_time, losses],
            prefix=f"Epoch: [{epoch+1}]"
        )
        
        end = time.time()
        
        for i, data in enumerate(self.data_loader['train_loader']):
            # 测量数据加载时间
            data_time.update(time.time() - end)
            
            # 获取数据
            points = data['points'].to(self.device)
            
            # 应用几何增强
            points = geometric_augmentation(points)
            
            # 随机掩码部分点
            masked_points, mask = self._create_mask(points)
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 前向传播
                reconstructed_points, reconstructed_normals = self.model(masked_points, mask)
                
                # 计算原始法线 (在训练器内部计算)
                from utils import estimate_normals
                original_normals = estimate_normals(points, k=8)
                
                # 计算几何一致性损失
                loss = compute_geometric_consistency_loss(
                    points, 
                    reconstructed_points, 
                    original_normals,
                    reconstructed_normals,
                    mask,
                    lambda_normal=self.lambda_normal
                )
            
            # 记录损失
            losses.update(loss.item(), points.size(0))
            
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 测量批次处理时间
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 打印进度
            if i % self.config.print_freq == 0 and self.logger:
                progress.display(i)
        
        return losses.avg, {}
    
    def _validate_epoch(self, epoch):
        """验证一个轮次"""
        self.model.eval()
        
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(self.data_loader['val_loader']),
            [batch_time, losses],
            prefix='验证: '
        )
        
        end = time.time()
        
        with torch.no_grad():
            for i, data in enumerate(self.data_loader['val_loader']):
                # 获取数据
                points = data['points'].to(self.device)
                
                # 随机掩码部分点
                masked_points, mask = self._create_mask(points)
                
                # 前向传播
                reconstructed_points, reconstructed_normals = self.model(masked_points, mask)
                
                # 计算原始法线
                from utils import estimate_normals
                original_normals = estimate_normals(points, k=8)
                
                # 计算几何一致性损失
                loss = compute_geometric_consistency_loss(
                    points, 
                    reconstructed_points, 
                    original_normals,
                    reconstructed_normals,
                    mask,
                    lambda_normal=self.lambda_normal
                )
                
                # 记录损失
                losses.update(loss.item(), points.size(0))
                
                # 测量批次处理时间
                batch_time.update(time.time() - end)
                end = time.time()
                
                # 打印进度
                if i % self.config.print_freq == 0 and self.logger:
                    progress.display(i)
        
        return losses.avg, {}
    
    def _create_mask(self, points, mask_ratio=None):
        """创建点云掩码"""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        batch_size, num_points, _ = points.shape
        
        # 为每个点云创建随机掩码
        mask = torch.rand(batch_size, num_points, device=self.device) > mask_ratio
        
        # 应用掩码
        masked_points = points.clone()
        masked_points[~mask.unsqueeze(-1).expand_as(points)] = 0.0
        
        return masked_points, mask