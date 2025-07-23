import os
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from utils import AverageMeter, ProgressMeter, save_checkpoint


class BaseTrainer(ABC):
    """训练器基类，定义通用训练流程"""
    
    def __init__(self, model, config, data_loader, logger=None):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.logger = logger
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # 创建保存目录
        self.save_dir = config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 初始化学习率调度器
        self.lr_scheduler = self._create_lr_scheduler()
        
        # 训练状态
        self.start_epoch = 0
        self.best_metric = 0.0
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # 恢复训练
        if config.resume:
            self._resume_checkpoint(config.resume)
    
    @abstractmethod
    def _create_optimizer(self):
        """创建优化器"""
        pass
    
    @abstractmethod
    def _create_lr_scheduler(self):
        """创建学习率调度器"""
        pass
    
    @abstractmethod
    def _train_epoch(self, epoch):
        """训练一个轮次"""
        pass
    
    @abstractmethod
    def _validate_epoch(self, epoch):
        """验证一个轮次"""
        pass
    
    def _resume_checkpoint(self, resume_path):
        """恢复训练检查点"""
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.best_metric = checkpoint.get('best_metric', 0.0)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            if self.lr_scheduler and 'lr_scheduler' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            # 恢复混合精度scaler状态
            if 'scaler' in checkpoint and self.use_amp:
                self.scaler.load_state_dict(checkpoint['scaler'])
            
            if self.logger:
                self.logger.info(f"恢复训练从轮次 {self.start_epoch}")
                self.logger.info(f"最佳指标: {self.best_metric:.4f}")
        else:
            raise FileNotFoundError(f"检查点文件不存在: {resume_path}")
    
    def train(self):
        """执行完整训练过程"""
        if self.logger:
            self.logger.info(f"开始训练，总轮次: {self.config.epochs}")
        
        for epoch in range(self.start_epoch, self.config.epochs):
            if self.logger:
                self.logger.info(f"===== 轮次 [{epoch+1}/{self.config.epochs}] =====")
                self.logger.info(f"学习率: {self.optimizer.param_groups[0]['lr']}")
            
            # 训练一个轮次
            train_loss, train_metrics = self._train_epoch(epoch)
            
            # 验证一个轮次
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            # 更新学习率
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            
            # 保存检查点
            current_metric = val_metrics.get('miou', val_metrics.get('accuracy', 0.0))
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_metric': self.best_metric,
                'scaler': self.scaler.state_dict() if self.use_amp else None
            }
            
            if self.lr_scheduler:
                checkpoint_state['lr_scheduler'] = self.lr_scheduler.state_dict()
            
            save_checkpoint(
                checkpoint_state,
                is_best,
                filename=os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth.tar')
            )
            
            # 打印本轮结果
            if self.logger:
                self.logger.info(f"训练结果: Loss={train_loss:.4f}, " + 
                                 ', '.join([f"{k}={v:.4f}" for k, v in train_metrics.items()]))
                self.logger.info(f"验证结果: Loss={val_loss:.4f}, " + 
                                 ', '.join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))
                self.logger.info(f"最佳 {list(val_metrics.keys())[0]}: {self.best_metric:.4f}")
        
        if self.logger:
            self.logger.info(f"训练完成，最佳指标: {self.best_metric:.4f}")