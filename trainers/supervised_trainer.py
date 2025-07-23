import torch
import torch.nn as nn
from .base_trainer import BaseTrainer
from .trainer_utils import compute_class_iou, compute_boundary_iou, get_boundary_mask
from utils import compute_overall_accuracy, compute_mean_class_accuracy, compute_iou
from utils import AverageMeter, ProgressMeter
import time
import numpy as np

class SupervisedTrainer(BaseTrainer):
    """监督学习训练器，用于语义分割任务"""
    
    def __init__(self, model, config, data_loader, logger=None):
        super().__init__(model, config, data_loader, logger)
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        
        # 类别数
        self.num_classes = config.num_classes
        
        # 边界损失权重
        self.boundary_weight = config.get('boundary_weight', 1.0)
    
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
        elif self.config.lr_scheduler.type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_scheduler.gamma,
                patience=self.config.lr_scheduler.patience,
                verbose=True
            )
        else:
            return None
    
    def _train_epoch(self, epoch):
        """训练一个轮次"""
        self.model.train()
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        accuracy = AverageMeter('Acc', ':6.2f')
        boundary_iou = AverageMeter('BoundaryIoU', ':6.2f')
        
        progress = ProgressMeter(
            len(self.data_loader['train_loader']),
            [batch_time, data_time, losses, accuracy, boundary_iou],
            prefix=f"Epoch: [{epoch+1}]"
        )
        
        end = time.time()
        
        all_preds = []
        all_labels = []
        
        for i, data in enumerate(self.data_loader['train_loader']):
            # 测量数据加载时间
            data_time.update(time.time() - end)
            
            # 获取数据
            points = data['points'].to(self.device)
            labels = data['labels'].to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 前向传播
                outputs = self.model(points)
                
                # 计算边界掩码
                boundary_mask = get_boundary_mask(labels, ignore_index=self.config.ignore_index)
                
                # 计算分割损失
                seg_loss = self.criterion(outputs, labels)
                
                # 计算边界损失
                if self.boundary_weight > 0:
                    boundary_outputs = outputs[boundary_mask]
                    boundary_labels = labels[boundary_mask]
                    boundary_loss = self.criterion(boundary_outputs, boundary_labels)
                    loss = seg_loss + self.boundary_weight * boundary_loss
                else:
                    loss = seg_loss
            
            # 计算准确率
            _, preds = torch.max(outputs, 1)
            acc = compute_overall_accuracy(preds, labels)
            
            # 计算边界IoU
            if boundary_mask.any():
                b_iou, _ = compute_boundary_iou(preds.cpu().numpy(), 
                                               labels.cpu().numpy(), 
                                               boundary_mask.cpu().numpy(),
                                               self.num_classes)
                boundary_iou.update(b_iou, points.size(0))
            
            # 记录损失和准确率
            losses.update(loss.item(), points.size(0))
            accuracy.update(acc, points.size(0))
            
            # 存储预测和标签用于计算IoU
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
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
        
        # 计算IoU
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        iou, miou = compute_iou(all_preds, all_labels, self.num_classes)
        class_iou = compute_class_iou(all_preds, all_labels, self.num_classes)
        
        return losses.avg, {
            'accuracy': accuracy.avg,
            'miou': miou,
            'boundary_iou': boundary_iou.avg,
            'class_iou': class_iou  # 每个类别的IoU
        }
    
    def _validate_epoch(self, epoch):
        """验证一个轮次"""
        self.model.eval()
        
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        accuracy = AverageMeter('Acc', ':6.2f')
        boundary_iou = AverageMeter('BoundaryIoU', ':6.2f')
        
        progress = ProgressMeter(
            len(self.data_loader['val_loader']),
            [batch_time, losses, accuracy, boundary_iou],
            prefix='验证: '
        )
        
        end = time.time()
        
        all_preds = []
        all_labels = []
        all_class_iou = []
        
        with torch.no_grad():
            for i, data in enumerate(self.data_loader['val_loader']):
                # 获取数据
                points = data['points'].to(self.device)
                labels = data['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(points)
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                
                # 计算边界掩码
                boundary_mask = get_boundary_mask(labels, ignore_index=self.config.ignore_index)
                
                # 计算准确率
                _, preds = torch.max(outputs, 1)
                acc = compute_overall_accuracy(preds, labels)
                
                # 计算边界IoU
                if boundary_mask.any():
                    b_iou, _ = compute_boundary_iou(preds.cpu().numpy(), 
                                                   labels.cpu().numpy(), 
                                                   boundary_mask.cpu().numpy(),
                                                   self.num_classes)
                    boundary_iou.update(b_iou, points.size(0))
                
                # 记录损失和准确率
                losses.update(loss.item(), points.size(0))
                accuracy.update(acc, points.size(0))
                
                # 存储预测和标签用于计算IoU
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                # 计算每个批次的类别IoU
                batch_class_iou = compute_class_iou(preds.cpu().numpy(), 
                                                   labels.cpu().numpy(), 
                                                   self.num_classes)
                all_class_iou.append(batch_class_iou)
                
                # 测量批次处理时间
                batch_time.update(time.time() - end)
                end = time.time()
                
                # 打印进度
                if i % self.config.print_freq == 0 and self.logger:
                    progress.display(i)
        
        # 计算评估指标
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        iou, miou = compute_iou(all_preds, all_labels, self.num_classes)
        mca = compute_mean_class_accuracy(all_preds, all_labels, self.num_classes)
        
        # 计算平均类别IoU
        class_iou = np.nanmean(np.stack(all_class_iou, axis=0), axis=0)
        
        return losses.avg, {
            'accuracy': accuracy.avg,
            'miou': miou,
            'mca': mca,
            'boundary_iou': boundary_iou.avg,
            'class_iou': class_iou  # 每个类别的平均IoU
        }