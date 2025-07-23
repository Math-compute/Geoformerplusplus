import os
import torch
import time
import datetime
import numpy as np
import json

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """显示训练进度"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename=None):
    """保存模型检查点（支持额外信息）"""
    # 保存当前检查点
    torch.save(state, filename)
    
    # 保存最佳模型
    if is_best:
        if best_filename is None:
            best_filename = filename.replace('checkpoint', 'model_best')
        torch.save(state, best_filename)
        print(f"Saved best model to {best_filename}")

def save_training_config(config, save_dir):
    """保存训练配置到JSON文件"""
    config_dict = vars(config) if not isinstance(config, dict) else config
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    return config_path

def load_training_config(config_path):
    """从JSON文件加载训练配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def time_since(start_time):
    """计算从开始时间起经过的时间"""
    seconds = int(time.time() - start_time)
    return str(datetime.timedelta(seconds=seconds))

def seed_everything(seed=42):
    """设置所有随机种子以保证可重复性"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_model(model):
    """冻结模型所有参数"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    """解冻模型所有参数"""
    for param in model.parameters():
        param.requires_grad = True

def get_device():
    """获取可用设备 (GPU或CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enable_cudnn_benchmark():
    """启用CuDNN benchmark以加速训练"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def memory_usage_report():
    """打印当前内存使用情况报告"""
    if torch.cuda.is_available():
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"保留内存: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    else:
        print("无可用GPU内存信息")