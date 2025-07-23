import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import torch.cuda.amp

def init_distributed_mode(args):
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)
    
    # 初始化混合精度scaler
    if args.use_amp:
        args.scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        args.scaler = None

def setup_for_distributed(is_master):
    """确保只有主进程会输出信息"""
    import builtins as __builtin__
    builtin_print = __builtin__.print
    
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def get_world_size():
    """获取分布式训练的进程数"""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """获取当前进程的rank"""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    """判断当前进程是否为主进程"""
    return get_rank() == 0

def reduce_dict(input_dict, average=True):
    """归约字典中的值到所有进程"""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        # 确保字典中的项按相同顺序处理
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict

def reduce_tensor(tensor, average=True):
    """归约张量到所有进程"""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    
    with torch.no_grad():
        dist.all_reduce(tensor)
        if average:
            tensor /= world_size
    
    return tensor

def distributed_model(model, device, find_unused_parameters=False):
    """将模型转换为分布式数据并行模式"""
    if device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device],
            output_device=device,
            find_unused_parameters=find_unused_parameters
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=find_unused_parameters
        )
    return model

def sync_metrics(metrics, average=True):
    """同步多个进程间的指标"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return metrics
    
    # 转换为张量
    metrics_tensor = torch.tensor(
        [metrics.get(k, 0.0) for k in sorted(metrics.keys())],
        dtype=torch.float32,
        device='cuda'
    )
    
    # 归约指标
    dist.all_reduce(metrics_tensor)
    if average:
        metrics_tensor /= dist.get_world_size()
    
    # 转换回字典
    synced_metrics = {k: v.item() for k, v in zip(sorted(metrics.keys()), metrics_tensor)}
    return synced_metrics