import argparse
import os
import torch
from models import GeoFormerSeg
from utils import setup_logger
import onnx
import onnxruntime as ort
import numpy as np
from utils.pointcloud import normalize_point_cloud

def main():
    parser = argparse.ArgumentParser(description='导出GeoFormer模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./output/export', help='输出目录')
    parser.add_argument('--onnx', action='store_true', help='导出为ONNX格式')
    parser.add_argument('--torchscript', action='store_true', help='导出为TorchScript格式')
    parser.add_argument('--num_points', type=int, default=4096, help='输入点云点数')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置日志
    logger = setup_logger('GeoFormer_export', args.output_dir, 0)
    logger.info(f'使用检查点: {args.checkpoint}')

    # 创建模型
    model = GeoFormerSeg(
        num_classes=13,  # 占位符，实际从检查点加载
        num_layers=6,
        embed_dim=256,
        num_heads=8,
        dropout=0.0  # 导出时禁用dropout
    ).eval()

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"成功加载模型权重，轮次: {checkpoint.get('epoch', 'unknown')}")

    # 创建虚拟输入
    dummy_points = np.random.randn(args.num_points, 3).astype(np.float32)
    dummy_points, _, _ = normalize_point_cloud(dummy_points, return_stats=True)
    dummy_input = torch.from_numpy(dummy_points).unsqueeze(0)  # [1, N, 3]

    # 导出ONNX模型
    if args.onnx:
        onnx_path = os.path.join(args.output_dir, 'geoformer.onnx')
        
        # 设置输入输出名称
        input_names = ['points']
        output_names = ['logits']
        
        # 导出模型
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=12,
            dynamic_axes={
                'points': {1: 'num_points'},  # 支持可变点数
                'logits': {1: 'num_points'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 测试ONNX推理
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        logger.info(f"ONNX模型已成功导出到: {onnx_path}")
        logger.info(f"ONNX模型输出形状: {ort_outs[0].shape}")

    # 导出TorchScript模型
    if args.torchscript:
        ts_path = os.path.join(args.output_dir, 'geoformer.pt')
        
        # 导出模型
        scripted_model = torch.jit.trace(model, dummy_input)
        scripted_model.save(ts_path)
        
        # 测试TorchScript推理
        loaded_model = torch.jit.load(ts_path)
        ts_output = loaded_model(dummy_input)
        
        logger.info(f"TorchScript模型已成功导出到: {ts_path}")
        logger.info(f"TorchScript模型输出形状: {ts_output.shape}")

    logger.info("模型导出完成")

if __name__ == '__main__':
    main()