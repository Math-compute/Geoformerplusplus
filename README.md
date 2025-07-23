<!--
GeoFormerPlusPlus/
├── configs/                  # 配置文件目录
│   ├── semantickitti.yaml    # SemanticKITTI数据集配置
│   ├── s3dis.yaml            # S3DIS数据集配置
│   └── model_configs/        # 模型配置子目录
│       ├── geoformer_pp_base.yaml
│       └── geoformer_pp_large.yaml
│
├── datasets/                 # 数据集处理模块
│   ├── __init__.py
│   ├── base_dataset.py       # 基类定义
│   ├── semantickitti.py      # SemanticKITTI数据集处理
│   ├── s3dis.py              # S3DIS数据集处理
│   ├── scannet.py            # ScanNet数据集处理
│   └── transforms.py         # 数据增强和预处理
│
├── models/                   # 模型定义模块
│   ├── __init__.py
│   ├── geoformer_pp.py       # 主模型定义
│   ├── geometric_modules.py  # 几何感知模块
│   ├── transformer_modules.py # Transformer相关模块
│   ├── decoder_modules.py    # 解码器模块
│   └── self_supervised.py    # 自监督学习模块
│
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── pointcloud.py         # 点云处理工具
│   ├── metrics.py            # 评估指标计算
│   ├── visualization.py      # 可视化工具
│   ├── logger.py             # 日志记录
│   └── distributed.py        # 分布式训练工具
│
├── trainers/                 # 训练和评估模块
│   ├── __init__.py
│   ├── base_trainer.py       # 基类定义
│   ├── supervised_trainer.py # 监督学习训练器
│   └── ssl_trainer.py        # 自监督学习训练器
│
├── pretrain.py               # 自监督预训练入口
├── train.py                  # 监督训练入口
├── evaluate.py               # 评估入口
├── inference.py              # 推理入口
└── README.md                 # 项目说明
-->
1. s3dis.py (数据集加载)
问题：
    没有正确加载标签数据
    分块处理逻辑不完善
    缺少必要的数据增强
2. train.py (训练脚本)
问题：
    模型初始化不正确
    数据加载方式错误
    损失函数维度问题
    缺少关键配置项
3. geoformer.py (模型定义)
问题：
    输入维度固定为3，无法处理RGB特征
    缺少分类头 
4. pointcloud_dataset.py (数据集基类)
问题：
缺少实际数据集实现
数据增强未实现
5. geoformer_seg.py (分割模型)
问题：
    缺少必要的模块导入
    FPSDownsample未定义
6. pretrain.py (预训练脚本)
问题：
    配置加载未实现
    缺少关键参数
7. inference.py (推理脚本)
问题：
    配置硬编码
    缺少模型参数
8. self_supervised.py (自监督模块)
问题：
    掩码生成效率低
    点云重建损失计算不准确