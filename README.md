# 自行车租赁需求预测系统

基于 Transformer 架构的时序预测系统，提供标准 Transformer 和改进的 iTransformer 两种实现，用于预测共享单车的租赁需求。

## 项目结构

```txt
ML_Bike_Rental_Forecast/
├── baseline/                # 标准 Transformer 实现
│   ├── models/
│   │   └── transformer.py  # 标准 Transformer 模型
│   └── utils/
│       └── __init__.py
├── improved/               # 改进版实现
│   ├── models/
│   │   └── itransformer.py # iTransformer 模型
│   └── utils/
│       └── __init__.py
├── data/                   # 数据目录
│   ├── train_data.csv
│   └── test_data.csv
├── output/                 # 输出目录
│   ├── saved_models/      # 保存的模型
│   ├── plots/             # 可视化结果
│   │   ├── losses/
│   │   └── predictions/
│   └── checkpoints/       # 训练检查点
└── README.md
```
## 模型架构

### 标准 Transformer (transformer.py)
- 基于标准 Transformer 编码器-解码器架构
- 使用时序注意力机制和位置编码
- 适用于一般时序预测任务

### 改进版 iTransformer (itransformer.py)
创新特点：
- 特征维度注意力：将特征作为序列长度维度进行建模
- 多尺度特征提取：使用不同卷积核大小捕获局部模式
- 增强型特征交互：通过特征混合层和跳跃连接加强特征间关系
- GLU门控机制：改善信息流动和梯度传播
- Huber Loss：平衡MSE和MAE的优势，提高鲁棒性

## 支持功能

- 短期预测(96小时)和长期预测(240小时)
- 断点续训和模型检查点保存
- 自动保存训练日志和可视化结果
- 实验结果的统计分析(MSE/MAE均值和标准差)
- 多次实验自动化运行(支持不同随机种子)

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- pandas
- numpy
- matplotlib
- scikit-learn

## 安装依赖

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## 使用方法

1. 准备数据

   ```bash
   # 确保数据文件位于 data/ 目录下
   cp your_train_data.csv data/train_data.csv
   cp your_test_data.csv data/test_data.csv
   ```

2. 运行标准 Transformer

   ```bash
      # 运行短期预测
   ./run_transformer.sh short

   # 运行长期预测
   ./run_transformer.sh long

   # 从检查点恢复训练
   ./run_transformer.sh short --resume
   ```

   ```bash
   # 短期预测
   python -m baseline.models.transformer --mode short --epochs 100
   
   # 长期预测
   python -m baseline.models.transformer --mode long --epochs 100
   ```

3. 运行改进版 iTransformer

   ```bash
   # 短期预测
   python -m Improved-iTransformer.models.itransformer --mode short --epochs 100
   
   # 长期预测
   python -m Improved-iTransformer.models.itransformer --mode long --epochs 100
   ```

## 输出目录说明

- `output/saved_models/`: 模型保存位置
  - `transformer/`: 标准 Transformer 模型
  - `itransformer/`: 改进版模型
- `output/plots/`: 可视化结果
  - `transformer/`: 标准模型结果
  - `itransformer/`: 改进版结果
- `output/checkpoints/`: 训练检查点
  - `transformer/`: 标准模型检查点
  - `itransformer/`: 改进版检查点

## 参数说明

### 命令行参数

- `--mode`: 预测模式 ['short'|'long']
- `--epochs`: 训练轮数
- `--experiments`: 实验重复次数
- `--resume`: 从检查点恢复训练

## 实验结果对比

可以通过运行两个模型的实验，对比以下指标：

- MSE (均方误差)
- MAE (平均绝对误差)
- 预测的稳定性 (多次实验的标准差)
- 计算效率和训练时间

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
