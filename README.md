# 自行车租赁需求预测系统

基于 Transformer 模型的时序预测系统，用于预测共享单车的租赁需求。支持短期（96小时）和长期（240小时）预测。

## 项目特点

- 使用 Transformer 架构进行时序预测
- 支持短期（96小时）和长期（240小时）预测
- 实现断点续训功能
- 自动保存训练日志和可视化结果
- 支持学习率自适应调整

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

## 文件结构

```txt
ML_Bike_Rental_Forecast/
├── transformer.py     # 主要模型和训练代码
├── run_transformer.sh # 训练脚本
├── train_data.csv    # 训练数据
├── test_data.csv     # 测试数据
├── saved_models/     # 保存的模型
├── plots/           
│   ├── losses/      # 损失曲线
│   └── predictions/ # 预测结果可视化
└── checkpoints/     # 训练检查点
```

## 使用方法

1. 准备数据
   - 确保 `train_data.csv` 和 `test_data.csv` 在项目根目录

2. 运行训练

   ```bash
   # 运行短期预测
   ./run_transformer.sh short

   # 运行长期预测
   ./run_transformer.sh long

   # 从检查点恢复训练
   ./run_transformer.sh short --resume
   ```

3. 直接使用 Python 运行

    ```bash
    # 短期预测
    python transformer.py --mode short --epochs 100 --experiments 5

    # 长期预测
    python transformer.py --mode long --epochs 100 --experiments 5

    # 从检查点恢复训练
    python transformer.py --mode short --epochs 100 --experiments 5 --resume
    ```

## 输出说明

- `saved_models/`: 保存训练好的模型
- `plots/losses/`: 训练过程中的损失曲线
- `plots/predictions/`: 预测结果可视化
- `checkpoints/`: 训练检查点，用于断点续训

## 参数说明

- `--mode`: 预测模式 ['short'|'long']
- `--epochs`: 训练轮数
- `--experiments`: 实验重复次数
- `--resume`: 从检查点恢复训练
