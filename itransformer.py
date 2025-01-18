# 保持与 transformer.py 相同的导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import warnings
import argparse
import os
import csv
import json
from pathlib import Path
import sys

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 引入所有数据处理相关函数
from transformer import (
    load_data, preprocess_data, BikeDataset,
    train_model, evaluate_model, plot_losses, plot_predictions
)

# iTransformer 模型定义
class InvertedPositionalEncoding(nn.Module):
    """Feature-wise positional encoding"""
    def __init__(self, feature_dim, time_dim, dropout=0.1):
        super(InvertedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(1, feature_dim, time_dim)
        pos = torch.arange(0, feature_dim, dtype=torch.float).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, time_dim, 2).float() * (-np.log(10000.0) / time_dim))
        
        pe[0, :, 0::2] = torch.sin(pos * div_term[:time_dim//2].view(1, -1))
        if time_dim % 2 == 0:
            pe[0, :, 1::2] = torch.cos(pos * div_term[:time_dim//2].view(1, -1))
        else:
            pe[0, :, 1::2] = torch.cos(pos * div_term[:(time_dim-1)//2].view(1, -1))
            
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, feature_dim, time_dim]
        """
        x = x + self.pe
        return self.dropout(x)

class FeatureAttention(nn.Module):
    """Enhanced feature-wise attention mechanism"""
    def __init__(self, time_dim, nhead):
        super(FeatureAttention, self).__init__()
        self.attention = nn.MultiheadAttention(time_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(time_dim)
        
    def forward(self, x):
        # x shape: [batch_size, feature_dim, time_dim]
        orig_x = x
        x = x.transpose(1, 2)  # [batch_size, time_dim, feature_dim]
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.transpose(1, 2)  # [batch_size, feature_dim, time_dim]
        return self.norm(orig_x + attn_out)

class FeatureMultiScaleConv(nn.Module):
    """多尺度卷积 + GLU 门控"""
    def __init__(self, in_channels):
        super().__init__()
        # 不同卷积核
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        # 门控
        self.gate = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 多尺度卷积
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        # 拼接或加和
        out = x3 + x5

        # GLU 门控
        gate_out = torch.sigmoid(self.gate(x))
        out = out * gate_out
        return out

class iTransformerTimeSeries(nn.Module):
    """
    Improved Transformer with inverted architecture focusing on feature interactions
    """
    def __init__(self, feature_size=12, num_layers=3, dropout=0.1,
                 forward_expansion=2048, nhead=4, input_window=96,
                 output_window=96):
        super(iTransformerTimeSeries, self).__init__()
        
        self.feature_size = feature_size
        self.input_window = input_window
        self.output_window = output_window
        
        # Feature-wise position encoding
        self.pos_encoder = InvertedPositionalEncoding(
            feature_size, input_window, dropout)
        
        # Feature attention layers
        self.feature_attention_layers = nn.ModuleList([
            FeatureAttention(input_window, nhead) 
            for _ in range(num_layers)
        ])
        
        # Feature mixing with 1D convolutions
        self.conv_layers = nn.ModuleList([
            # nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
            # 将原先的卷积替换为多尺度卷积 + GLU 门控
            FeatureMultiScaleConv(feature_size)
            for _ in range(num_layers)
        ])
        
        # Temporal attention for sequence modeling
        self.temporal_attention = nn.MultiheadAttention(
            feature_size, nhead, batch_first=True)
        
        # Improved decoder network
        self.decoder = nn.Sequential(
            nn.Linear(input_window * feature_size, forward_expansion),
            nn.LayerNorm(forward_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion, forward_expansion // 2),
            nn.LayerNorm(forward_expansion // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion // 2, output_window)
        )
        
        # Additional layers for feature interaction
        self.feature_mixer = nn.Sequential(
            nn.Linear(feature_size, feature_size * 2),
            nn.ReLU(),
            nn.Linear(feature_size * 2, feature_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(input_window)
        self.norm2 = nn.LayerNorm(feature_size)

        # 跳跃连接转换层
        self.skip_connections = nn.ModuleList([
            nn.Linear(feature_size, feature_size)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        """
        Args:
            src: Input tensor [batch_size, input_window, feature_size]
        Returns:
            output: Predictions [batch_size, output_window]
        """
        # 1. Transpose input for feature-wise processing
        x = src.transpose(1, 2)  # [batch_size, feature_size, input_window]
        original_x = x  # 保存原始输入用于全局残差
        
        # 2. Apply positional encoding
        x = self.pos_encoder(x)

        # 存储跳跃连接特征
        skip_features = []
        
        # 3. Feature attention and convolution processing
        for feat_attn, conv, skip in zip(
            self.feature_attention_layers, 
            self.conv_layers,
            self.skip_connections  # 添加跳跃连接
        ):
            # Feature attention
            # x = feat_attn(x)
            x_attn = feat_attn(x)
            x = x + x_attn  # 第一个残差连接

            # Convolution for local feature patterns
            x_conv = conv(x)   

            # Residual connection
            # x = x + x_conv
            x = x + x_conv  # 第二个残差连接
            x = self.norm1(x.transpose(1, 2)).transpose(1, 2)

            # 保存跳跃特征
            skip_features.append(
                skip(x.transpose(1, 2)).transpose(1, 2)
            )
        
        # 4. Feature mixing
        b, f, t = x.shape
        x_mixed = self.feature_mixer(x.transpose(1, 2))
        x = x + x_mixed.transpose(1, 2)
        
        # 5. Temporal attention
        x_temporal = x.transpose(1, 2)  # [batch_size, input_window, feature_size]
        temporal_out, _ = self.temporal_attention(x_temporal, x_temporal, x_temporal)
        x = x + temporal_out.transpose(1, 2)  # 时序注意力残差

        # 6. 全局残差连接
        x = x + original_x + sum(skip_features)

        # 7. 最终层归一化
        x = self.norm2(x.transpose(1, 2))  
        # x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        
        # # 6. Final prediction
        # x = x.reshape(b, -1)  # Flatten for final prediction
        # output = self.decoder(x)

        # 8. 解码预测
        b, t, f = x.shape  
        x = x.reshape(b, -1)  # 展平特征
        # 或者写成：x = x.reshape(x.size(0), -1)  # 展平特征
        output = self.decoder(x)
        
        return output

# 使用与transformer.py相同的exp_configs
exp_configs = [
    {
        'description': f'固定学习率(0.001) - 随机种子 {seed}',
        'seed': seed
    } for seed in [42, 123, 456, 789, 1024]  # 使用不同的随机种子
]

def run_experiment(train_loader, val_loader, test_loader, scaler_cnt, features, target, prediction_type, input_window,
                   output_window, num_experiments=5, epochs=10, resume=False):
    """
    使用与transformer.py相同的实验运行逻辑，但使用iTransformerTimeSeries替代TransformerTimeSeries
    """
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # 仅在 resume=True 时尝试加载实验进度
    progress_file = Path('checkpoints/progress_itransformer.json')  # 修改文件名以区分
    if resume and progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_exp = progress['current_exp']
        print(f"Resuming from experiment {start_exp}")
    else:
        start_exp = 0
        progress = {'current_exp': 0}
        # 如果不是恢复训练，则删除旧的检查点文件
        if not resume:
            for f in Path('checkpoints').glob('checkpoint_itransformer_exp*.pt'):  # 修改文件名以区分
                f.unlink()
            if progress_file.exists():
                progress_file.unlink()

    mse_list = []
    mae_list = []

    try:
        for exp in range(start_exp, num_experiments):
            config = exp_configs[exp]
            print(f"\nExperiment {exp + 1}/{num_experiments} - {config['description']}")
            print(f"Random seed: {config['seed']}")
            
            set_seed(config['seed'])
            
            # 使用 iTransformerTimeSeries 替代 TransformerTimeSeries
            model = iTransformerTimeSeries(
                feature_size=len(features),
                num_layers=3,
                dropout=0.1,
                forward_expansion=2048,
                nhead=4,
                input_window=input_window,
                output_window=output_window
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            save_path = os.path.join('saved_models', f'itransformer_model_{prediction_type}_exp{exp + 1}.pth')

            progress['current_exp'] = exp
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

            # 使用相同的训练函数
            train_losses, val_losses = train_model(
                model, train_loader, val_loader, criterion,
                optimizer, epochs, save_path, scheduler=None,
                checkpoint_dir='checkpoints', exp_id=f'itransformer_{exp + 1}'  # 修改检查点ID以区分
            )

            plot_losses(train_losses, val_losses, f"iTransformer_Experiment_{exp + 1} - {prediction_type.capitalize()}")

            # 评估部分保持不变
            mse, mae, preds, trues = evaluate_model(model, test_loader, scaler_cnt)
            print(f"Experiment {exp + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}")
            mse_list.append(mse)
            mae_list.append(mae)

            model.load_state_dict(torch.load(save_path, weights_only=True))

            # 预测和绘图部分保持不变
            model.eval()
            with torch.no_grad():
                X, y, input_cnt = next(iter(test_loader))
                print("数据形状:")
                print(f"X (输入特征): {X.shape}")
                print(f"y (目标值): {y.shape}")
                print(f"input_cnt (输入序列): {input_cnt.shape}")
                X = X.to(device)
                y = y.to(device)
                predictions = model(X).cpu().numpy()
                true = y.cpu().numpy()
                predictions = scaler_cnt.inverse_transform(predictions)
                true = scaler_cnt.inverse_transform(true)
                plot_predictions(input_cnt[0], true[0], predictions[0],
                               f'iTransformer_Experiment_{exp + 1} - {prediction_type.capitalize()} Predictions vs Ground Truth',
                               scaler_cnt)

    except KeyboardInterrupt:
        print("\n训练被中断，已保存进度。下次运行时将从中断处继续。")
        sys.exit(0)

    # 结果统计保持不变
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)

    print(f"\nFinal Results for iTransformer {prediction_type.capitalize()} Prediction:")
    print(f"MSE: Mean = {mse_mean:.4f}, Std = {mse_std:.4f}")
    print(f"MAE: Mean = {mae_mean:.4f}, Std = {mae_std:.4f}")

def main():
    # 保持与transformer.py相同的参数解析和主流程
    parser = argparse.ArgumentParser(description='Bike Rental Forecasting using iTransformer')
    parser.add_argument('--mode', type=str, choices=['short', 'long'], default='short',
                       help='Prediction mode: "short" for short-term (96 hours), "long" for long-term (240 hours)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint if available')
    args = parser.parse_args()

    # 其余主函数逻辑与transformer.py完全相同
    train_path = 'train_data.csv'
    test_path = 'test_data.csv'
    
    train_df, test_df = load_data(train_path, test_path)
    train, val, test, scaler_features, scaler_cnt, features, target = preprocess_data(train_df, test_df)

    if args.mode == 'short':
        input_window = 96
        output_window = 96
        prediction_type = 'short'
    else:
        input_window = 96
        output_window = 240
        prediction_type = 'long'

    print(f"\nRunning iTransformer {prediction_type.capitalize()} Prediction with input window = {input_window} and output window = {output_window}")

    # 数据集创建
    dataset_train = BikeDataset(train, input_window, output_window, features, target)
    dataset_val = BikeDataset(val, input_window, output_window, features, target)
    dataset_test = BikeDataset(test, input_window, output_window, features, target)

    # 数据加载器
    batch_size = 64
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # 运行实验
    run_experiment(train_loader, val_loader, test_loader, scaler_cnt, features, target, prediction_type, input_window,
                  output_window, num_experiments=args.experiments, epochs=args.epochs, resume=args.resume)

if __name__ == "__main__":
    main()
