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


# 数据预处理
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    # 合并训练和测试数据以进行统一的预处理
    combined = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

    # 处理日期
    combined['dteday'] = pd.to_datetime(combined['dteday'])

    # 选择特征
    features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'

    # 填充缺失值（如果有）
    combined.ffill(inplace=True)  # 使用 ffill 代替 fillna

    # 标准化数值特征
    scaler_features = StandardScaler()
    combined[features] = scaler_features.fit_transform(combined[features])

    # 标准化目标变量
    scaler_cnt = StandardScaler()
    combined['cnt'] = scaler_cnt.fit_transform(combined[['cnt']])

    # 分割为训练集、验证集和测试集
    train_size = len(train_df)
    train_val = combined.iloc[:train_size]
    test = combined.iloc[train_size:]

    # 将训练数据分成训练集和验证集
    train_idx = int(len(train_val) * 0.8)
    train = train_val.iloc[:train_idx]
    val = train_val.iloc[train_idx:]

    return train, val, test, scaler_features, scaler_cnt, features, target


# 自定义数据集
class BikeDataset(Dataset):
    def __init__(self, data, input_window, output_window, features, target):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.features = features
        self.target = target
        # 新增：用来存储输入窗口的 cnt
        self.X, self.y, self.input_cnt = self.create_sequences()

    def create_sequences(self):
        X = []
        y = []
        input_c = []
        total_window = self.input_window + self.output_window
        for i in range(len(self.data) - total_window + 1):
            seq_x = self.data.iloc[i:i + self.input_window][self.features].values
            seq_y = self.data.iloc[i + self.input_window:i + total_window][self.target].values
            seq_input_cnt = self.data.iloc[i:i + self.input_window][self.target].values
            X.append(seq_x)
            y.append(seq_y)
            input_c.append(seq_input_cnt)
        return np.array(X), np.array(y), np.array(input_c)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 返回真实的输入窗口 cnt
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx]), torch.FloatTensor(self.input_cnt[idx])


# 定义Transformer模型
class TransformerTimeSeries(nn.Module):
    def __init__(self, feature_size=12, num_layers=3, dropout=0.1, forward_expansion=2048, nhead=4):  # 修改nhead为4
        # 验证feature_size是否能被nhead整除
        assert feature_size % nhead == 0, f"feature_size ({feature_size}) must be divisible by nhead ({nhead})"
        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        # 设置 batch_first=True 以解决警告
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead,
                                                    dim_feedforward=forward_expansion, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_len, feature_size)
        src = self.pos_encoder(src)  # 添加位置编码
        output = self.transformer_encoder(src)  # (batch_size, seq_len, feature_size)
        output = self.decoder(output)  # (batch_size, seq_len, 1)
        return output.squeeze(-1)  # (batch_size, seq_len)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, save_path, scheduler=None,
                checkpoint_dir=None, exp_id=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_epoch = 0

    # 尝试加载检查点
    if checkpoint_dir and exp_id:
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_exp{exp_id}.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        try:
            # 训练阶段
            model.train()
            train_loss = 0
            for X, y, _ in train_loader:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X, y, _ in val_loader:
                    X = X.to(device)
                    y = y.to(device)
                    output = model(X)
                    loss = criterion(output, y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 简化学习率打印
            if epoch % 10 == 0:
                print(f'Learning rate: 0.001 (固定) - Epoch {epoch + 1}')

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 保存检查点
            if checkpoint_dir and exp_id:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, checkpoint_path)

        except KeyboardInterrupt:
            print("训练被中断，保存检查点...")
            if checkpoint_dir and exp_id:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, checkpoint_path)
            raise KeyboardInterrupt

    return train_losses, val_losses


def plot_losses(train_losses, val_losses, title):
    os.makedirs('plots/losses', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{title} 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(f'plots/losses/{title}_loss.png')
    plt.close()  # 不显示图像，只保存

    # 将损失数据写入 CSV 文件
    with open(f'plots/losses/{title}_loss_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'TrainLoss', 'ValLoss'])
        for i, (tr_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, tr_loss, val_loss])


# 评估函数
def evaluate_model(model, dataloader, scaler_cnt):
    model.eval()
    mse = 0
    mae = 0
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            # 将预测和真实值进行反标准化
            output = scaler_cnt.inverse_transform(output.cpu().numpy())
            y = scaler_cnt.inverse_transform(y.cpu().numpy())
            mse += mean_squared_error(y, output)
            mae += mean_absolute_error(y, output)
            all_preds.append(output)
            all_trues.append(y)
    mse /= len(dataloader)
    mae /= len(dataloader)
    return mse, mae, np.concatenate(all_preds, axis=0), np.concatenate(all_trues, axis=0)


# 绘制预测与真实值对比图
def plot_predictions(input_seq, true_seq, pred_seq, title, scaler_cnt):
    os.makedirs('plots/predictions', exist_ok=True)
    # 重新定义 X 轴坐标，使输入区间与输出区间相连
    L = len(input_seq)
    M = len(true_seq)
    time_input = np.arange(L + 1)
    time_output = np.arange(L, L + M)

    plt.figure(figsize=(15, 5))
    # 反标准化输入序列
    input_seq = scaler_cnt.inverse_transform(input_seq.reshape(-1, 1)).flatten()
    input_seq_extended = np.concatenate([input_seq, [true_seq[0]]])  # 将第一个真实值添加到输入序列中
    plt.plot(time_input, input_seq_extended, label='Input (前一个时间步 cnt)', color='gray')

    # 反标准化真实值、预测值在外部已完成，这里只负责绘图
    plt.plot(time_output, true_seq, label='True', color='blue')
    plt.plot(time_output, pred_seq, label='Predicted', color='red')

    plt.legend()
    plt.title(f'{title} 结果')
    plt.xlabel('时间步')
    plt.ylabel('租赁数量')
    plt.savefig(f'plots/predictions/{title}_prediction.png')
    plt.close()  # 不显示图像，只保存


# 封装实验流程
# 修改exp_configs，添加不同的随机种子
exp_configs = [
    {
        'description': f'固定学习率(0.001) - 随机种子 {seed}',
        'seed': seed
    } for seed in [42, 123, 456, 789, 1024]  # 使用不同的随机种子
]

def run_experiment(train_loader, val_loader, test_loader, scaler_cnt, features, target, prediction_type, input_window,
                   output_window, num_experiments=5, epochs=10, resume=False):  # 新增 resume 参数

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # 仅在 resume=True 时尝试加载实验进度
    progress_file = Path('checkpoints/progress.json')
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
            for f in Path('checkpoints').glob('checkpoint_exp*.pt'):
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
            
            # 使用配置中的随机种子
            set_seed(config['seed'])
            
            model = TransformerTimeSeries(feature_size=len(features), nhead=4).to(device)
            
            criterion = nn.MSELoss()
            # 使用固定学习率
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 定义模型保存路径，保存在 "saved_models" 文件夹
            save_path = os.path.join('saved_models', f'transformer_model_{prediction_type}_exp{exp + 1}.pth')

            # 更新进度文件
            progress['current_exp'] = exp
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

            # 使用训练和验证集进行训练
            train_losses, val_losses = train_model(
                model, train_loader, val_loader, criterion,
                optimizer, epochs, save_path, scheduler=None,  # 移除scheduler
                checkpoint_dir='checkpoints', exp_id=exp + 1
            )

            plot_losses(train_losses, val_losses, f"Experiment {exp + 1} - {prediction_type.capitalize()}")

            # 评估模型
            mse, mae, preds, trues = evaluate_model(model, test_loader, scaler_cnt)
            print(f"Experiment {exp + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}")
            mse_list.append(mse)
            mae_list.append(mae)

            # 加载模型（可选，如果有验证集，可以在这里加载最佳模型）
            model.load_state_dict(torch.load(save_path, weights_only=True))

            # 选择一部分测试数据进行预测并绘图（仅绘制第一个样本）
            model.eval()
            with torch.no_grad():
                X, y, input_cnt = next(iter(test_loader))
                X = X.to(device)
                y = y.to(device)
                predictions = model(X).cpu().numpy()
                true = y.cpu().numpy()
                # 将预测和真实值进行反标准化
                predictions = scaler_cnt.inverse_transform(predictions)
                true = scaler_cnt.inverse_transform(true)
                # 绘制时传入真实输入序列 input_cnt
                plot_predictions(input_cnt[1], true[1], predictions[1],
                                 f'Experiment {exp + 1} - {prediction_type.capitalize()} Predictions vs Ground Truth',
                                 scaler_cnt)

    except KeyboardInterrupt:
        print("\n训练被中断，已保存进度。下次运行时将从中断处继续。")
        sys.exit(0)

    # 计算平均值和标准差
    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)

    print(f"\nFinal Results for {prediction_type.capitalize()} Prediction:")
    print(f"MSE: Mean = {mse_mean:.4f}, Std = {mse_std:.4f}")
    print(f"MAE: Mean = {mae_mean:.4f}, Std = {mae_std:.4f}")


# 主函数
def main():
    parser = argparse.ArgumentParser(description='Bike Rental Forecasting using Transformer')
    parser.add_argument('--mode', type=str, choices=['short', 'long'], default='short',
                        help='Prediction mode: "short" for short-term (96 hours), "long" for long-term (240 hours)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--experiments', type=int, default=5, help='Number of experiments to run')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint if available')
    args = parser.parse_args()

    train_path = 'train_data.csv'
    test_path = 'test_data.csv'

    # 加载数据
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

    print(
        f"\nRunning {prediction_type.capitalize()} Prediction with input window = {input_window} and output window = {output_window}")

    # 创建训练集、验证集和测试集的数据集
    dataset_train = BikeDataset(train, input_window, output_window, features, target)
    dataset_val = BikeDataset(val, input_window, output_window, features, target)
    dataset_test = BikeDataset(test, input_window, output_window, features, target)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # 运行实验
    run_experiment(train_loader, val_loader, test_loader, scaler_cnt, features, target, prediction_type, input_window,
                   output_window, num_experiments=args.experiments, epochs=args.epochs, resume=args.resume)


if __name__ == "__main__":
    main()
