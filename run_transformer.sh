#!/bin/bash

# 确保脚本在出错时停止执行
set -e

# 创建必要的目录
mkdir -p saved_models plots/losses plots/predictions checkpoints

# 定义常用的训练参数
EPOCHS=100
EXPERIMENTS=5

# 函数：显示使用方法
show_usage() {
    echo "Usage: $0 [short|long] [--resume]"  # $0: 代表当前脚本的名称
    echo "Example:"
    echo "  $0 short        # 运行短期预测"
    echo "  $0 long         # 运行长期预测"
    echo "  $0 short --resume  # 从检查点恢复短期预测训练"
}

# 检查参数
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

MODE=$1
RESUME=""

# 检查是否包含 --resume 参数
if [ "$2" == "--resume" ]; then
    RESUME="--resume"
fi

# 根据模式运行相应的训练
case $MODE in
    "short")
        echo "运行短期预测 (96小时) ..."
        python transformer.py --mode short --epochs $EPOCHS --experiments $EXPERIMENTS $RESUME
        ;;
    "long")
        echo "运行长期预测 (240小时) ..."
        python transformer.py --mode long --epochs $EPOCHS --experiments $EXPERIMENTS $RESUME
        ;;
    *)
        echo "错误: 未知的模式 '$MODE'"
        show_usage
        exit 1
        ;;
esac
