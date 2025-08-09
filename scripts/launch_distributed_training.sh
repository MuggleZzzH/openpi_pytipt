#!/bin/bash

# PI0 RIPT 分布式训练启动脚本
# 第10阶段：完整的多GPU分布式训练支持

set -e

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 输出带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示使用说明
show_usage() {
    cat << EOF
PI0 RIPT 分布式训练启动脚本

使用方法:
    $0 [OPTIONS]

选项:
    -c, --config CONFIG_PATH    YAML配置文件路径
    -n, --nodes NUM_NODES       节点数量 (默认: 1)
    -g, --gpus NUM_GPUS         每节点GPU数量 (默认: 4)
    -r, --node-rank NODE_RANK   当前节点rank (默认: 0)
    -m, --master-addr ADDR      主节点地址 (默认: localhost)
    -p, --master-port PORT      主节点端口 (默认: 12355)
    --debug                     使用调试配置
    --dry-run                   只显示命令不执行
    -h, --help                  显示此帮助信息

示例:
    # 单机4GPU训练
    $0 --config pi0/ript/config/distributed_train_pi0.yaml

    # 单机8GPU训练
    $0 --config pi0/ript/config/distributed_train_pi0.yaml --gpus 8

    # 调试模式（2GPU）
    $0 --config pi0/ript/config/debug_distributed_train.yaml --gpus 2 --debug

    # 多机训练（节点0）
    $0 --config pi0/ript/config/distributed_train_pi0.yaml \\
       --nodes 2 --gpus 4 --node-rank 0 --master-addr 192.168.1.100

    # 多机训练（节点1）
    $0 --config pi0/ript/config/distributed_train_pi0.yaml \\
       --nodes 2 --gpus 4 --node-rank 1 --master-addr 192.168.1.100

环境变量:
    CUDA_VISIBLE_DEVICES       指定可见的GPU设备
    NCCL_DEBUG                  NCCL调试级别 (INFO, WARN, etc.)
    NCCL_TIMEOUT               NCCL通信超时时间（秒）
    OMP_NUM_THREADS            OpenMP线程数

EOF
}

# 默认参数
CONFIG_PATH=""
NUM_NODES=1
NUM_GPUS=4
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=12355
DEBUG_MODE=false
DRY_RUN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -n|--nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -r|--node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        -m|--master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        -p|--master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查配置文件
if [[ -z "$CONFIG_PATH" ]]; then
    if [[ "$DEBUG_MODE" == true ]]; then
        CONFIG_PATH="pi0/ript/config/debug_distributed_train.yaml"
        print_info "使用调试配置: $CONFIG_PATH"
    else
        print_error "必须指定配置文件路径 (-c/--config)"
        show_usage
        exit 1
    fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    print_error "配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 检查项目根目录
PROJECT_ROOT="/zhaohan/ZJH/openpi_pytorch"
if [[ ! -d "$PROJECT_ROOT" ]]; then
    print_error "项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查训练脚本
TRAIN_SCRIPT="10_train_with_distributed.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    print_error "训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# 检查CUDA设备
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    # 自动设置CUDA设备
    if [[ "$NUM_GPUS" -eq 1 ]]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [[ "$NUM_GPUS" -eq 2 ]]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [[ "$NUM_GPUS" -eq 4 ]]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [[ "$NUM_GPUS" -eq 8 ]]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    else
        print_warning "自动设置CUDA_VISIBLE_DEVICES失败，GPU数量: $NUM_GPUS"
    fi
fi

# 设置环境变量
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-108000}  # 30小时
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

# 调试模式下的额外设置
if [[ "$DEBUG_MODE" == true ]]; then
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    print_info "调试模式已启用"
fi

# 显示配置信息
print_info "分布式训练配置:"
echo "  项目根目录: $PROJECT_ROOT"
echo "  配置文件: $CONFIG_PATH"
echo "  节点数量: $NUM_NODES"
echo "  每节点GPU数: $NUM_GPUS"
echo "  当前节点rank: $NODE_RANK"
echo "  主节点地址: $MASTER_ADDR"
echo "  主节点端口: $MASTER_PORT"
echo "  CUDA设备: $CUDA_VISIBLE_DEVICES"
echo "  NCCL超时: $NCCL_TIMEOUT 秒"
echo "  OpenMP线程: $OMP_NUM_THREADS"
echo "  调试模式: $DEBUG_MODE"

# 验证GPU可用性
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_info "检测到 $GPU_COUNT 个GPU设备"
    
    if [[ "$NUM_GPUS" -gt "$GPU_COUNT" ]]; then
        print_warning "请求的GPU数量($NUM_GPUS)超过可用GPU数量($GPU_COUNT)"
    fi
else
    print_warning "无法检测NVIDIA GPU设备"
fi

# 构建torchrun命令
TORCHRUN_CMD="torchrun"

# 分布式参数
TORCHRUN_CMD="$TORCHRUN_CMD --nnodes=$NUM_NODES"
TORCHRUN_CMD="$TORCHRUN_CMD --nproc_per_node=$NUM_GPUS"
TORCHRUN_CMD="$TORCHRUN_CMD --node_rank=$NODE_RANK"
TORCHRUN_CMD="$TORCHRUN_CMD --master_addr=$MASTER_ADDR"
TORCHRUN_CMD="$TORCHRUN_CMD --master_port=$MASTER_PORT"

# 训练脚本和参数
TORCHRUN_CMD="$TORCHRUN_CMD $TRAIN_SCRIPT"
TORCHRUN_CMD="$TORCHRUN_CMD --config_path $CONFIG_PATH"

# 添加调试参数
if [[ "$DEBUG_MODE" == true ]]; then
    TORCHRUN_CMD="$TORCHRUN_CMD --debug_sampling"
fi

# 显示即将执行的命令
print_info "即将执行的命令:"
echo "$TORCHRUN_CMD"
echo

# 创建输出目录
OUTPUT_DIR="pi0/ript/output/distributed"
if [[ "$DEBUG_MODE" == true ]]; then
    OUTPUT_DIR="pi0/ript/output/distributed_debug"
fi
mkdir -p "$OUTPUT_DIR/logs"

# 日志文件
LOG_FILE="$OUTPUT_DIR/logs/distributed_training_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"

# Dry run模式
if [[ "$DRY_RUN" == true ]]; then
    print_info "Dry run模式，不执行实际训练"
    print_success "命令验证完成"
    exit 0
fi

# 执行训练
print_info "开始分布式训练..."
print_info "日志文件: $LOG_FILE"

# 创建trap来捕获中断信号
cleanup() {
    print_warning "收到中断信号，正在清理..."
    # 杀死所有子进程
    pkill -P $$
    exit 1
}
trap cleanup SIGINT SIGTERM

# 执行训练命令并同时输出到控制台和日志文件
if eval "$TORCHRUN_CMD 2>&1 | tee $LOG_FILE"; then
    print_success "分布式训练完成！"
    print_info "日志文件已保存: $LOG_FILE"
    
    # 显示结果摘要
    if [[ -f "$OUTPUT_DIR/distributed_training_results.json" ]]; then
        print_info "训练结果已保存: $OUTPUT_DIR/distributed_training_results.json"
    fi
    
    exit 0
else
    print_error "分布式训练失败！"
    print_info "检查日志文件: $LOG_FILE"
    exit 1
fi