#!/bin/bash
#SBATCH -A m4999             # 请替换为你的 Allocation ID
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:02:00
#SBATCH -N 2                 # 2个节点 (Project 要求多机通信)
#SBATCH --ntasks-per-node=4  # Perlmutter 每个节点 4 个 A100
#SBATCH --gpus-per-node=4
#SBATCH -J llama_mscclpp
#SBATCH -o logs/%x-%j.out

# 1. 环境
source ~/.bashrc
cd pipeline-mscclpp
conda activate ccl-bench
mkdir -p logs

# 2. 定位库 (指向旧目录的构建结果)
OLD_PROJECT_ROOT=/pscratch/sd/x/xz987/CS5470/final_project
MSCCLPP_LIB=$(find $OLD_PROJECT_ROOT/mscclpp -name "libmscclpp_nccl.so" | head -n 1)
CONDA_CUDA_LIB=$(find $CONDA_PREFIX/lib -name "libcudart.so.12*" | head -n 1)

# 3. 注入配置
export MY_PRELOAD="${CONDA_CUDA_LIB}:${MSCCLPP_LIB}"
export MSCCLPP_XML_FILE=$(pwd)/allreduce.json
export MSCCLPP_DEBUG=INFO

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Starting Native DDP (Clean Start)..."

srun --label --export=ALL,LD_PRELOAD=$MY_PRELOAD \
    bash -c "
    export PYTHONPATH=\$PYTHONPATH:$(pwd)
    export LOCAL_RANK=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    
    # 运行新脚本
    python train_native.py
    "