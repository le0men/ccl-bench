#!/bin/bash
#SBATCH -A m3502
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J benchmark_mscclpp
#SBATCH -o logs/mscclpp_%j.out

source ~/.bashrc
conda activate ccl-bench

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# === MSCCL++ SETUP ===
# 1. 库路径 (请确保这些路径在你机器上是正确的)
OLD_ROOT=/pscratch/sd/x/xz987/CS5470/final_project
MSCCLPP_LIB=$(find $OLD_ROOT/mscclpp -name "libmscclpp_nccl.so" | head -n 1)
CONDA_CUDA_LIB=$(find $CONDA_PREFIX/lib -name "libcudart.so.12*" | head -n 1)

export MY_PRELOAD="${CONDA_CUDA_LIB}:${MSCCLPP_LIB}"

# 2. 配置文件 (使用之前生成的 allreduce_ddp.json)
# 确保文件存在于当前目录
export MSCCLPP_XML_FILE=$(pwd)/allreduce.json
export MSCCLPP_DEBUG=INFO

echo "========================================"
echo "RUNNING EXPERIMENT B: MSCCL++"
echo "========================================"

srun --label --export=ALL,LD_PRELOAD=$MY_PRELOAD \
    bash -c "
    export PYTHONPATH=\$PYTHONPATH:$(pwd)
    export LOCAL_RANK=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    
    # 使用 nsys 抓取 Trace 证据 (可选)
    # nsys profile --trace=cuda,nvtx --output=mscclpp_c4_%p --duration=15 \
    python train_c4.py
    "