#!/bin/bash
#SBATCH -A m3502
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J benchmark_baseline
#SBATCH -o logs/baseline_%j.out

source ~/.bashrc
conda activate ccl-bench

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# === BASELINE SETUP (No MSCCL++) ===
unset LD_PRELOAD
unset MSCCLPP_XML_FILE

echo "========================================"
echo "RUNNING EXPERIMENT A: NCCL BASELINE"
echo "========================================"

srun --label \
    bash -c "
    export PYTHONPATH=\$PYTHONPATH:$(pwd)
    export LOCAL_RANK=\$SLURM_LOCALID
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    
    # 运行 C4 训练脚本
    python train_c4.py
    "