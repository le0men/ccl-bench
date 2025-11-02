#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Only log from local‐rank 0 on SLURM nodes 0 and 2 (i.e. global 0 and 8)
case "$SLURM_NODEID" in
  0|2) LOG_RANK=0   ;;
  *)   LOG_RANK=-1  ;;
esac
export LOG_RANK

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b_tp_4_dp_2_pp_2_cp_1_astra.toml"}
export HF_HOME=/pscratch/sd/c/co232/hf_cache
export TORCHTITAN_LOGDIR=/pscratch/sd/c/co232/my_tb_logs_8b_tp_4_dp_2_pp_2_cp_1_astra

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \

# Multinode Configs
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$( srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address )

# Custom NCCL path for NCCL 2.27 build
export NCCL_HOME=/global/u2/c/co232/ReCCL-workspace/nccl/build
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'nccl' | paste -sd: -)"
export LD_PRELOAD=$NCCL_HOME/lib/libnccl.so.2.27.6
export LIBRARY_PATH="$NCCL_HOME/lib:$LIBRARY_PATH"
export CPATH="$NCCL_HOME/include:$CPATH"

# Memory & NCCL Configs
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,ENV,COLL
export NCCL_DEBUG_SUBSYS=INIT

export NCCL_BUFFSIZE=1048576
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export NCCL_NVTX_LEVEL=2

num_nodes=4

SLURM_ARGS="--nodes=${num_nodes} \
            --ntasks=${num_nodes} \
            --ntasks-per-node=1 \
            --gpus-per-task=4 \
            --cpus-per-task=16 \
            --partition=gpu \
            --time=04:00:00"

TORCHRUN_CMD="torchrun \
  --nnodes=${num_nodes} \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_id=tt_multi_${SLURM_JOB_ID} \
  --rdzv_endpoint=${head_node_ip}:29500 \
  --role rank \
  --local-ranks-filter ${LOG_RANK} \
  --tee 3 \
  -m torchtitan.train \
    --job.config_file ${CONFIG_FILE} \
    ${overrides}"

srun $SLURM_ARGS $TORCHRUN_CMD