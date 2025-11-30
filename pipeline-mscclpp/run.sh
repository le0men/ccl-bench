export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. 重新设置 MSCCL++ 魔法变量
# (复制之前的逻辑)
OLD_ROOT=/pscratch/sd/x/xz987/CS5470/final_project
MSCCLPP_LIB=$(find $OLD_ROOT/mscclpp -name "libmscclpp_nccl.so" | head -n 1)
CONDA_CUDA_LIB=$(find $CONDA_PREFIX/lib -name "libcudart.so.12*" | head -n 1)

export LD_PRELOAD="${CONDA_CUDA_LIB}:${MSCCLPP_LIB}"
export MSCCLPP_XML_FILE=$(pwd)/allreduce.json
export MSCCLPP_DEBUG=INFO
torchrun --nproc_per_node=4 --standalone train_c4.py