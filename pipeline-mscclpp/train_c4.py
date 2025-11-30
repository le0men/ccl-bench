import os
import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.distributed.utils
from torch.utils.data import DataLoader

sys.path.append(os.getcwd()) 

# === 强制导入，如果出错直接崩，不要掩盖 ===
print(f"[DEBUG] Current Working Directory: {os.getcwd()}")
print(f"[DEBUG] Files in current dir: {os.listdir('.')}")

# 直接 import，不要 try-except
from data_c4 import C4IterableDataset 
print("[INFO] Successfully imported C4IterableDataset.")

# 1. 路径适配
sys.path.append(os.getcwd())

# 2. 导入 Llama 模型
try:
    from torchtitan.models.llama3 import Transformer, TransformerModelArgs
    print("[INFO] Successfully imported Llama model.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# ================== Monkey Patches (针对 MSCCL++) ==================
def patched_pre_forward(self, *inputs, **kwargs):
    if self.module.training:
        self.reducer.prepare_for_forward()
    # [HACK] 删除了 _rebuild_buckets 调用
    if self.parameters_to_ignore:
        self._check_and_sync_module_buffers()
    return inputs, kwargs

DDP._pre_forward = patched_pre_forward

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # [HACK] 禁用 Broadcast
    def no_op_broadcast(tensor, src, group=None, async_op=False):
        return None
    dist.broadcast = no_op_broadcast
    if hasattr(dist, "_broadcast_coalesced"):
        dist._broadcast_coalesced = lambda *args, **kwargs: None

    if rank == 0:
        print(f"=== Starting Training on {world_size} GPUs (C4 Dataset) ===")

    # Llama 3 配置 (为了跑得快，层数设为 2，但 hidden_dim 保持 4096 以保证通信压力)
    # max_seq_len 设为 512
    model_args = TransformerModelArgs(
        dim=4096,             
        n_layers=2,           
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        multiple_of=1024,
        max_seq_len=512,
    )

    with torch.device("meta"):
        model = Transformer(model_args)
    model = model.to_empty(device=local_rank)
    
    # [HACK] 禁用初始化检查
    try:
        dist._verify_params_across_processes = lambda *args, **kwargs: None
    except AttributeError: pass
    torch.distributed.utils._sync_params_and_buffers = lambda *args, **kwargs: None

    # DDP 包装
    model = DDP(
        model, 
        device_ids=[local_rank], 
        broadcast_buffers=False, 
        static_graph=True
    )
    
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # === 数据加载逻辑 ===
    if C4IterableDataset is not None:
        if rank == 0: print("Loading C4 Dataset stream...")
        dataset = C4IterableDataset(seq_len=model_args.max_seq_len)
        # num_workers=2 利用 CPU 预处理数据，防止 I/O 阻塞 GPU
        dataloader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    else:
        if rank == 0: print("[WARN] data_c4.py not found, using dummy data.")
        # Fallback dummy data
        dataloader = [
            (torch.randint(0, 128256, (1, 512)), torch.randint(0, 128256, (1, 512)))
            for _ in range(100)
        ]

    model.train()
    
    if rank == 0: print("Start Training Loop...")
    torch.cuda.synchronize()
    start_time = time.time()

    # 运行 50 步 Benchmark
    TARGET_STEPS = 50
    iterator = iter(dataloader)

    for i in range(TARGET_STEPS):
        try:
            input_ids, labels = next(iterator)
        except StopIteration:
            break
            
        input_ids = input_ids.to(local_rank)
        labels = labels.to(local_rank)

        optimizer.zero_grad()
        pred = model(input_ids)
        loss = torch.nn.functional.cross_entropy(pred.flatten(0, 1), labels.flatten())
        loss.backward()
        optimizer.step()
        
        if rank == 0 and i % 5 == 0:
            print(f"Step {i+1}/{TARGET_STEPS} | Loss: {loss.item():.4f}")

    torch.cuda.synchronize()
    end_time = time.time()
    
    if rank == 0:
        total_time = end_time - start_time
        avg_time = total_time / TARGET_STEPS
        print(f"=== RESULT ===")
        print(f"Total Steps: {TARGET_STEPS}")
        print(f"Total Time:  {total_time:.2f} s")
        print(f"Avg Step Time: {avg_time*1000:.2f} ms") # 重点关注这个指标

    dist.destroy_process_group()

if __name__ == "__main__":
    main()