import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer

class C4IterableDataset(IterableDataset):
    def __init__(self, seq_len, infinite=True):
        self.seq_len = seq_len
        self.infinite = infinite
        
        # 尝试加载 Llama Tokenizer，如果权限不足则回退到 GPT2
        # 注意：Benchmark 只需要 token 数量对即可，用哪个 tokenizer 不影响通信测速
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            # Llama 3 通常没有 pad token，设为 eos
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            print("[WARN] Using GPT2 tokenizer fallback (Llama 3 requires login).")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 加载 C4 数据集 (allenai/c4), 使用 'en' 子集
        # streaming=True 是关键，不下载到本地
        try:
            dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        except Exception as e:
            # 如果 C4 连不上，回退到 wikitext (更小，更稳)
            print(f"[WARN] C4 load failed ({e}), using wikitext.")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

        # 多进程分片：确保每个 GPU/Worker 读到不同的数据
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        iterator = iter(dataset)
        
        while True:
            try:
                item = next(iterator)
                text = item.get('text', '')
            except StopIteration:
                if self.infinite:
                    iterator = iter(dataset)
                    continue
                else:
                    break

            if not text: continue

            # Tokenize & Truncate
            # 关键：max_length 必须与模型 model_args 一致
            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len + 1, # +1 是为了错位做 label
                padding="max_length",
                return_tensors="pt"
            )
            
            ids = encodings.input_ids.squeeze(0)
            
            # 简单的 Next Token Prediction 任务构造
            x = ids[:-1]
            y = ids[1:]
            
            yield x, y