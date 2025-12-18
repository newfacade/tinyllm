import os
from collections import deque
from typing import Generator, List

import torch
import tiktoken
import pyarrow.parquet as pq


def list_parquet_files(data_dir="./base_data"):
    """列出目录下所有 .parquet 文件，按文件名排序"""
    paths = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith('.parquet'):
            paths.append(os.path.join(data_dir, file))
    return paths


def is_ddp() -> bool:
    """
    判断是否为分布式训练（DDP）。依据环境变量 WORLD_SIZE 是否大于 1。
    """
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False

def get_dist_info():
    """
    - 若 is_ddp() 为真，则从环境变量读取并返回分布式信息：
      ddp_rank：全局进程号（[0..WORLD_SIZE-1]），用于区分每个 worker。
      ddp_local_rank：本机内进程号（通常对应 GPU 序号），用于选择设备等。
      ddp_world_size：总进程数（全局 worker 数量）。
    - 否则返回单机模式 (False, 0, 0, 1)。
    """
    if is_ddp():
        assert all(
            var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
        ), "DDP 环境变量缺失：需要 RANK, LOCAL_RANK, WORLD_SIZE"
        ddp_rank = int(os.environ["RANK"])             # 全局进程号
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本机进程号/设备序号
        ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
        # 比如说4机，每个机器8卡训练，ddp_world_size 为 32
        # 第2个机器的第4个GPU: node_rank=1, local_rank=3, ddp_rank=1*8+3=11
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def tokenizing_distributed_data_loader(
    B: int,
    T: int,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    data_dir: str = "./base_data",
    encoding_name: str = "cl100k_base",
) -> Generator:
    """
    简化版分布式数据加载器（去掉 split，使用全部文件）：
    - 使用 tiktoken 进行分词，无状态恢复逻辑。
    - 支持 DDP：不同 rank 以 row_group 维度做步进分片。

    参数解释：
    - B：batch size，每个训练步的样本条数（张量的批维度）。
    - T：每条样本的序列长度（token 数，张量的时间/序列维度）。
    - tokenizer_threads：tiktoken 并行分词线程数（encode_batch 的并行度）。
    - tokenizer_batch_size：将一批文档再切分为子批进行分词的大小，控制内存与并发。
    - device：目标设备，例如 "cuda"、"cuda:0" 或 "cpu"。
    - data_dir：包含 .parquet 文件的目录路径。
    - encoding_name：tiktoken 编码器名称，例如 "cl100k_base"。

    产出：(inputs, targets) 二元组，形状均为 [B, T]；遍历完数据后结束。
    inputs 为输入序列，targets 为右移一位的预测目标。
    """

    # 使用全部 .parquet 文件
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        raise FileNotFoundError(f"在目录 {data_dir} 下未找到 .parquet 文件")

    # ddp: 是否分布式；ddp_rank: 全局进程号；ddp_local_rank: 本机进程号；ddp_world_size: 总进程数
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # tiktoken 编码器
    encoding = tiktoken.get_encoding(encoding_name)
    # 在每个文档末尾插入 EOS（end-of-text）以避免跨文档拼接
    eos_id = encoding.encode("<|endoftext|>", allowed_special="all")[0]

    def _yield_row_group_texts() -> Generator[List[str], None, None]:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            # 起始于 rank，步长为 world_size
            rg_idx = ddp_rank if ddp else 0
            while rg_idx < pf.num_row_groups:
                # 读取第 rg_idx 个 row group，返回一个 Arrow Table
                rg = pf.read_row_group(rg_idx)
                # 假设文本列名为 'text'
                docs = rg.column('text').to_pylist()
                # 分成更小批次便于并行分词
                for i in range(0, len(docs), tokenizer_batch_size):
                    yield docs[i:i + tokenizer_batch_size]
                rg_idx += ddp_world_size if ddp else 1

    # 需要的 token 数（+1 用于构造 targets）
    needed_tokens = B * T + 1
    token_buffer: deque[int] = deque[int]()

    use_cuda_optimizations = device.startswith("cuda")

    # 遍历一次数据集，消耗完毕后结束生成
    for doc_batch in _yield_row_group_texts():
        # doc_batch: List[str] 长度通常为 tokenizer_batch_size（最后一批可能不足）
        token_lists = encoding.encode_batch(doc_batch, num_threads=tokenizer_threads)
        for tokens in token_lists:
            # 为每个文档追加一个 EOS，形成明确的边界
            tokens.append(eos_id)
            token_buffer.extend(tokens)

        # 只要缓冲区足够，就持续产出训练步
        while len(token_buffer) >= needed_tokens:
            step_tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
            scratch = torch.tensor(step_tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
            inputs_cpu = scratch[:-1]
            targets_cpu = scratch[1:]
            # non_blocking=True 在满足“源张量是 CPU 端的 pinned memory”时，会让 CPU→GPU 的拷贝变成异步拷贝（不阻塞当前主机线程/流）
            # 可以与 GPU 计算重叠，提高吞吐。
            inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
            targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
            yield inputs, targets
    # 数据遍历结束；若剩余 tokens 不足以组成一个完整训练步，直接结束生成器。
