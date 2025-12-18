import argparse
import torch
import tiktoken
from model import TransformerModel
from data import tokenizing_distributed_data_loader, get_dist_info


def train_one_epoch(model, batch_iter, optimizer, criterion, device):
    """
    预训练单个 epoch：遍历一次（有限）数据迭代器。
    - batch_iter: 一个有限迭代器，yield (inputs, targets)，形状均为 [B, T]。
    """
    ddp, ddp_rank, _, _ = get_dist_info()
    model.train()
    last_loss = None
    for inputs, targets in batch_iter:
        # 确保与模型设备一致（若迭代器已在目标设备，此操作为 no-op）
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # nn.CrossEntropyLoss 期望输入为 (N, C)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    if (not ddp) or (ddp_rank == 0):
        print(f"Epoch done, Last Loss: {last_loss}")


def build_arg_parser():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description="TinyLLM Pretraining")
    # 模型参数
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_hidden", type=int, default=1024)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    # 训练参数
    p.add_argument("--batch_size", type=int, default=8, help="B")
    p.add_argument("--block_size", type=int, default=1024, help="T")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--device", type=str, default=default_device)
    # 数据参数
    p.add_argument("--data_dir", type=str, default="./base_data")
    p.add_argument("--encoding_name", type=str, default="cl100k_base")
    p.add_argument("--tokenizer_threads", type=int, default=4)
    p.add_argument("--tokenizer_batch_size", type=int, default=128)
    return p


def main():
    # 解析参数
    args = build_arg_parser().parse_args()

    # 分布式信息，用于只在 rank 0 打印日志
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # 根据配置和 DDP 自动绑定设备（需与 data.py 的参数保持一致）
    train_device_str = args.device
    if ddp and train_device_str.startswith("cuda"):
        train_device_str = f"cuda:{ddp_local_rank}"
    device = torch.device(train_device_str)

    # 词表大小取自 tiktoken 编码器
    enc = tiktoken.get_encoding(args.encoding_name)
    vocab_size = enc.n_vocab

    # 检查 max_seq_len 与 block_size 的兼容性
    if args.max_seq_len < args.block_size:
        raise ValueError(f"max_seq_len({args.max_seq_len}) 应 >= block_size({args.block_size})")

    # 实例化模型
    model = TransformerModel(
        d_model=args.d_model,
        n_head=args.n_head,
        d_hidden=args.d_hidden,
        n_layer=args.n_layer,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
    ).to(device)

    # 优化器与损失（自回归）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 预训练：每个 epoch 重新创建一次（有限）数据迭代器并完整遍历
    for epoch in range(args.epochs):
        batch_iter = tokenizing_distributed_data_loader(
            B=args.batch_size,
            T=args.block_size,
            tokenizer_threads=args.tokenizer_threads,
            tokenizer_batch_size=args.tokenizer_batch_size,
            device=train_device_str,
            data_dir=args.data_dir,
            encoding_name=args.encoding_name,
        )
        train_one_epoch(
            model,
            batch_iter,
            optimizer,
            criterion,
            device,
        )


if __name__ == '__main__':
    main()
