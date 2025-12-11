import torch
from torch.utils.data import DataLoader, TensorDataset
from model import TransformerModel


def train(model, dataloader, optimizer, criterion, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            tokens, labels = batch
            tokens, labels = tokens.to(device), labels.to(device)
            # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            optimizer.zero_grad()
            # forward
            outputs = model(tokens)
            # 注意：nn.CrossEntropyLoss 期望输入为 (batch_size, num_classes)，而输出为 (batch_size, seq_len, num_classes)
            # 因此，我们需要将输出的最后一个维度进行 flatten
            # criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


# 简单字符级 tokenizer（含特殊符号）
class CharTokenizer:
    def __init__(self, corpus):
        # 特殊符号
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        specials = [self.pad_token, self.bos_token, self.eos_token]
        # 收集字符集
        charset = set()
        for text in corpus:
            charset.update(list(text))
        # 构建词表
        self.itos = specials + sorted(list(charset))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text: str):
        ids = [self.bos_id]
        ids.extend(self.stoi[ch] for ch in text)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i >= 0 and i < len(self.itos))


def build_dataloader(corpus, tokenizer: CharTokenizer, block_size=64, batch_size=8):
    # 将所有文本编码并拼接为一个连续令牌流
    stream = []
    for text in corpus:
        stream.extend(tokenizer.encode(text))
    # 构造不重叠的训练片段，每个片段长度为 block_size+1（用于右移标签）
    tokens_list, labels_list = [], []
    for i in range(0, len(stream) - (block_size + 1) + 1, block_size):
        chunk = stream[i : i + block_size + 1]
        tokens = chunk[:-1]  # 模型输入
        labels = chunk[1:]   # 预测目标（右移一位）
        tokens_list.append(tokens)
        labels_list.append(labels)
    if not tokens_list:
        raise ValueError("语料过短，无法构造训练样本；请增大语料或减小 block_size。")
    # 张量化
    tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    dataset = TensorDataset(tokens_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    # 准备一个小语料（示例）
    corpus = [
        "你好，世界！",
        "今天天气不错。",
        "大型语言模型很有趣。",
        "Transformers 支持长上下文的自回归生成。",
        "用 <eos> 作为段落边界，拼接后切分训练样本。",
    ]

    # 构建 tokenizer 与 dataloader
    tokenizer = CharTokenizer(corpus)
    batch_size = 8
    block_size = 64  # 每个训练样本的上下文长度（不含右移标签的最后一个 token）
    dataloader = build_dataloader(corpus, tokenizer, block_size=block_size, batch_size=batch_size)

    # 实例化模型（确保 head_dim=d_model//n_head 为偶数）
    d_model = 256
    n_head = 8
    d_hidden = 1024
    n_layer = 4
    max_seq_len = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(d_model, n_head, d_hidden, n_layer, vocab_size=tokenizer.vocab_size, max_seq_len=max_seq_len)
    model = model.to(device)

    # 优化器与损失（自回归，忽略索引可不设，因为无 pad）
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 演示训练过程（1 个 epoch）
    train(model, dataloader, optimizer, criterion, device, epochs=1)


if __name__ == '__main__':
    main()


