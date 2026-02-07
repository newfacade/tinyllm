import torch
from configuration_tinyllm import TinyLLMConfig
from modeling_tinyllm import TinyLLMForCausalLM

def test_kv_cache():
    # 1. 初始化一个小模型
    config = TinyLLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        use_cache=True
    )
    model = TinyLLMForCausalLM(config)
    model.eval()

    # 2. 模拟输入: "I love"
    input_ids = torch.tensor([[10, 20]]) 
    
    # -----------------------------------------
    # 情况 A: 不使用 Cache，一次性输入全量序列
    # -----------------------------------------
    with torch.no_grad():
        outputs_no_cache = model(input_ids, use_cache=False)
        logits_no_cache = outputs_no_cache.logits
        # 最后一个 token 的 logits (预测 "love" 的下一个词)
        next_token_logits_expected = logits_no_cache[:, -1, :]

    # -----------------------------------------
    # 情况 B: 使用 Cache，分步推理
    # -----------------------------------------
    with torch.no_grad():
        # 第一步：输入 "I"，计算并拿到 cache
        outputs_step1 = model(input_ids[:, :1], use_cache=True)
        past_key_values = outputs_step1.past_key_values
        
        # 第二步：输入 "love"，传入之前的 cache
        outputs_step2 = model(input_ids[:, 1:2], past_key_values=past_key_values, use_cache=True)
        next_token_logits_actual = outputs_step2.logits[:, -1, :]

    # 3. 对比结果
    diff = (next_token_logits_expected - next_token_logits_actual).abs().max()
    print(f"Logits 最大误差: {diff.item():.8f}")
    
    if diff < 1e-5:
        print("✅ KV Cache 验证通过！数值一致。")
    else:
        print("❌ KV Cache 验证失败！数值差异过大。")

if __name__ == "__main__":
    test_kv_cache()
