import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from configuration_tinyllm import TinyLLMConfig
from normalization import RMSNorm
from positional_embedding import apply_rope_emb, compute_freqs_cis

class TinyLLMAttention(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freq_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        q = self.wq(hidden_states)
        k = self.wk(hidden_states)
        v = self.wv(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_heads, self.head_dim)

        # RoPE application (apply before transpose to match shape (b, t, n, h))
        q, k = apply_rope_emb(q, k, freq_cis)

        # Transpose for attention calculation -> [b, n, t, h]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV Cache logic
        if past_key_value is not None:
            # past_key_value contains (past_k, past_v) from previous steps
            # reuse k, v, self_attention
            past_key_value = [x.to(k.device) for x in past_key_value] # ensure device match
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)

        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Attention calculation
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # attention_mask shape needs to be broadcastable to scores: [b, n, q_len, k_len]
            # Usually mask is [1, 1, q_len, k_len]
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = attn_weights @ v
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present_key_value

class TinyLLMMLP(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class TinyLLMDecoderLayer(nn.Module):
    def __init__(self, config: TinyLLMConfig):
        super().__init__()
        self.self_attn = TinyLLMAttention(config)
        self.mlp = TinyLLMMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freq_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, present_key_value = self.self_attn(
            hidden_states, 
            freq_cis, 
            attention_mask, 
            past_key_value, 
            use_cache
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

class TinyLLMPreTrainedModel(PreTrainedModel):
    config_class = TinyLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TinyLLMDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.hidden_size ** -0.5
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TinyLLMModel(TinyLLMPreTrainedModel):
    def __init__(self, config: TinyLLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TinyLLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # RoPE setup
        head_dim = config.hidden_size // config.num_attention_heads
        self.freq_cis = compute_freqs_cis(config.max_position_embeddings, head_dim, config.rope_theta)
        
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        inputs_embeds = self.embed_dropout(inputs_embeds)
        hidden_states = inputs_embeds

        # KV Cache handling
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # Prepare causal mask if attention_mask is not provided
        if attention_mask is None:
            mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            attention_mask = mask.view(1, 1, seq_length, seq_length)
            
            # If we have past_key_values, we need to adjust the mask
            # For decoding (seq_len=1), we attend to all past keys + current key, so mask is usually all zeros (visible)
            # However, if passed attention_mask is 2D [batch, seq], we might need expansion
            # Here we assume standard causal masking. 
            # If past is present, we are generating token N+1.
            # The query is token N+1. Keys are 0...N+1.
            # Query N+1 can see 0...N+1. So no masking needed for the past part.
            # The standard causal mask generated above (seq_len x seq_len) covers the current step(s).
            # If seq_len=1, mask is [[0]]. Correct.
            pass
        else:
             # Expand mask for [batch, 1, q_len, k_len]
             if attention_mask.dim() == 2:
                 # [batch, seq_len] -> [batch, 1, 1, seq_len]
                 # Note: Hugging Face usually passes 4D mask in prepare_inputs_for_generation if not causal
                 # But for simple causal generation, we often just get 2D mask.
                 # Let's trust standard HF flow or provided mask.
                 # For simplicity in this tutorial model, we assume caller handles complex masking or we rely on implicit broadcasting.
                 pass

        # Prepare RoPE frequencies
        seq_length_with_past = seq_length + past_key_values_length
        if seq_length_with_past > self.freq_cis.shape[0]:
             # Dynamic resizing could happen here, but for now we assume max_position_embeddings is enough
             pass
        freq_cis = self.freq_cis[past_key_values_length:seq_length_with_past].to(device)

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states, 
                freq_cis, 
                attention_mask, 
                past_key_value, 
                use_cache
            )
            
            if use_cache:
                next_decoder_cache += (present_key_value,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states] if v is not None)

        return CausalLMOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
        )

class TinyLLMForCausalLM(TinyLLMPreTrainedModel):
    def __init__(self, config: TinyLLMConfig):
        super().__init__(config)
        self.model = TinyLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not self.config.use_return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
