import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from xformers.ops.fmha.attn_bias import AttentionBias, LowerTriangularMask

from transformers.models.blip_2.modeling_blip_2 import Blip2Attention
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel


def blip2_forward(
    self: Blip2Attention,
    hidden_states: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    mixed_qkv = self.qkv(hidden_states)

    if self._sdpa_impl == "xformers":
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 1, 3, 4
        )
    else:
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )

    query_states, key_states, value_states = (
        mixed_qkv[0],
        mixed_qkv[1],
        mixed_qkv[2],
    )

    if self._sdpa_impl == "original":
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states * self.scale, key_states.transpose(-1, -2))

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_states)
    elif self._sdpa_impl == "torch_sdpa":
        context_layer = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
        )
    elif self._sdpa_impl == "xformers":
        context_layer = xops.memory_efficient_attention(
            query_states, key_states, value_states, scale=self.scale
        )
    else:
        raise NotImplementedError(self._sdpa_impl)

    if self._sdpa_impl != "xformers":
        context_layer = context_layer.permute(0, 2, 1, 3)

    new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
    context_layer = context_layer.reshape(new_context_layer_shape)

    output = self.projection(context_layer)

    outputs = (output, attention_probs) if output_attentions else (output, None)

    return outputs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    with_xformers: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if with_xformers:
        # [1, 1, seq_len, head_dim] -> [1, seq_len, 1, head_dim]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        gather_indices = position_ids[:, :, None, None]  # [bs, seq_len, 1, 1]
        gather_indices = gather_indices.repeat(1, 1, cos.shape[2], cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 1, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 1, gather_indices)
    else:
        gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
        gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
        cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
        sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def llama_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)

    if self._sdpa_impl == "xformers":
        # [b, seq_len, nh, head_dim]
        seq_len_dim = 1
    else:
        # [b, nh, seq_len, head_dim]
        seq_len_dim = 2
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[seq_len_dim]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[seq_len_dim]

    # cos, sin: [1, 1, seq_len, head_dim]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids,
        with_xformers=self._sdpa_impl == "xformers",
    )

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=seq_len_dim)
        value_states = torch.cat([past_key_value[1], value_states], dim=seq_len_dim)

    past_key_value = (key_states, value_states) if use_cache else None

    if self._sdpa_impl == "original":
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
    elif self._sdpa_impl == "torch_sdpa":
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )
    elif self._sdpa_impl == "xformers":
        attn_output = xops.memory_efficient_attention(
            query_states, key_states, value_states, attn_bias=attention_mask,
        )
    else:
        raise NotImplementedError(self._sdpa_impl)

    if self._sdpa_impl != "xformers":
        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def llama_prepare_decoder_attention_mask(
    self: LlamaModel,
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, ...],
    inputs_embeds: Optional[torch.Tensor],
    past_key_values_length: int,
) -> AttentionBias:
    # create causal mask
    combined_attention_mask = LowerTriangularMask()

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = attention_mask.to(inputs_embeds.device)
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        expanded_attn_mask = expanded_attn_mask.repeat(1, self.config.num_attention_heads, 1, 1)
        expanded_attn_mask = expanded_attn_mask.contiguous()
        combined_attention_mask = combined_attention_mask.add_bias(expanded_attn_mask)

    return combined_attention_mask


def optimize_sdpa_ops(model: nn.Module, sdpa_impl: str = "auto") -> None:
    if sdpa_impl == "auto":
        # TODO: detect if xformers is installed
        sdpa_impl = "torch_sdpa"

    for module in model.modules():
        module._sdpa_impl = sdpa_impl
        if isinstance(module, Blip2Attention):
            module.forward = blip2_forward.__get__(module, Blip2Attention)
        elif isinstance(module, LlamaAttention):
            module.forward = llama_forward.__get__(module, LlamaAttention)
        elif isinstance(module, LlamaModel) and sdpa_impl == "xformers":
            module._prepare_decoder_attention_mask = llama_prepare_decoder_attention_mask.__get__(
                module, LlamaModel
            )
