from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops
from transformers import (
    Blip2Config,
    Blip2VisionConfig,
    Blip2VisionModel as Blip2VisionModelBase,
)
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2VisionEmbeddings,
    Blip2Encoder as Blip2EncoderBase,
    Blip2EncoderLayer as Blip2EncoderLayerBase,
    Blip2Attention as Blip2AttentionBase,
    Blip2MLP,
    Blip2PreTrainedModel,
)


class Blip2Attention(Blip2AttentionBase):
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)

        if self.config.attention_type == "xformers":
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

        if self.config.attention_type == "vanilla":
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_states * self.scale, key_states.transpose(-1, -2))

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_states)
        elif self.config.attention_type == "torch_2.0":
            context_layer = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
            )
        elif self.config.attention_type == "xformers":
            context_layer = xformers.ops.memory_efficient_attention(
                query_states, key_states, value_states, scale=self.scale
            )
        else:
            raise NotImplementedError(self.config.attention_type)

        if self.config.attention_type != "xformers":
            context_layer = context_layer.permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


class Blip2EncoderLayer(Blip2EncoderLayerBase):
    def __init__(self, config: Blip2Config):
        nn.Module.__init__(self)
        self.embed_dim = config.hidden_size
        self.self_attn = Blip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Blip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)


class Blip2Encoder(Blip2EncoderBase):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Blip2EncoderLayer`].

    Args:
        config (`Blip2Config`):
            The corresponding vision configuration for the `Blip2Encoder`.
    """

    def __init__(self, config: Blip2Config):
        nn.Module.__init__(self)
        self.config = config
        self.layers = nn.ModuleList([Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


class Blip2VisionModel(Blip2VisionModelBase):
    def __init__(self, config: Blip2VisionConfig):
        Blip2PreTrainedModel.__init__(self, config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = Blip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()
