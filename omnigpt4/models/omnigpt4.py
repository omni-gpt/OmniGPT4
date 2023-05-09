import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    Blip2VisionConfig,
    Blip2VisionModel,
    Blip2ForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
    CONFIG_MAPPING,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithPast,
    Seq2SeqLMOutput,
)
from transformers.models.blip_2.modeling_blip_2 import Blip2Encoder, Blip2VisionEmbeddings
from transformers.modeling_utils import no_init_weights
from transformers.utils import logging
from safetensors.torch import load_model, save_model
from slugify import slugify

from omnigpt4.utils.freeze import freeze
from omnigpt4.utils.init import no_init
from omnigpt4.utils.sdpa_hooks import optimize_sdpa_ops

logger = logging.get_logger(__name__)


class OmniGPT4Config(PretrainedConfig):
    model_type = "omnigpt4"
    is_composition = True

    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Blip2VisionConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`LLaMaConfig`).")

        if isinstance(vision_config, dict):
            vision_config = Blip2VisionConfig(**vision_config)

        if isinstance(qformer_config, dict):
            qformer_config = Blip2QFormerConfig(**qformer_config)

        if isinstance(text_config, dict):
            text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.vision_config = vision_config
        self.qformer_config = qformer_config
        self.text_config = text_config

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: Blip2VisionConfig,
        qformer_config: Blip2QFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config,
            qformer_config=qformer_config,
            text_config=text_config,
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class OmniGPT4PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OmniGPT4Config
    base_model_prefix = "omnigpt4"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
        r"language_model.lm_head.weight",
    ]
    _no_split_modules = ["Blip2Attention"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, Blip2VisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Blip2Encoder):
            module.gradient_checkpointing = value


@dataclass
class OmniGPT4ModelOutput:
    """
    Class defining the outputs of [`OmniGPT4`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[BaseModelOutputWithPooling] = None
    qformer_outputs: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    language_model_outputs: Optional[Union[CausalLMOutputWithPast, Seq2SeqLMOutput]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class OmniGPT4Model(OmniGPT4PreTrainedModel):
    def __init__(self, config: OmniGPT4Config):
        super().__init__(config)

        with no_init():
            self.vision_model = Blip2VisionModel(config.vision_config)

        with no_init():
            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
            )
            self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        with no_init():
            trust_remote_code = hasattr(config.text_config, "auto_map")
            self.language_model: PreTrainedModel = AutoModelForCausalLM.from_config(
                config.text_config,
                trust_remote_code=trust_remote_code,
            )
            self.language_model.tie_weights()

    @classmethod
    def from_vision_qformer_text_pretrained(
        cls,
        visual_model_name_or_path: str,
        language_model_name_or_path: str,
        language_projection_weight_path: Optional[str] = None,
        sdpa_impl: str = "auto",
        compile_visual_model: bool = True,
        compile_qformer: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "OmniGPT4Model":
        trust_remote_code = language_model_name_or_path in ["THUDM/chatglm-6b"]
        revision = "main" if trust_remote_code else None

        config = OmniGPT4Config.from_vision_qformer_text_configs(
            vision_config=Blip2VisionConfig.from_pretrained(visual_model_name_or_path),
            qformer_config=Blip2QFormerConfig.from_pretrained(visual_model_name_or_path),
            text_config=AutoConfig.from_pretrained(
                language_model_name_or_path,
                trust_remote_code=language_model_name_or_path in ["THUDM/chatglm-6b"],
                revision=revision,
            ),
        )
        config.vision_config.layer_norm_eps = 1e-6  # following the original EVA-ViT config

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "omnigpt4"
        else:
            cache_dir = Path(cache_dir)

        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        cache_name = slugify(f"{language_model_name_or_path}_{visual_model_name_or_path}")
        cache_path = cache_dir / (cache_name + ".safetensors")

        with no_init_weights():
            model = cls(config)

        if cache_path.exists():
            logger.info("Loading cached weights...")
            load_model(model, str(cache_path))
        else:
            logger.info("Loading BLIP2...")
            with no_init():
                blip2 = Blip2ForConditionalGeneration.from_pretrained(
                    visual_model_name_or_path,
                    torch_dtype=torch.float16,
                )
            model.vision_model = blip2.vision_model
            model.query_tokens = blip2.query_tokens
            model.qformer = blip2.qformer
            del blip2

            logger.info("Loading LLM...")
            trust_remote_code = language_model_name_or_path in ["THUDM/chatglm-6b"]
            revision = "main" if trust_remote_code else None

            with no_init():
                llm = AutoModelForCausalLM.from_pretrained(
                    language_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    torch_dtype=torch.float16,
                )
            model.language_model = llm
            logger.info("Done.")

            save_model(model, str(cache_path))

        if language_projection_weight_path is not None:
            logger.info("Loading language projection...")
            load_model(model.language_projection, language_projection_weight_path)
            logger.info("Done.")
        else:
            model.language_projection.reset_parameters()

        optimize_sdpa_ops(model, sdpa_impl=sdpa_impl)

        if compile_visual_model:
            model.vision_model = torch.compile(model.vision_model)

        if compile_qformer:
            model.qformer = torch.compile(model.qformer)

        return model

    def freeze_vision_model(self):
        freeze(self.vision_model)

    def freeze_qformer(self):
        self.query_tokens.requires_grad = False
        freeze(self.qformer)

    def freeze_language_projection(self):
        freeze(self.language_projection)

    def freeze_language_model(self):
        freeze(self.language_model)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + OmniGPT4 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for",
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_token_indices: Optional[torch.LongTensor] = None,  # TODO: find a better name
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        image_embeds_for_lm = self.language_projection(query_output)

        # with torch.set_grad_enabled(not freeze_language_model):
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(image_embeds_for_lm.dtype)

        # replace the [unk] token with the image embeddings
        inputs_embeds_shape = inputs_embeds.shape
        inputs_embeds.view(-1, *inputs_embeds_shape[2:])[
            vision_token_indices.view(-1)
        ] = image_embeds_for_lm.flatten(0, 1)
        inputs_embeds = inputs_embeds.view(*inputs_embeds_shape)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(image_embeds_for_lm.dtype)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return OmniGPT4ModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_token_positions: Optional[torch.LongTensor] = None,  # TODO: find a better name
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        image_embeds_for_lm = self.language_projection(query_output)

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(image_embeds_for_lm.dtype)

        # replace the [IMG] token with the image embeddings
        vision_token_idxs = vision_token_positions.view(-1)
        inputs_embeds_shape = inputs_embeds.shape
        inputs_embeds.view(-1, *inputs_embeds_shape[2:])[
            vision_token_idxs
        ] = image_embeds_for_lm.flatten(0, 1)
        inputs_embeds = inputs_embeds.view(*inputs_embeds_shape)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(image_embeds_for_lm.dtype)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
