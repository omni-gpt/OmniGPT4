import copy
import os
from typing import Union

from transformers import CONFIG_MAPPING, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging

logger = logging.get_logger(__name__)

# TODO: make it configurable
def get_config_cls(model_type: str) -> PretrainedConfig:
    if model_type == "blip_2_vision_model":
        from transformers import Blip2VisionConfig
        return Blip2VisionConfig
    elif model_type == "clip_vision_model":
        from transformers import CLIPVisionConfig
        return CLIPVisionConfig
    elif model_type == "blip_2_qformer":
        from transformers import Blip2QFormerConfig
        return Blip2QFormerConfig
    elif model_type == "openflamingo_perceiver_resampler":
        ...
    else:
        return CONFIG_MAPPING[model_type]


class OpenFlamingoPerceiverResamplerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OpenFlamingoPerceiverResamplerModel`].
    It is used to instantiate a OpenFlamingo Perceiver Resampler model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the OpenFlamingo [vivym/openflamingo-9b](https://huggingface.co/vivym/openflamingo-9b)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            The hidden size of the hidden states for cross-attention.
    """

    model_type = "openflamingo_perceiver_resampler"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        encoder_hidden_size=1408,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.encoder_hidden_size = encoder_hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from OpenFlamingoConfig
        if config_dict.get("model_type") == "openflamingo":
            config_dict = config_dict["resampler_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class OpenFlamingoConfig(PretrainedConfig):
    r"""
    [`OpenFlamingoConfig`] is the configuration class to store the configuration of a [`OpenFlamingoForConditionalGeneration`].
    It is used to instantiate a OpenFlamingo model according to the specified arguments, defining the vision model,
    Perceiver Resampler model and language model configs. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the OpenFlamingo [vivym/openflamingo-9b](https://huggingface.co/vivym/openflamingo-9b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OpenFlamingoVisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`OpenFlamingoQFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     OpenFlamingoVisionConfig,
    ...     OpenFlamingoQFormerConfig,
    ...     OPTConfig,
    ...     OpenFlamingoConfig,
    ...     OpenFlamingoForConditionalGeneration,
    ... )

    >>> # Initializing a OpenFlamingoConfig with vivym/openflamingo-9b style configuration
    >>> configuration = OpenFlamingoConfig()

    >>> # Initializing a OpenFlamingoForConditionalGeneration (with random weights) from the vivym/openflamingo-9b style configuration
    >>> model = OpenFlamingoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a OpenFlamingoConfig from a OpenFlamingoVisionConfig, OpenFlamingoQFormerConfig and any PretrainedConfig

    >>> # Initializing OpenFlamingo vision, OpenFlamingo Q-Former and language model configurations
    >>> vision_config = OpenFlamingoVisionConfig()
    >>> qformer_config = OpenFlamingoQFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = OpenFlamingoConfig.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```"""

    model_type = "openflamingo"
    is_composition = True

    def __init__(self, vision_config=None, resampler_config=None, text_config=None, **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the text config with default values (`CLIPVisionConfig`).")

        if resampler_config is None:
            resampler_config = {}
            logger.info("resampler_config is None. Initializing the OpenFlamingoQFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`LlaMaConfig`).")

        vision_model_type = vision_config["model_type"] if "model_type" in text_config else "clip_vision_model"
        self.vision_config = get_config_cls(vision_model_type)(**vision_config)
        self.resampler_config = OpenFlamingoPerceiverResamplerConfig(**resampler_config)
        text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        # self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_resampler_text_configs(
        cls,
        vision_config: PretrainedConfig,
        resampler_config: OpenFlamingoPerceiverResamplerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`OpenFlamingoConfig`] (or a derived class) from a OpenFlamingo vision model, resampler and language model
        configurations.

        Returns:
            [`OpenFlamingoConfig`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            resampler_config=resampler_config.to_dict(),
            text_config=text_config.to_dict(),
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
        output["resampler_config"] = self.resampler_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
