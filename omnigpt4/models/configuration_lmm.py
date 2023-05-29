import copy

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


class LMMConfig(PretrainedConfig):
    r"""
    [`LMMConfig`] is the configuration class to store the configuration of a [`LMMForConditionalGeneration`].
    It is used to instantiate a LMM model according to the specified arguments, defining the vision model, resampler model
    and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the OpenFlamingo [vivym/openflamingo-9b](https://huggingface.co/vivym/openflamingo-9b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        resampler_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
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

    model_type = "LMM"
    is_composition = True

    def __init__(self, vision_config, resampler_config, text_config, **kwargs):
        super().__init__(**kwargs)

        vision_model_type = vision_config["model_type"]
        self.vision_config = get_config_cls(vision_model_type)(**vision_config)

        resampler_model_type = resampler_config["model_type"]
        resampler_model_type
        self.resampler_config = get_config_cls(resampler_model_type)(**resampler_config)

        text_model_type = text_config["model_type"]
        self.text_config = get_config_cls(text_model_type)(**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        # self.resampler_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = text_model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_resampler_text_configs(
        cls,
        vision_config: PretrainedConfig,
        resampler_config: PretrainedConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`LMMConfig`] (or a derived class) from a LMM vision model, resampler and language model
        configurations.

        Returns:
            [`LMMConfig`]: An instance of a configuration object
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
