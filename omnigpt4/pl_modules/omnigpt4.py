import lightning.pytorch as pl
from pathlib import Path
from typing import Optional

from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    Blip2ForConditionalGeneration,
    LlamaConfig,
    LlamaForCausalLM,
)
from safetensors import safe_open
from safetensors.torch import save_file
from slugify import slugify

from omnigpt4.models.omnigpt4 import OmniGPT4Config, OmniGPT4Model


def _load_pretrained_model_state_dict(
    lm_model_name_or_path: str,
    visual_model_name_or_path: str,
    language_projection_weight_path: Optional[str] = None,
) -> OmniGPT4Model:
    cache_root = Path(".cache")
    if not cache_root.exists():
        cache_root.mkdir()

    cache_name = slugify(f"{lm_model_name_or_path}_{visual_model_name_or_path}")
    cache_path = cache_root / (cache_name + ".safetensors")

    state_dict = {}

    if cache_path.exists():
        print(f"Loading cached state dict from {cache_path}")
        with safe_open(cache_path, framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        vicuna = LlamaForCausalLM.from_pretrained(lm_model_name_or_path)
        for key, val in vicuna.state_dict().items():
            state_dict["language_model." + key] = val

        blip2 = Blip2ForConditionalGeneration.from_pretrained(visual_model_name_or_path)
        for key, val in blip2.state_dict().items():
            if key.startswith("language_"):
                continue
            state_dict[key] = val

        save_file(state_dict, cache_path)

    if language_projection_weight_path is not None:
        with safe_open(language_projection_weight_path, framework="pt") as f:
            for k in f.keys():
                assert k in ["language_projection.weight", "language_projection.bias"]
                state_dict[k] = f.get_tensor(k)

    return state_dict


class OmniGPT4(pl.LightningModule):
    def __init__(
        self,
        lm_model_name_or_path: str = "./weights/vicuna-7b-v1.1",
        visual_model_name_or_path: str = "Salesforce/blip2-opt-6.7b",
        language_projection_weight_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        config = OmniGPT4Config.from_vision_qformer_text_configs(
            vision_config=Blip2VisionConfig.from_pretrained(visual_model_name_or_path),
            qformer_config=Blip2QFormerConfig.from_pretrained(visual_model_name_or_path),
            text_config=LlamaConfig.from_pretrained(lm_model_name_or_path),
        )

        state_dict = _load_pretrained_model_state_dict(
            lm_model_name_or_path=lm_model_name_or_path,
            visual_model_name_or_path=visual_model_name_or_path,
            language_projection_weight_path=language_projection_weight_path,
        )
        self.model = OmniGPT4Model.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
        )
