from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    Blip2ForConditionalGeneration,
    AutoConfig,
    AutoModelForCausalLM,
)
from safetensors import safe_open
from safetensors.torch import save_file
from slugify import slugify

from omnigpt4.models.omnigpt4 import OmniGPT4Config, OmniGPT4Model
from omnigpt4.utils.attention import optimize_attention_ops
from omnigpt4.utils.optim import get_param_groups, WarmupCosineAnnealingLR


def _load_pretrained_model_state_dict(
    language_model_name_or_path: str,
    visual_model_name_or_path: str,
    language_projection_weight_path: Optional[str] = None,
    cache_root_path: Optional[str] = None,
    is_global_zero: bool = True,
) -> OmniGPT4Model:
    if cache_root_path is None:
        cache_root_path = Path(".cache")
    else:
        cache_root_path = Path(cache_root_path)
    if not cache_root_path.exists():
        cache_root_path.mkdir()

    cache_name = slugify(f"{language_model_name_or_path}_{visual_model_name_or_path}")
    cache_path = cache_root_path / (cache_name + ".safetensors")

    state_dict = {}

    if cache_path.exists():
        print(f"Loading cached state dict from {cache_path}")
        with safe_open(cache_path, framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        # TODO: support AutoCausalLM
        llm = AutoModelForCausalLM.from_pretrained(
            language_model_name_or_path,
            torch_dtype=torch.float16,
        )
        for key, val in llm.state_dict().items():
            state_dict["language_model." + key] = val
        print("LLM weights loaded.")

        blip2 = Blip2ForConditionalGeneration.from_pretrained(
            visual_model_name_or_path,
            torch_dtype=torch.float16,
        )
        for key, val in blip2.state_dict().items():
            if key.startswith("language_"):
                continue
            state_dict[key] = val
        print("BLIP2 weights loaded.")

        if is_global_zero:
            save_file(state_dict, cache_path)

    if language_projection_weight_path is not None:
        with safe_open(language_projection_weight_path, framework="pt") as f:
            for k in f.keys():
                assert k in ["language_projection.weight", "language_projection.bias"]
                state_dict[k] = f.get_tensor(k)

    return state_dict


def disabled_train(self, mode: bool = True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@dataclass
class OptimizerConfig:
    init_lr: float = 1e-4
    min_lr: float = 8e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.05
    norm_weight_decay: float = 0.0
    num_warmup_steps: int = 5000
    warmup_init_lr: float = 1e-6


class OmniGPT4(pl.LightningModule):
    def __init__(
        self,
        visual_model_name_or_path: str = "Salesforce/blip2-opt-6.7b",
        language_projection_weight_path: Optional[str] = None,
        language_model_name_or_path: str = "./weights/vicuna-7b-v1.1",
        attention_type: str = "original",
        freeze_visual_model: bool = True,
        freeze_qformer: bool = True,
        freeze_language_model: bool = True,
        cache_root_path: Optional[str] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> None:
        super().__init__()

        if optimizer_config is None:
            optimizer_config = OptimizerConfig()

        self.save_hyperparameters()

        self.freeze_visual_model = freeze_visual_model
        self.freeze_qformer = freeze_qformer
        self.freeze_language_model = freeze_language_model
        self.optimizer_config = optimizer_config

        config = OmniGPT4Config.from_vision_qformer_text_configs(
            vision_config=Blip2VisionConfig.from_pretrained(visual_model_name_or_path),
            qformer_config=Blip2QFormerConfig.from_pretrained(visual_model_name_or_path),
            text_config=AutoConfig.from_pretrained(language_model_name_or_path),
        )
        config.vision_config.layer_norm_eps = 1e-6  # following the original EVA-ViT config

        state_dict = _load_pretrained_model_state_dict(
            language_model_name_or_path=language_model_name_or_path,
            visual_model_name_or_path=visual_model_name_or_path,
            language_projection_weight_path=language_projection_weight_path,
            cache_root_path=cache_root_path,
            is_global_zero=self.global_rank == 0,
        )
        self.model = OmniGPT4Model.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=state_dict,
            torch_dtype=torch.float16,
        )
        # TODO: fix this. This will round the weights to fp16, which is not what we want.
        self.model.language_projection = self.model.language_projection.float()

        optimize_attention_ops(self.model, attention_type=attention_type)

        self.model.vision_model = torch.compile(self.model.vision_model)
        self.model.qformer = torch.compile(self.model.qformer)

        if freeze_visual_model:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            self.model.vision_model.eval()
            self.model.vision_model.train = disabled_train

        if freeze_qformer:
            self.model.query_tokens.requires_grad = False
            for param in self.model.qformer.parameters():
                param.requires_grad = False
            self.model.qformer.eval()
            self.model.qformer.train = disabled_train

        if freeze_language_model:
            for param in self.model.language_model.parameters():
                param.requires_grad = False
            self.model.language_model.eval()
            self.model.language_model.train = disabled_train

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        images, vision_token_positions = batch["images"], batch["vision_token_positions"]
        input_ids, attention_masks = batch["input_ids"], batch["attention_masks"]
        target_ids = batch["target_ids"]

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=images,
            vision_token_positions=vision_token_positions,
            attention_mask=attention_masks,
            labels=target_ids,
        )
        loss = outputs.loss

        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_token_positions: Optional[torch.LongTensor] = None,  # TODO: find a better name
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        return self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            vision_token_positions=vision_token_positions,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            params=get_param_groups(
                model=self,
                weight_decay=self.optimizer_config.weight_decay,
                norm_weight_decay=self.optimizer_config.norm_weight_decay,
            ),
            lr=self.optimizer_config.init_lr,
            betas=self.optimizer_config.betas,
        )

        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_config.num_warmup_steps,
            num_training_steps=self.trainer.max_steps,
            min_lr=self.optimizer_config.min_lr,
            warmup_init_lr=self.optimizer_config.warmup_init_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
