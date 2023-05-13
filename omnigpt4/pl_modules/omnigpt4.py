from dataclasses import dataclass
from typing import Optional, Tuple

import peft
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import AdamW

from omnigpt4.models.omnigpt4 import OmniGPT4Model
from omnigpt4.prompts import ChatPrompts
from omnigpt4.utils.optim import get_param_groups, WarmupCosineAnnealingLR


@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: float = 8.
    lora_dropout: float = 0.1
    fan_in_fan_out: bool = False
    bias: str = "none"
    inference_mode: bool = False


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
        visual_model_name_or_path: str,
        language_model_name_or_path: str,
        language_projection_weight_path: Optional[str] = None,
        sdpa_impl: str = "auto",
        compile_visual_model: bool = True,
        compile_qformer: bool = True,
        lora_config: Optional[LoraConfig] = None,
        freeze_visual_model: bool = True,
        freeze_qformer: bool = True,
        freeze_language_projection: bool = False,
        freeze_language_model: bool = True,
        cache_dir: Optional[str] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> None:
        super().__init__()

        if optimizer_config is None:
            optimizer_config = OptimizerConfig()

        self.save_hyperparameters()

        self.optimizer_config = optimizer_config

        self.model = OmniGPT4Model.from_vision_qformer_text_pretrained(
            visual_model_name_or_path=visual_model_name_or_path,
            language_model_name_or_path=language_model_name_or_path,
            language_projection_weight_path=language_projection_weight_path,
            sdpa_impl=sdpa_impl,
            compile_visual_model=compile_visual_model,
            compile_qformer=compile_qformer,
            cache_dir=cache_dir,
        )

        if freeze_visual_model:
            self.model.freeze_vision_model()

        if freeze_qformer:
            self.model.freeze_qformer()

        if freeze_language_projection:
            self.model.freeze_language_projection()

        if freeze_language_model:
            self.model.freeze_language_model()

        if lora_config is not None:
            lora_config = peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                fan_in_fan_out=lora_config.fan_in_fan_out,
                bias=lora_config.bias,
                inference_mode=lora_config.inference_mode,
            )

            self.model.language_model = peft.get_peft_model(
                self.model.language_model, peft_config=lora_config
            )
            self.model.language_model.print_trainable_parameters()

    def training_step(self, batch: ChatPrompts, batch_idx: int) -> STEP_OUTPUT:
        outputs = self.model(
            input_ids=batch.input_ids,
            pixel_values=batch.pixel_values,
            vision_token_indices=batch.vision_token_indices,
            attention_mask=batch.attention_mask,
            labels=batch.target_ids,
        )
        loss = outputs.loss

        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        vision_token_indices: Optional[torch.LongTensor] = None,  # TODO: find a better name
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        return self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
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
