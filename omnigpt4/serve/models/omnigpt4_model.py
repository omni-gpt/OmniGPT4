from typing import List, Tuple, Optional

import torch
from ray import serve
from transformers import AutoTokenizer

from omnigpt4.pl_modules.omnigpt4 import OmniGPT4
from omnigpt4.prompts import ChatPrompts


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
class OmniGPT4Deployment:
    def __init__(self) -> None:
        self.device = torch.device("cuda")

        self.model = OmniGPT4(
            visual_model_name_or_path="Salesforce/blip2-flan-t5-xxl",
            language_model_name_or_path="bigscience/bloomz-7b1",
            language_projection_weight_path="./weights/blip2_bloomz_7b1_stage_2_cpp20tmc_00004000.safetensors",
            cache_dir=".cache",
        )
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bigscience/bloomz-7b1",
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompts: ChatPrompts,
        max_length: int = 20,
        max_new_tokens: Optional[int] = None,
        max_time: Optional[float] = None,
        do_sample: bool = False,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        penalty_alpha: Optional[float] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        typical_p: float = 1.0,
        epsilon_cutoff: float = 0.0,
        eta_cutoff: float = 0.0,
        diversity_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        encoder_repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_return_sequences: int = 1,
    ) -> Tuple[List[str], List[int]]:
        input_ids = torch.from_numpy(prompts.input_ids.copy())
        input_ids = input_ids.to(self.device, non_blocking=True)
        if prompts.pixel_values is not None:
            pixel_values = torch.from_numpy(prompts.pixel_values.copy())
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            vision_token_indices = torch.from_numpy(prompts.vision_token_indices.copy())
            vision_token_indices = vision_token_indices.to(
                self.device, non_blocking=True
            )
        else:
            pixel_values = None
            vision_token_indices = None
        attention_mask = torch.from_numpy(prompts.attention_mask.copy())
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            max_time=max_time,
            do_sample=do_sample,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            penalty_alpha=penalty_alpha,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            epsilon_cutoff=epsilon_cutoff,
            eta_cutoff=eta_cutoff,
            diversity_penalty=diversity_penalty,
            repetition_penalty=repetition_penalty,
            encoder_repetition_penalty=encoder_repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
        )

        num_tokens = [output.shape[0] for output in outputs]

        return self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ), num_tokens
