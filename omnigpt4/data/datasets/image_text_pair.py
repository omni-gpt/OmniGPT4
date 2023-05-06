import re
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
import webdataset as wds
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer
from webdataset.shardlists import expand_urls


def text_processor(caption: str) -> str:
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    return caption


def build_image_text_pair_pipeline(
    urls: List[str],
    batch_size: int = 64,
    image_size: int = 224,
    min_scale: int = 0.5,
    max_scale: int = 1.0,
    max_tokens: int = 64,
    num_tokens_per_image: int = 32,
    tokenizer_name_or_path: str = "bert-base-uncased",
    end_sym: str = "\n",
    prompt_template: str = "",
    prompts_path: Optional[str] = None,
) -> wds.DataPipeline:
    vis_processor = T.Compose([
        T.RandomResizedCrop(
            size=image_size,
            scale=(min_scale, max_scale),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    unk_token_id = tokenizer.unk_token_id

    if prompts_path is not None:
        with open(prompts_path, "r") as f:
            prompts = f.read().splitlines()

    def tokenize(sample):
        # Random pick a prompt if prompts are provided
        if prompts_path is not None:
            prompt = prompt_template.format(np.random.choice(prompts))
        else:
            prompt = "<Img><ImageEmbeds></Img>"

        response = text_processor(sample[1]["caption"])

        input_ids = [bos_token_id]
        attention_mask = [1]
        vision_token_positions = []
        target_ids = [-100]

        end_tokens: BatchEncoding = tokenizer(end_sym, add_special_tokens=False)
        num_end_tokens = len(end_tokens.input_ids)

        num_remaining_tokens = max_tokens - 1 - num_end_tokens

        text_pieces = prompt.split("<ImageEmbeds>")
        for i, text_piece in enumerate(text_pieces):
            if num_remaining_tokens <= 0:
                break

            if i > 0 and num_remaining_tokens >= num_end_tokens:
                start_pos = len(input_ids)
                vision_token_positions.append(
                    list(range(start_pos, start_pos + num_tokens_per_image))
                )

                # unk_token -> image token
                input_ids += [unk_token_id] * num_tokens_per_image
                attention_mask += [1] * num_tokens_per_image
                target_ids += [-100] * num_tokens_per_image
                num_remaining_tokens -= num_tokens_per_image

            if num_remaining_tokens > 0:
                tokens: BatchEncoding = tokenizer(
                    text=text_piece,
                    padding=False,
                    truncation=True,
                    max_length=num_remaining_tokens,
                    add_special_tokens=False,
                )
                input_ids += tokens.input_ids
                attention_mask += tokens.attention_mask
                target_ids += [-100] * len(tokens.input_ids)
                num_remaining_tokens -= len(tokens.input_ids)

        if num_remaining_tokens > 0:
            tokens: BatchEncoding = tokenizer(
                text=response,
                padding=False,
                truncation=True,
                max_length=num_remaining_tokens,
                add_special_tokens=False,
            )
            input_ids += tokens.input_ids
            attention_mask += tokens.attention_mask
            target_ids += tokens.input_ids
            num_remaining_tokens -= len(tokens.input_ids)

        input_ids += end_tokens.input_ids
        attention_mask += end_tokens.attention_mask
        target_ids += end_tokens.input_ids

        # to tensor
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)

        if len(vision_token_positions) > 0:
            vision_token_positions = torch.as_tensor(vision_token_positions, dtype=torch.long)
        else:
            vision_token_positions = torch.zeros(0, num_tokens_per_image, dtype=torch.long)

        target_ids = torch.as_tensor(target_ids, dtype=torch.long)

        return {
            "images": sample[0][None, ...],
            "vision_token_positions": vision_token_positions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

    def collate(batch):
        batch_size = len(batch)
        images = torch.cat([data["images"] for data in batch], dim=0)

        max_input_tokens = max(data["input_ids"].shape[0] for data in batch)
        input_ids = torch.full(
            (batch_size, max_input_tokens), fill_value=eos_token_id, dtype=torch.long
        )
        attention_masks = torch.zeros(batch_size, max_input_tokens, dtype=torch.long)
        for i, data in enumerate(batch):
            input_ids[i, :data["input_ids"].shape[0]] = data["input_ids"]
            attention_masks[i, :data["input_ids"].shape[0]] = data["attention_mask"]

        max_target_ids = max(data["target_ids"].shape[0] for data in batch)
        target_ids = torch.full(
            (batch_size, max_target_ids), fill_value=-100, dtype=torch.long
        )
        for i, data in enumerate(batch):
            target_ids[i, :data["target_ids"].shape[0]] = data["target_ids"]

        vision_token_positions = []
        for i, data in enumerate(batch):
            offset = i * max_input_tokens
            vision_token_positions.append(data["vision_token_positions"] + offset)
        vision_token_positions = torch.cat(vision_token_positions, dim=0)

        return {
            "images": images,
            "vision_token_positions": vision_token_positions,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "target_ids": target_ids,
        }

    urls = list(chain.from_iterable([expand_urls(url) for url in urls]))

    # TODO: add cache
    return wds.DataPipeline(
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(1000, handler=wds.warn_and_continue),
        wds.decode("pilrgb", handler=wds.warn_and_continue),
        wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
        wds.map_tuple(vis_processor, handler=wds.warn_and_continue),
        wds.map(tokenize, handler=wds.warn_and_continue),
        wds.batched(batch_size, collation_fn=collate, partial=False),
    )
