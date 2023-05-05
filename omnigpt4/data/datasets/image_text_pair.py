import re
from itertools import chain
from typing import List

import torch
import webdataset as wds
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import BatchEncoding, LlamaTokenizer
from webdataset.shardlists import expand_urls


def text_processor(caption: str, max_words: int = 50) -> str:
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

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


def build_image_text_pair_pipeline(
    urls: List[str],
    batch_size: int = 64,
    image_size: int = 224,
    min_scale: int = 0.5,
    max_scale: int = 1.0,
    max_words: int = 50,
    max_text_tokens: int = 32,
    num_tokens_per_image: int = 32,
    tokenizer_name_or_path: str = "./weights/vicuna-7b-v0",
    end_sym: str = "\n",
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

    # TODO: make AutoTokenizer work
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    unk_token_id = tokenizer.unk_token_id

    def tokenize(sample):
        text = text_processor(sample[1]["caption"], max_words=max_words)

        tokens: BatchEncoding = tokenizer(
            text=text + end_sym,
            padding=False,
            truncation=True,
            max_length=max_text_tokens,
            add_special_tokens=False,
        )

        # TODO: support multi images, limit total tokens (text + image)
        # unk_token: image token
        input_ids = [bos_token_id] + [unk_token_id] * num_tokens_per_image + tokens.input_ids
        attention_mask = [1] * (1 + num_tokens_per_image) + tokens.attention_mask

        # to tensor
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)

        target_ids = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
        # bos + image tokens
        target_ids[:1 + num_tokens_per_image] = -100

        vision_token_positions = torch.arange(1, num_tokens_per_image + 1, dtype=torch.long)

        return {
            "images": sample[0][None, ...],
            "vision_token_positions": vision_token_positions[None, :],
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

        max_target_tokens = max(data["target_ids"].shape[0] for data in batch)
        target_ids = torch.full(
            (batch_size, max_target_tokens), fill_value=-100, dtype=torch.long
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
