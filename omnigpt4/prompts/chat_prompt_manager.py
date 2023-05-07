import copy
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BatchEncoding

from .prompt_store import PromptStore


class Role(Enum):
    NONE = "NONE"
    HUMAN = "HUMAN"
    ASSISTANT = "ASSISTANT"


class Message(BaseModel):
    # Role of the speaker
    role: Role =  Role.NONE

    tag: str = "text"

    # Text of the message
    text: Optional[str] = None

    extra_data: Optional[Dict[str, Any]] = None

    trainable: Optional[bool] = None

    def get_text(self, prompt_store: PromptStore) -> str:
        if self.tag == "text":
            return self.text if self.text is not None else ""
        else:
            return prompt_store.get_prompt_by_tag(self.tag)


class HumanMessage(Message):
    role: Role = Role.HUMAN


class AssistantMessage(Message):
    role: Role = Role.ASSISTANT


class Conversation(BaseModel):
    human: HumanMessage

    assistant: AssistantMessage


@dataclass
class ChatPrompts:
    input_ids: torch.Tensor

    attention_mask: torch.Tensor

    pixel_values: Optional[torch.Tensor] = None

    vision_token_indices: Optional[torch.Tensor] = None

    target_ids: Optional[torch.Tensor] = None

    is_batched: bool = False

    @classmethod
    def collate(
        cls,
        batch: List["ChatPrompts"],
        eos_token_id: int,
    ) -> "ChatPrompts":
        batch_size = len(batch)

        max_input_ids = max(data.input_ids.shape[0] for data in batch)
        # padding to multiple of 8
        max_input_ids = (max_input_ids + 7) // 8 * 8
        input_ids = torch.full(
            (batch_size, max_input_ids), fill_value=eos_token_id, dtype=torch.long
        )
        attention_masks = torch.zeros(batch_size, max_input_ids, dtype=torch.long)
        for i, data in enumerate(batch):
            input_ids[i, : data.input_ids.shape[0]] = data.input_ids
            attention_masks[i, : data.input_ids.shape[0]] = data.attention_mask

        pixel_values = []
        vision_token_indices = []

        offset = 0
        for data in batch:
            if data.pixel_values is not None:
                assert data.vision_token_indices is not None, (
                    "Vision token indices must be provided if pixel values are provided."
                )
                pixel_values.append(data.pixel_values)
                vision_token_indices.append(data.vision_token_indices + offset)
                offset += max_input_ids

        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
            vision_token_indices = torch.cat(vision_token_indices, dim=0)
        else:
            pixel_values = None
            vision_token_indices = None

        if batch[0].target_ids is not None:
            max_target_ids = max(data.target_ids.shape[0] for data in batch)
            # padding to multiple of 8
            max_target_ids = (max_target_ids + 7) // 8 * 8
            target_ids = torch.full(
                (batch_size, max_target_ids), fill_value=-100, dtype=torch.long
            )
            for i, data in enumerate(batch):
                target_ids[i, :data.target_ids.shape[0]] = data.target_ids

        return cls(
            input_ids=input_ids,
            attention_mask=attention_masks,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
            target_ids=target_ids,
            is_batched=True,
        )


class ImageProcessor:
    def __init__(
        self,
        resize_size: int = 256,
        crop_size: int = 224,
        min_scale: float = 0.5,
        max_scale: float = 1.2,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.train_transform = T.Compose([
            T.RandomResizedCrop(
                size=crop_size,
                scale=(min_scale, max_scale),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        self.inference_transform = T.Compose([
            T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image: Image.Image, inference_mode: bool = False) -> torch.Tensor:
        if inference_mode:
            return self.inference_transform(image)
        else:
            return self.train_transform(image)


@dataclass
class ChatPromptManager:
    system_message: List[str] = field(default_factory=list)

    # Role name of the human user
    human_name: str = "Human"

    # Role name of the AI assistant
    assistant_name: str = "Assistant"

    # Few shot examples
    conversations: List[Conversation] = field(default_factory=list)

    conversation_template: str = "{human_name}: {human_text} ###\n{assistant_name}: {assistant_text} ###\n"

    tokenizer_name_or_path: str = "bert-base-uncased"

    image_processor: Optional[ImageProcessor] = None

    num_tokens_per_image: int = 32

    prompt_store: Optional[PromptStore] = None

    def __post_init__(self):
        if self.image_processor is None:
            self.image_processor = ImageProcessor()

    @property
    def tokenizer(self):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                use_fast=False,
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self._tokenizer

    def text_processor(self, text: str) -> str:
        text = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            text.lower(),
        )
        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        )
        text = text.rstrip("\n")
        text = text.strip(" ")

        return text

    def parse_conversations(self, conversations: dict) -> List[Conversation]:
        conversations: List[Conversation] = [
            Conversation.parse_obj(conv)
            for conv in conversations
        ]

        for conv in conversations:
            if conv.human.text is not None:
                conv.human.text = self.text_processor(conv.human.text)
            if conv.assistant.text is not None:
                conv.assistant.text = self.text_processor(conv.assistant.text)

        return conversations

    def format_conversation(
        self,
        conversation: Conversation,
        round: int,
        inference_mode: bool = False,
    ) -> Tuple[Message, Message]:
        human_text = conversation.human.get_text(self.prompt_store)
        assistant_text = conversation.assistant.get_text(self.prompt_store)

        index = self.conversation_template.find("{assistant_text}")
        assert index >= 0, "conversation_template must contain '{assistant_text}'."

        template_1 = self.conversation_template[:index]
        if template_1[-1] == " ":
            template_1 = template_1[:-1]
        message_1 = Message(
            tag="text",
            text=template_1.format(
                human_name=self.human_name,
                human_text=human_text,
                assistant_name=self.assistant_name,
                round=round,
            ),
            extra_data=conversation.human.extra_data,
            trainable=False,
        )

        template_2 = self.conversation_template[index:]
        message_2 = Message(
            tag="text",
            text=template_2.format(
                human_name=self.human_name,
                assistant_name=self.assistant_name,
                assistant_text=assistant_text,
            ),
            extra_data=conversation.assistant.extra_data,
            trainable=not inference_mode,
        )

        return message_1, message_2

    def tokenize_and_append(
        self,
        message: Union[str, Message],
        max_length: int,
        input_ids: List[int],
        pixel_values: List[torch.Tensor],
        vision_token_indices: List[List[int]],
        target_ids: Optional[List[int]] = None,
        trainable: Optional[bool] = None,
    ) -> int:
        if max_length <= 0:
            return 0

        if isinstance(message, str):
            assert trainable is not None, "trainable must be specified if message is a string."

            message = Message(
                tag="text",
                text=message,
                trainable=trainable,
            )

        text = message.get_text(self.prompt_store)

        num_remaining_tokens = max_length

        text_pieces = re.split("(\<\|ref\_\w+\_\d+\|\>)", text, flags=re.I)
        for text_piece in text_pieces:
            if num_remaining_tokens <= 0:
                break

            m = re.match("^\<\|ref\_((\w+)\_\d+)\|\>$", text_piece, flags=re.I)
            if m:
                ref_key = m[1]
                ref_type = m[2]

                ref_obj = message.extra_data[ref_key]

                if ref_type == "image":
                    if num_remaining_tokens < self.num_tokens_per_image:
                        break

                    assert isinstance(ref_obj, Image.Image), (
                        f"Image reference must be of type PIL.Image.Image, but got {type(ref_obj)}."
                    )
                    assert self.image_processor is not None, (
                        "image_processor must be specified if message contains an image."
                    )

                    pixel_values.append(self.image_processor(ref_obj))
                    start_index = len(input_ids)
                    vision_token_indices.append([start_index + i for i in range(self.num_tokens_per_image)])

                    input_ids += [self.tokenizer.unk_token_id] * self.num_tokens_per_image

                    if target_ids is not None:
                        target_ids += [-100] * self.num_tokens_per_image

                    num_remaining_tokens -= self.num_tokens_per_image
                else:
                    raise NotImplementedError(f"Unsupported reference type: {ref_type}")
            else:
                tokens: BatchEncoding = self.tokenizer(
                    text=text_piece,
                    padding=False,
                    truncation=True,
                    max_length=num_remaining_tokens,
                    add_special_tokens=False,
                )
                input_ids += tokens.input_ids
                num_tokens = len(tokens.input_ids)

                if target_ids is not None:
                    if trainable is None:
                        trainable = message.trainable

                    if trainable:
                        target_ids += tokens.input_ids
                    else:
                        target_ids += [-100] * num_tokens

                num_remaining_tokens -= num_tokens

        return max_length - num_remaining_tokens

    def get_prompt(
        self,
        conversations: Union[List[Conversation], List[dict]],
        max_length: int = 512,
        inference: bool = False,
    ) -> ChatPrompts:
        assert len(conversations) > 0, "At least one conversation is required."
        if isinstance(conversations[0], dict):
            conversations = self.parse_conversations(conversations)

        tokenizer = self.tokenizer

        input_ids = [tokenizer.bos_token_id]
        pixel_values = []
        vision_token_indices = []
        target_ids = None if inference else [-100]
        num_remaining_tokens = max_length - 1

        tokenize_and_append = partial(
            self.tokenize_and_append,
            input_ids=input_ids,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
            target_ids=target_ids,
        )

        if len(self.system_message) > 0:
            system_message = random.choice(self.system_message)
            num_remaining_tokens -= tokenize_and_append(
                system_message,
                max_length=num_remaining_tokens,
                trainable=False,
            )

        for i, conversation in enumerate(self.conversations + conversations):
            message_1, message_2 = self.format_conversation(conversation, round=i)
            num_remaining_tokens -= tokenize_and_append(
                message_1, max_length=num_remaining_tokens
            )
            num_remaining_tokens -= tokenize_and_append(
                message_2, max_length=num_remaining_tokens
            )

        # To tensor
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if len(pixel_values) > 0:
            pixel_values = torch.stack(pixel_values, dim=0)
            vision_token_indices = torch.tensor(vision_token_indices, dtype=torch.long)
        else:
            pixel_values = None
            vision_token_indices = None

        if target_ids is not None:
            target_ids = torch.as_tensor(target_ids, dtype=torch.long)

        return ChatPrompts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
            target_ids=target_ids,
        )
