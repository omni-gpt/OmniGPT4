import copy
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import numpy as np
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

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

    to_be_predicted: Optional[bool] = None

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
    input_ids: Union[torch.Tensor, np.ndarray]

    attention_mask: Union[torch.Tensor, np.ndarray]

    pixel_values: Optional[Union[torch.Tensor, np.ndarray]] = None

    vision_token_indices: Optional[Union[torch.Tensor, np.ndarray]] = None

    target_ids: Optional[Union[torch.Tensor, np.ndarray]] = None

    num_tokens: Union[int, List[int]] = 0

    is_batched: bool = False

    def __post_init__(self):
        if self.is_batched:
            assert isinstance(self.num_tokens, list)
        else:
            assert isinstance(self.num_tokens, int)

    @classmethod
    def collate(
        cls,
        batch: List["ChatPrompts"],
        eos_token_id: int,
        pad_to_multiple_of: int = 1,
    ) -> "ChatPrompts":
        for data in batch:
            assert not data.is_batched, "Cannot collate batched samples."

        batch_size = len(batch)

        max_input_ids = max(data.input_ids.shape[0] for data in batch)
        max_input_ids = (max_input_ids + (pad_to_multiple_of - 1)) // pad_to_multiple_of * pad_to_multiple_of

        if isinstance(batch[0].input_ids, np.ndarray):
            input_ids = np.full(
                (batch_size, max_input_ids), fill_value=eos_token_id, dtype=np.int64
            )
            attention_masks = np.zeros_like(input_ids)
        else:
            input_ids = torch.full(
                (batch_size, max_input_ids), fill_value=eos_token_id, dtype=torch.long
            )
            attention_masks = torch.zeros_like(input_ids)

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
            if isinstance(pixel_values[0], np.ndarray):
                pixel_values = np.concatenate(pixel_values, axis=0)
                vision_token_indices = np.concatenate(vision_token_indices, axis=0)
            else:
                pixel_values = torch.cat(pixel_values, dim=0)
                vision_token_indices = torch.cat(vision_token_indices, dim=0)
        else:
            pixel_values = None
            vision_token_indices = None

        if batch[0].target_ids is not None:
            if isinstance(batch[0].target_ids, np.ndarray):
                target_ids = np.full(
                    (batch_size, max_input_ids), fill_value=-100, dtype=np.int64
                )
            else:
                target_ids = torch.full(
                    (batch_size, max_input_ids), fill_value=-100, dtype=torch.long
                )

            for i, data in enumerate(batch):
                target_ids[i, :data.target_ids.shape[0]] = data.target_ids
        else:
            target_ids = None

        return cls(
            input_ids=input_ids,
            attention_mask=attention_masks,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
            target_ids=target_ids,
            num_tokens=[data.num_tokens for data in batch],
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

    tokenizer_name_or_path: str = "bigscience/bloomz-7b1"

    tokenizer: Optional[PreTrainedTokenizer] = None

    image_processor: Optional[ImageProcessor] = None

    num_tokens_per_image: int = 32

    prompt_store: Optional[PromptStore] = None

    def __post_init__(self):
        if self.image_processor is None:
            self.image_processor = ImageProcessor()

        if self.tokenizer is None:
            trust_remote_code = self.tokenizer_name_or_path in ["THUDM/chatglm-6b"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path,
                padding_side="left",
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def copy(self) -> "ChatPromptManager":
        return ChatPromptManager(
            system_message=copy.copy(self.system_message),
            human_name=self.human_name,
            assistant_name=self.assistant_name,
            conversations=copy.copy(self.conversations),
            conversation_template=self.conversation_template,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            num_tokens_per_image=self.num_tokens_per_image,
            prompt_store=self.prompt_store,
        )

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
        text_1 = template_1.format(
            human_name=self.human_name,
            human_text=human_text,
            assistant_name=self.assistant_name,
            round=round,
        )

        # Remove trailing newline if assistant_text is empty
        if not assistant_text:
            text_1 = text_1.rstrip()

        message_1 = Message(
            tag="text",
            text=text_1,
            extra_data=conversation.human.extra_data,
            trainable=False,
        )

        if assistant_text:
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
        else:
            message_2 = Message(
                tag="text",
                text="",
                trainable=False,
                to_be_predicted=True,
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

        if len(text) == 0:
            return 0

        num_remaining_tokens = max_length

        text_pieces = re.split("(\<\|ref\_\w+\_[a-z0-9]+\|\>)", text, flags=re.I)
        for text_piece in text_pieces:
            if num_remaining_tokens <= 0:
                break

            m = re.match("^\<\|ref\_((\w+)\_[a-z0-9]+)\|\>$", text_piece, flags=re.I)
            if m:
                ref_key = m[1]
                ref_type = m[2]

                ref_obj = message.extra_data[ref_key]

                if ref_type == "image":
                    if num_remaining_tokens < self.num_tokens_per_image:
                        break

                    assert isinstance(ref_obj, (Image.Image, np.ndarray)), (
                        "Image reference must be of type PIL.Image.Image or np.ndarray, "
                        f"but got {type(ref_obj)}."
                    )
                    assert self.image_processor is not None, (
                        "image_processor must be specified if message contains an image."
                    )

                    if isinstance(ref_obj, Image.Image):
                        pixel_values.append(self.image_processor(ref_obj))
                    else:
                        pixel_values.append(torch.from_numpy(ref_obj.copy()))
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
        return_tensors: str = "pt",
    ) -> ChatPrompts:
        assert len(conversations) > 0, "At least one conversation is required."
        if isinstance(conversations[0], dict):
            conversations = self.parse_conversations(conversations)

        input_ids = [self.tokenizer.bos_token_id]
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

        all_conversations = self.conversations + conversations
        for i, conversation in enumerate(all_conversations):
            message_1, message_2 = self.format_conversation(conversation, round=i)
            if message_2.to_be_predicted:
                assert i == len(all_conversations) - 1, (
                    "Only the last message of the last conversation can be to_be_predicted."
                )
            num_remaining_tokens -= tokenize_and_append(
                message_1, max_length=num_remaining_tokens
            )
            num_remaining_tokens -= tokenize_and_append(
                message_2, max_length=num_remaining_tokens
            )

        input_ids = np.asarray(input_ids, dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)

        if len(pixel_values) > 0:
            pixel_values = np.stack(pixel_values, axis=0)
            vision_token_indices = np.asarray(vision_token_indices, dtype=np.int64)
        else:
            pixel_values = None
            vision_token_indices = None

        if target_ids is not None:
            target_ids = np.asarray(target_ids, dtype=np.int64)

        # To tensor
        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.from_numpy(attention_mask)

            if len(pixel_values) > 0:
                pixel_values = torch.from_numpy(pixel_values)
                vision_token_indices = torch.from_numpy(vision_token_indices)

            if target_ids is not None:
                target_ids = torch.from_numpy(target_ids)
        else:
            assert return_tensors == "np", (
                f"return_tensors must be 'pt' or 'np', but got {return_tensors}."
            )

        return ChatPrompts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            vision_token_indices=vision_token_indices,
            target_ids=target_ids,
            num_tokens=input_ids.shape[0],
        )
