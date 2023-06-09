from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, root_validator


class Role(str, Enum):
    system: str = "system"
    user: str = "user"
    assistant: str = "assistant"


class FinishReason(str, Enum):
    length: str = "length"
    stop: str = "stop"
    null: str = "null"


class ObjectType(str, Enum):
    model: str = "model"
    list: str = "list"
    chat_completion: str = "chat.completion"


class ModelMeta(BaseModel):
    id: str

    object: ObjectType = ObjectType.model

    owned_by: str = "omnigpt"

    permission: List[str]


class ModelMetaList(BaseModel):
    object: str = "list"

    data: List[ModelMeta]


class Usage(BaseModel):
    prompt_tokens: int

    completion_tokens: int

    total_tokens: Optional[int] = None

    @root_validator
    def compute_total_tokens(cls, values) -> Dict:
        if "total_tokens" not in values:
            values["total_tokens"] = values["prompt_tokens"] + values["completion_tokens"]

        return values


class ChatMessage(BaseModel):
    role: Optional[Role] = None

    content: Optional[str] = None

    extra_data: Optional[Dict] = None


class ChatCompletionRequest(BaseModel):
    model: str

    # A list of messages describing the conversation so far.
    messages: List[ChatMessage]

    # What sampling temperature to use, between 0 and 2.
    # Higher values like 0.8 will make the output more random,
    # while lower values like 0.2 will make it more focused and deterministic.
    temperature: float = 1.0

    # An alternative to sampling with temperature, called nucleus sampling,
    # where the model considers the results of the tokens with top_p probability mass.
    # So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    # We generally recommend altering this or temperature but not both.
    top_p: float = 1.0

    # How many chat completion choices to generate for each input message.
    n: int = 1

    stream: bool = False

    # Up to 4 sequences where the API will stop generating further tokens.
    stop: Optional[List[str]] = None

    # The maximum number of tokens to generate in the chat completion.
    max_tokens: int = 4096

    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # whether they appear in the text so far, increasing the model's likelihood
    # to talk about new topics.
    presence_penalty: float = 0.0

    # Number between -2.0 and 2.0. Positive values penalize new tokens based on
    # their existing frequency in the text so far, decreasing the model's likelihood
    # to repeat the same line verbatim.
    frequency_penalty: float = 0.0

    # Modify the likelihood of specified tokens appearing in the completion.
    # Accepts a json object that maps tokens (specified by their token ID in the tokenizer)
    # to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits
    # generated by the model prior to sampling. The exact effect will vary per model,
    # but values between -1 and 1 should decrease or increase likelihood of selection;
    # values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    logit_bias: Optional[Dict[str, float]] = None

    user: Optional[str] = None


class ChatResponse(BaseModel):
    index: int

    message: ChatMessage

    delta: Optional[ChatMessage] = None

    finish_reason: FinishReason


class ChatCompletionResponse(BaseModel):
    id: str

    object: ObjectType = ObjectType.chat_completion

    created: datetime

    choices: List[ChatResponse]

    usage: Usage
