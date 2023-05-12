import time
from typing import Any, Dict

import ray
from ray import serve
from ray.serve.dag import InputNode
from ray.serve.drivers import DAGDriver
from ray.serve.handle import RayServeDeploymentHandle
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from omnigpt4.prompts import ChatPrompts, ChatPromptManager
from .models.chat_prompt_manager import ChatPromptManagerDeployment
from .models.image_processor import ImageProcessorDeployment
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelMeta,
    ModelMetaList,
    Usage,
)

available_models = [
    ModelMeta(
        id="omnigpt4",
        permission=["all"],
    ),
]


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
class MMChatIngress:
    def __init__(self, chat_prompt_manager: RayServeDeploymentHandle) -> None:
        self.chat_prompt_manager = chat_prompt_manager

    # def inference(self, prompt: str) -> str:
    #     return "ok"

    # @app.get("/")
    # async def home(self) -> str:
    #     return "ok"

    # @app.get("/v1/models")
    # async def get_models(self) -> ModelMetaList:
    #     return ModelMetaList(data=available_models)

    # @app.get("/v1/models/{model_id}")
    # async def get_model(self, model_id: str) -> ModelMeta:
    #     for model in available_models:
    #         if model_id == model.id:
    #             return model

    #     raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if request.model not in ["omnigpt4"]:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

        max_length = request.max_tokens
        if max_length > 4096:
            max_length = 4096

        try:
            chat_promps = await self.chat_prompt_manager.get_prompt.remote(
                request.messages, max_length=max_length
            )
            eos_token_id = await self.chat_prompt_manager.get_eos_token_id.remote()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        chat_promps: ChatPrompts = ray.get(chat_promps)
        eos_token_id: int = ray.get(eos_token_id)
        chat_promps = ChatPrompts.collate([chat_promps], eos_token_id=eos_token_id)

        if request.stream:
            ...
        else:
            ...

        return ChatCompletionResponse(
            id="chat",
            created=int(time.time()),
            choices=[],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
            )
        )


with InputNode() as request:
    image_processor = ImageProcessorDeployment.bind()

    chat_prompt_manager = ChatPromptManagerDeployment.bind(image_processor)

    mmchat = MMChatIngress.bind(chat_prompt_manager)

    rsp = mmchat.chat_completions.bind(request)

    ingress = DAGDriver.options(route_prefix="/v1/chat/completions").bind(rsp, http_adapter=ChatCompletionRequest)
