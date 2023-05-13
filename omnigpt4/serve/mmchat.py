import time

import ray
from ray import serve
from ray.serve.dag import InputNode
from ray.serve.drivers import DAGDriver
from ray.serve.handle import RayServeDeploymentHandle
from starlette.exceptions import HTTPException

from omnigpt4.prompts import ChatPrompts
from .models.chat_prompt_manager import ChatPromptManagerDeployment
from .models.image_processor import ImageProcessorDeployment
from .models.omnigpt4_model import OmniGPT4Deployment
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatResponse,
    FinishReason,
    Role,
    ModelMeta,
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
    def __init__(
        self,
        chat_prompt_manager: RayServeDeploymentHandle,
        omnigpt4_model: RayServeDeploymentHandle,
    ) -> None:
        self.chat_prompt_manager = chat_prompt_manager
        self.omnigpt4_model = omnigpt4_model

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

        outputs = await self.omnigpt4_model.generate.remote(
            chat_promps,
            max_length=max_length,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.n,
        )
        messages, num_completion_tokens = ray.get(outputs)

        choices = [
            ChatResponse(
                index=i,
                message=ChatMessage(
                    role=Role.assistant,
                    content=msg,
                ),
                finish_reason=FinishReason.stop,
            )
            for i, msg in enumerate(messages)
        ]

        return ChatCompletionResponse(
            id="chat",
            created=int(time.time()),
            choices=choices,
            usage=Usage(
                prompt_tokens=sum(chat_promps.num_tokens),
                completion_tokens=sum(num_completion_tokens),
            )
        )


with InputNode() as request:
    image_processor = ImageProcessorDeployment.bind()

    chat_prompt_manager = ChatPromptManagerDeployment.bind(image_processor)

    omnigpt4_model = OmniGPT4Deployment.bind()

    mmchat = MMChatIngress.bind(chat_prompt_manager, omnigpt4_model)

    rsp = mmchat.chat_completions.bind(request)

    ingress = DAGDriver.options(route_prefix="/v1/chat/completions").bind(rsp, http_adapter=ChatCompletionRequest)
