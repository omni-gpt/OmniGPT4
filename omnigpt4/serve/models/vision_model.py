from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
class VisionModelDeployment:
    def __init__(self, model_path: str) -> None:
        self.model = None

    async def __call__(self, http_request: Request) -> str:
        params = await http_request.json()
