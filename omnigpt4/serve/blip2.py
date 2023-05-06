from ray import serve
from starlette.requests import Request


class BLIP2Deployment:
    def __init__(self, model_name_or_path: str) -> None:
        self.model = None
