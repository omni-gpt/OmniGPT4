import io
import hashlib
from typing import Callable, Tuple

import requests
import numpy as np
from ray import serve
from PIL import Image

from omnigpt4.prompts import ImageProcessor


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
class ImageProcessorDeployment:
    def __init__(self) -> None:
        self.processor = ImageProcessor()
        self.max_image_size = 4 * 1024 * 1024

    # TODO: reconfigure
    # TODO: make async

    def download_image(self, url: str) -> Tuple[Image.Image, str]:
        # TODO: add cache

        sha1 = hashlib.sha1()

        with requests.get(url, stream=True) as rsp:
            rsp.raise_for_status()

            image_size = rsp.headers["Content-length"]
            if int(image_size) > self.max_image_size:
                raise ValueError(
                    f"Image size {image_size} exceeds max size {self.max_image_size}"
                )

            image_io = io.BytesIO()
            for chunk in rsp.iter_content(1024 * 1024):
                image_io.write(chunk)
                sha1.update(chunk)

            return Image.open(image_io).convert("RGB"), sha1.hexdigest()

    def process(self, image_url: str) -> Tuple[np.ndarray, str]:
        image, image_id = self.download_image(image_url)

        return self.processor(image).numpy(), image_id
