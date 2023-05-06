from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import lightning.pytorch as pl
import webdataset as wds
from torch.utils.data import DataLoader

from omnigpt4.data.datasets import build_image_text_pair_pipeline


@dataclass
class DatasetConfig:
    urls: List[str]
    sample_rate: float = 1.0


class MultiSourcePipeline(wds.DataPipeline):
    def __init__(
        self,
        src_pipelines: List[wds.DataPipeline],
        sample_rates: List[float],
    ) -> None:
        super().__init__()

        self.src_pipelines = src_pipelines
        self.sample_rates = np.array(sample_rates, dtype=np.float32)

        total = self.sample_rates.sum()
        if total > 0:
            self.sample_rates /= total

    def __iter__(self):
        iters = [iter(pipeline) for pipeline in self.src_pipelines]

        while True:
            idx = np.random.choice(len(iters), p=self.sample_rates)
            yield next(iters[idx])


class ImageTextPair(pl.LightningDataModule):
    def __init__(
        self,
        datasets: List[DatasetConfig],
        batch_size: int = 64,
        num_workers: int = 32,
        image_size: int = 224,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        max_tokens: int = 32,
        num_tokens_per_image: int = 32,
        tokenizer_name_or_path: str = "bert-base-uncased",
        end_sym: str = "\n",
        prompt_template: str = "",
        prompts_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.dataset_configs = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_tokens = max_tokens
        self.num_tokens_per_image = num_tokens_per_image
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        self.prompts_path = prompts_path

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_pipelines = [
                build_image_text_pair_pipeline(
                    config.urls,
                    batch_size=self.batch_size,
                    image_size=self.image_size,
                    min_scale=self.min_scale,
                    max_scale=self.max_scale,
                    max_tokens=self.max_tokens,
                    num_tokens_per_image=self.num_tokens_per_image,
                    tokenizer_name_or_path=self.tokenizer_name_or_path,
                    end_sym=self.end_sym,
                    prompt_template=self.prompt_template,
                    prompts_path=self.prompts_path,
                )
                for config in self.dataset_configs
            ]
            sample_rates = [config.sample_rate for config in self.dataset_configs]
            self.train_pipeline = MultiSourcePipeline(train_pipelines, sample_rates)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_pipeline,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
