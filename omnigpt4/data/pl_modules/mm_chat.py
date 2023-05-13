from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import lightning.pytorch as pl
import webdataset as wds
from torch.utils.data import DataLoader

from omnigpt4.data.datasets import build_mm_chat_pipeline
from omnigpt4.prompts import ChatPromptManager


@dataclass
class DatasetConfig:
    urls: str
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
        iters = [iter(pipeline.repeat()) for pipeline in self.src_pipelines]

        while True:
            idx = np.random.choice(len(iters), p=self.sample_rates)
            yield next(iters[idx])


class MMChat(pl.LightningDataModule):
    def __init__(
        self,
        datasets: List[DatasetConfig],
        batch_size: int = 64,
        num_workers: int = 32,
        max_length: int = 256,
        chat_prompt_manager: Optional[ChatPromptManager] = None,
        shuffle_buffer_size: int = 1000,
    ) -> None:
        super().__init__()

        self.dataset_configs = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size

        if chat_prompt_manager is None:
            chat_prompt_manager = ChatPromptManager()
        self.chat_prompt_manager = chat_prompt_manager

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_pipelines = [
                build_mm_chat_pipeline(
                    config.urls,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    chat_prompt_manager=self.chat_prompt_manager,
                    shuffle_buffer_size=self.shuffle_buffer_size,
                    inference_mode=False,
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
