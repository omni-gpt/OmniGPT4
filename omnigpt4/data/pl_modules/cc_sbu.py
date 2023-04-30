import json
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader


class CCSBUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        batch_size: int = 64,
        num_workers: int = 32,
        image_size: int = 224,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        max_words: int = 50,
        backend: str = "webdataset",
    ) -> None:
        super().__init__()

        self.root_path =  Path(root_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_words = max_words
        self.backend = backend

        assert backend in ["webdataset"]

    def setup(self, stage: str) -> None:
        if self.backend == "webdataset":
            from omnigpt4.data.datasets.cc_sbu.wds_backend import build_wds_pipeline

            with open(self.root_path / "shards.json") as f:
                shards = json.load(f)

            self.dataset, self.collate_fn = build_wds_pipeline(
                # TODO: support remote urls
                [str(self.root_path / shard) for shard in shards],
                image_size=self.image_size,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                max_words=self.max_words,
            )
        else:
            raise NotImplementedError(self.backend)

    def train_dataloader(self) -> DataLoader:
        if self.backend in ["webdataset"]:
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
        else:
            raise NotImplementedError(self.backend)
