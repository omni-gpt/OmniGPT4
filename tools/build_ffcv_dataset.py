import argparse
from pathlib import Path

import webdataset as wds
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, JSONField
from webdataset.shardlists import expand_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", nargs="+", required=True, help="Urls for WebDataset")
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    shards = []
    for url in args.url:
        shards += expand_urls(url)

    # Pass a type for each data field
    writer = DatasetWriter(
        args.save_path,
        fields={
            "jpg": RGBImageField(write_mode="jpg"),
            "json": JSONField(),
        },
        num_workers=1,
    )

    def pipeline(dataset: wds.WebDataset):
        dataset = dataset.decode("rgb8", handler=wds.warn_and_continue)
        dataset = dataset.to_tuple("jpg json", handler=wds.warn_and_continue)
        return dataset

    # Write dataset
    writer.from_webdataset(
        shards,
        pipeline=pipeline,
    )


if __name__ == "__main__":
    main()
