import argparse
import json
from pathlib import Path

import webdataset as wds
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", action="append", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "ccs_synthetic_filtered_large"
    if not output_dir.exists():
        output_dir.mkdir()

    with wds.ShardWriter(str(output_dir / "%06d.tar"), maxsize=3e8) as sink:
        for url in args.urls:
            pipe = wds.WebDataset(url)

            for sample in tqdm(pipe):
                meta = json.loads(sample["json"])
                text = sample["txt"].decode("utf-8")

                formatted_conversations = [{
                    "human": {
                        "tag": "mm:describe_one_image",
                        "extra_data": {
                            "image_0": "image_0.jpg",
                        },
                    },
                    "assistant": {
                        "text": text,
                    },
                }]

                sink.write({
                    "__key__": meta["sha256"],
                    "convs.json": formatted_conversations,
                    "image_0.jpg": sample["jpg"],
                })


if __name__ == "__main__":
    main()
