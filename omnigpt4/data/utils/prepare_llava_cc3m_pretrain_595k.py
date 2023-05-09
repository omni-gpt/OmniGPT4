import argparse
import json
import io
import hashlib
import zipfile
from pathlib import Path

import webdataset as wds
from PIL import Image
from tqdm import tqdm

from omnigpt4.utils.download import download


def resize_image(image: Image.Image, target_size: int):
    width, height = image.size

    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)

    image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (256, 256), "white")

    x_offset = int((target_size - new_width) / 2)
    y_offset = int((target_size - new_height) / 2)
    new_image.paste(image, (x_offset, y_offset))

    return new_image


def process_llava_cc3m_pretrain_595k(
    sink: wds.ShardWriter,
    cache_dir: Path,
):
    images_path = cache_dir / "llava_cc3m_pretrain_595k_images.zip"
    if not images_path.exists():
        print("Downloading LLaVA-CC3M-Pretrain-595K images.zip...")
        download(
            url="https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip",
            save_path=images_path,
            progress_bar=True,
        )
        print("Done.")

    text_path = cache_dir / "llava_cc3m_pretrain_595k_chat.json"
    if not text_path.exists():
        print("Downloading LLaVA-CC3M-Pretrain-595K chat.json...")
        download(
            url="https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/raw/main/chat.json",
            save_path=text_path,
            progress_bar=True,
        )
        print("Done.")

    with open(text_path, "r") as f:
        samples = json.load(f)

    keys = set()

    with zipfile.ZipFile(images_path) as images_zip:
        for sample in tqdm(samples):
            sample_id = sample["id"]
            image_name = sample["image"]

            conversations = sample["conversations"]
            assert len(conversations) == 2
            assert conversations[0]["from"] == "human"
            assert conversations[1]["from"] == "gpt"

            text = conversations[1]["value"]
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[0:8]

            key = sample_id + "_" + text_hash
            if key in keys:
                print("deplicate key", key, "skip")
                continue
            keys.add(key)

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

            with images_zip.open(image_name, "r") as f:
                image = Image.open(f).convert("RGB")

            if image.size != (256, 256):
                image = resize_image(image, 256)

            image_bytes_io = io.BytesIO()
            image.save(image_bytes_io, format="JPEG")

            sink.write({
                "__key__": key,
                "convs.json": formatted_conversations,
                "image_0.jpg": image_bytes_io.getvalue(),
            })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./.cache")
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "llava_cc3m_pretrain_595k"
    if not output_dir.exists():
        output_dir.mkdir()

    with wds.ShardWriter(str(output_dir / "%06d.tar"), maxsize=3e8) as sink:
        process_llava_cc3m_pretrain_595k(sink, cache_dir=Path(args.cache_dir))


if __name__ == "__main__":
    main()
