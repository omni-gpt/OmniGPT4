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


def process_llava_instruct_150k(
    sink: wds.ShardWriter,
    cache_dir: Path,
):
    images_path = cache_dir / "train2014.zip"
    if not images_path.exists():
        print("Downloading train2014.zip...")
        download(
            url="http://images.cocodataset.org/zips/train2014.zip",
            save_path=images_path,
            progress_bar=True,
        )
        print("Done.")

    text_path = cache_dir / "llava_instruct_150k.json"
    if not text_path.exists():
        print("Downloading llava_instruct_150k.json...")
        download(
            url="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json",
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
            assert len(conversations) % 2 == 0

            text = ""
            for i, conv in enumerate(conversations):
                if i % 2 == 0:
                    assert conv["from"] == "human"
                else:
                    assert conv["from"] == "gpt"
                text += conv["value"] + "\n"

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
            for i in range(0, len(conversations), 2):
                conv = {
                    "human": {
                        "text": conversations[i]["value"],
                    },
                    "assistant": {
                        "text": conversations[i + 1]["value"],
                    },
                }
                if "<image>" in conv["human"]["text"]:
                    conv["human"]["text"] = conv["human"]["text"].replace("<image>", "<img><|ref_image_0|></img>")
                    conv["human"]["extra_data"] = {
                        "image_0": "image_0.jpg",
                    }
                formatted_conversations.append(conv)

            with images_zip.open("train2014/COCO_train2014_" + image_name, "r") as f:
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
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "llava_instruct_150k"
    if not output_dir.exists():
        output_dir.mkdir()

    if args.cache_dir is None:
        cache_dir = output_dir / ".cache"
    else:
        cache_dir = Path(args.cache_dir)

    with wds.ShardWriter(str(output_dir / "%06d.tar"), maxsize=3e8) as sink:
        process_llava_instruct_150k(sink, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
