import argparse
import json
from pathlib import Path

import webdataset as wds

from omnigpt4.utils.download import download


def process_llava_cc3m_pretrain_595k(cache_dir: Path):
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
        chats = json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=".cache")

    wds.ShardWriter()


if __name__ == "__main__":
    main()
