import argparse
from pathlib import Path

from omnigpt4.utils.download import download


def download_metadata(output_dir: Path):
    for i in range(0, 23099):
        url = f"https://storage.googleapis.com/ai2-jackh-mmc4-public/data/docs_no_face_shard_{i}_v2.jsonl.zip"
        try:
            download(url, save_path=output_dir / f"docs_no_face_shard_{i}_v2.jsonl.zip")
        except Exception as e:
            print(e, i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./.cache")
    parser.add_argument("--output_dir", type=str, default="./data")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
