import json
from pathlib import Path

import webdataset as wds
from tqdm import tqdm


def main():
    root_dir = Path("./data/cc_sbu_align")

    with open(root_dir / "filter_cap.json") as f:
        annos = json.load(f)["annotations"]

    with wds.TarWriter(str(root_dir / "wds.tar")) as sink:
        for annos in tqdm(annos):
            image_id = annos["image_id"]
            with open(root_dir / "image" / f"{image_id}.jpg", "rb") as f:
                image = f.read()

            sink.write({
                "__key__": f"{int(image_id):08d}",
                "jpg": image,
                "json": annos,
            })


if __name__ == "__main__":
    main()
