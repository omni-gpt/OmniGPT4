import shutil
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


def download(
    url: str,
    save_path: Union[str, Path],
    overwrite: bool = False,
    progress_bar: bool = True,
):
    save_path = Path(save_path)
    if save_path.exists() and not overwrite:
        raise FileExistsError(f"{save_path} already exists. Set `overwrite=True` to overwrite it.")

    with requests.get(url, stream=True) as r:
        with open(save_path, "wb") as f:
            if progress_bar:
                total_length = int(r.headers.get("content-length"))
                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024)):
                    if chunk:
                        f.write(chunk)
            else:
                shutil.copyfileobj(r.raw, f)
