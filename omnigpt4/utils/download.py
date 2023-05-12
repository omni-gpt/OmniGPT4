import shutil
from pathlib import Path
from typing import Union, List

import asyncio
import aiohttp
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


def parallel_download_to_memeory(urls: List[str], concurrency: int = 10):
    sema = asyncio.BoundedSemaphore(concurrency)

    async def fetch_file(url: str):
        async with sema, aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                else:
                    data = None

        return data

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(fetch_file(url)) for url in urls]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

    return [task.result() for task in tasks]
