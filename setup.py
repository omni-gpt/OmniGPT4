import os
import subprocess
from pathlib import Path
from setuptools import find_packages, setup

import torch

PROJECT_NAME = "OmniGPT4"
PACKAGE_NAME = PROJECT_NAME.lower().replace("-", "_")
DESCRIPTION = "OmniGPT4"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 13], "Requires PyTorch >= 1.13"


def get_version() -> str:
    cwd = Path(__file__).parent

    with open(cwd / "version.txt") as f:
        version = f.readline().strip()

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        sha = "Unknown"

    if os.getenv("OMNIGPT4_BUILD_VERSION"):
        version = os.getenv("OMNIGPT4_BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    # write version.py
    with open(cwd / PACKAGE_NAME / "version.py", "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")

    return version


def get_requirements(requirements_path: str):
    requirements = [
        line.strip()
        for line in Path(requirements_path).read_text().split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]
    return [req for req in requirements if req and not req.startswith(('"', "#", "-", "git+"))]


if __name__ == "__main__":
    version = get_version()

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Ming Yang",
        author_email="ymviv@qq.com",
        url=f"https://github.com/omni-gpt/{PROJECT_NAME}",
        download_url=f"https://github.com/omni-gpt/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.so", "*.dylib"]},
        zip_safe=False,
        python_requires=">=3.10",
        entry_points={
            "console_scripts": [f"{PROJECT_NAME} = {PROJECT_NAME}.__main__:main"]
        },
        install_requires=get_requirements("requirements.txt"),
        extras_require={
            "dev": get_requirements("requirements-dev.txt"),
        }
    )