import argparse

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    weight: torch.Tensor = ckpt["_forward_module.model.language_projection.weight"]
    bias: torch.Tensor = ckpt["_forward_module.model.language_projection.bias"]

    save_file(
        {
            "language_projection.weight": weight.clone(),
            "language_projection.bias": bias.clone(),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
