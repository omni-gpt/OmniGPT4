import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPVisionModel


def main():
    # model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    # model.save_pretrained(
    #     "./weights/openai-clip-vit-large-patch14",
    #     safe_serialization=True,
    # )

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print(state_dict.keys())


if __name__ == "__main__":
    main()
