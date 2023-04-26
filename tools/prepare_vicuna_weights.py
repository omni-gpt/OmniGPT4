import argparse
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.convert_llama_weights_to_hf import write_model, write_tokenizer

DELTA_PATH_MAP = {
    "7B": "lmsys/vicuna-7b-delta",
    "13B": "lmsys/vicuna-13b-delta",
}


def convert_llama_weights_to_hf(
    llama_dir: str, model_size: str, output_dir: str
) -> None:
    write_model(
        model_path=output_dir,
        input_base_path=os.path.join(llama_dir, model_size),
        model_size=model_size,
    )

    write_tokenizer(output_dir, os.path.join(llama_dir, "tokenizer.model"))


def apply_delta_legacy(
    base_model_path: str, target_model_path: str, delta_path: str
) -> None:
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)

    DEFAULT_PAD_TOKEN = "[PAD]"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    num_new_tokens = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data
    input_embeddings[-num_new_tokens:] = 0
    output_embeddings[-num_new_tokens:] = 0

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


def apply_delta(
    base_model_path: str, target_model_path: str, delta_path: str
) -> None:
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(
        delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B"],
    )
    parser.add_argument(
        "--version",
        choices=["v0", "v1.1"],
        default="v1.1",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()

    hf_llama_dir = os.path.join(args.output_dir, f"llama-{args.model_size.lower()}")
    if not os.path.exists(hf_llama_dir):
        convert_llama_weights_to_hf(
            llama_dir=args.llama_dir,
            model_size=args.model_size,
            output_dir=hf_llama_dir,
        )

    hf_vicuna_dir = os.path.join(
        args.output_dir, f"vicuna-{args.model_size.lower()}-{args.version}"
    )
    if not os.path.exists(hf_vicuna_dir):
        if args.version == "v0":
            apply_delta_fn = apply_delta_legacy
        else:
            apply_delta_fn = apply_delta

        apply_delta_fn(
            base_model_path=hf_llama_dir,
            target_model_path=hf_vicuna_dir,
            delta_path=f"{DELTA_PATH_MAP[args.model_size]}-{args.version}",
        )


if __name__ == "__main__":
    main()
