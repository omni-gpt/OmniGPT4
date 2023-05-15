from functools import partial
from typing import Optional

import webdataset as wds

from omnigpt4.prompts import ChatPromptManager, ChatPrompts


def build_mm_chat_pipeline(
    urls: str,
    batch_size: int = 64,
    max_length: int = 256,
    chat_prompt_manager: Optional[ChatPromptManager] = None,
    shuffle_buffer_size: int = 1000,
    inference_mode: bool = False,
) -> wds.DataPipeline:
    if chat_prompt_manager is None:
        chat_prompt_manager = ChatPromptManager()

    tokenizer = chat_prompt_manager.tokenizer
    eos_token_id = tokenizer.eos_token_id

    def process(sample: dict) -> ChatPrompts:
        conversations = sample["convs.json"]

        # Prepare reference objects
        for conv in conversations:
            for role in ["human", "assistant"]:
                message = conv[role]

                if "extra_data" not in message:
                    continue

                for ref_key, ref_path in message["extra_data"].items():
                    ref_obj = sample[ref_path]
                    message["extra_data"][ref_key] = ref_obj

        res = chat_prompt_manager.get_prompt(
            conversations=conversations,
            max_length=max_length,
            inference=inference_mode,
            return_tensors="pt",
        )

        return res

    return wds.DataPipeline(
        wds.ResampledShards(urls),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.shuffle(shuffle_buffer_size, handler=wds.warn_and_continue),
        wds.decode("pilrgb", handler=wds.warn_and_continue),
        wds.map(process, handler=wds.warn_and_continue),
        wds.batched(
            batch_size,
            collation_fn=partial(
                ChatPrompts.collate,
                eos_token_id=eos_token_id,
                pad_to_multiple_of=8,
            ),
            partial=False,  # drop last batch if it's not full
        ),
    )
