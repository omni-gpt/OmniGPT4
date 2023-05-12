import re
from typing import List

import ray
from ray import serve
from ray.serve.handle import RayServeDeploymentHandle

from omnigpt4.prompts import (
    ChatPromptManager, Conversation, Message, HumanMessage, AssistantMessage
)
from omnigpt4.serve.schemas import ChatMessage, Role

image_link_pattern = re.compile(
    r"(?:[!]\[(?P<caption>.*?)\])\((?P<image>.*?)\)"
)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
class ChatPromptManagerDeployment:
    def __init__(
        self,
        image_processor: RayServeDeploymentHandle
    ) -> None:
        self.chat_prompt_manager = ChatPromptManager()
        self.image_processor = image_processor

    # TODO: reconfigure
    # TODO: make async

    def get_eos_token_id(self):
        return self.chat_prompt_manager.tokenizer.eos_token_id

    def get_prompt(self, messages: List[ChatMessage], max_length: int = 4096) -> str:
        chat_prompt_manager = self.chat_prompt_manager.copy()

        conversations: List[Conversation] = []

        last_conv = None
        last_role = None

        for msg in messages:
            if msg.role == Role.user:
                if last_role == Role.assistant:
                    assert last_conv is not None
                    conversations.append(last_conv)
                    last_conv = None

                if last_conv is None:
                    last_conv = Conversation(
                        human=HumanMessage(text=""),
                        assistant=AssistantMessage(text=""),
                    )
                last_conv.human.text += ("\n" + msg.content)
            elif msg.role == Role.assistant:
                last_conv.assistant.text += ("\n" + msg.content)
            elif msg.role == Role.system:
                # TODO: add stop token
                chat_prompt_manager.system_message.append(msg.content + "\n")
            else:
                raise ValueError(f"Invalid role {msg.role}")

            last_role = msg.role

        if last_conv is not None:
            conversations.append(last_conv)

        image_refs = {}

        def find_images(msg: Message):
            if msg.text is None:
                return

            for m in image_link_pattern.finditer(msg.text):
                image_url = m.group("image")
                image_refs[image_url] = self.image_processor.process.remote(image_url)

        for conv in conversations:
            find_images(conv.human)
            find_images(conv.assistant)

        def replace_images(msg: Message):
            if msg.text is None:
                return

            def replace_fn(m):
                image_url = m.group("image")
                image, image_id = ray.get(image_refs[image_url])
                key = f"image_{image_id}"
                msg.extra_data[key] = image
                return f"<img><|ref_{key}|></img>"

            msg.text = image_link_pattern.sub(replace_fn, msg.text)

        for conv in conversations:
            replace_images(conv.human)
            replace_images(conv.assistant)

        return chat_prompt_manager.get_prompt(
            conversations,
            max_length=max_length,
            inference=True,
            return_tensors="np",
        )
