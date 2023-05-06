from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple


@dataclass
class Message:
    # Role of the speaker
    role: str

    # Text of the message
    text: str

    is_prompt: bool = False


@dataclass
class Conversation:
    # System prompt
    system: str

    # Role name of the human user
    human_name: str

    # Role name of the AI assistant
    assistant_name: str

    few_shot_examples: List[Message] = field(default_factory=list)

    # All messages in the conversation
    messages: List[Message] = field(default_factory=list)

    # Separator
    sep: Union[str, Tuple[str, str]]

    # Stop criteria (the default one is EOS token)
    stop_str: Optional[str] = None

    # Stops generation if meeting any token in this list, used for ambiguous stop criteria
    stop_token_ids: Optional[Union[List[int], List[List[int]]]] = None

    def say(self, role: str, text: str) -> None:
        assert role in [self.human_name, self.assistant_name]
        self.message.append(Message(role=role, text=text))

    def human_say(self, text: str) -> None:
        self.say(self.human_name, text)

    def assistant_say(self, text: str) -> None:
        self.say(self.assistant_name, text)

    def append_prompt(self, role: str) -> None:
        self.message.append(Message(role=role, text="", is_prompt=True))

    def withdraw(self) -> Message:
        return self.message.pop()

    def to_prompt(self):
        messages = self.few_shot_examples + self.messages

        assert messages[-1].is_prompt, "The last message is not a prompt."

        assert not any([msg.is_prompt for msg in messages[:-1]]), (
            "There are more than one prompts in the conversation."
        )

        if isinstance(self.sep, str):
            sep = [self.sep, self.sep]

        prompt = self.system + sep[0]
        for i, msg in enumerate(messages):
            if msg.is_prompt:
                prompt += msg.role + ":"
            else:
                prompt += msg.role + ": " + msg.text + sep[i % 2]

        return prompt


class VicunaV0Conversation(Conversation):
    def __init__(self):
        super().__init__(
            system=(
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions."
            ),
            human_name="Human",
            assistant_name="Assistant",
            sep="\n### ",
            stop_str="###",
            stop_token_ids=[[835], [2277, 29937]], # "###" can be tokenized in two different ways.
        )


class VicunaV1Conversation(Conversation):
    def __init__(self):
        super().__init__(
            system=(
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions."
            ),
            human_name="USER",
            assistant_name="ASSISTANT",
            sep=(" ", "\n"),
        )
