import json
import random
from typing import List, Optional

from braceexpand import braceexpand


class PromptStore:
    def __init__(
        self,
        urls: Optional[List[str]] = None,
    ):
        if urls is None:
            urls = []

        expanded_urls = []
        for url in urls:
            expanded_urls.extend(braceexpand(url))

        self.urls = expanded_urls

        self.prompts = {}

        prompts_list = []
        for url in self.urls:
            if url.endswith(".json"):
                with open(url) as f:
                    prompts_list += json.load(f)
            elif url.endswith(".jsonl"):
                with open(url) as f:
                    for line in f.read().splitlines():
                        prompts_list.append(json.loads(line))

        for prompt in prompts_list:
            tag = prompt["tag"]
            if tag not in self.prompts:
                self.prompts[tag] = []
            self.prompts[tag].append(prompt["text"])

    def get_prompt_by_tag(self, tag: str) -> str:
        assert tag in self.prompts, f"Prompt tag {tag} not found."
        return random.choice(self.prompts[tag])
