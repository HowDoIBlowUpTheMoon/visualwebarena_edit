from typing import Any

import tiktoken
from transformers import LlamaTokenizer, AutoProcessor  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            if "llava" in model_name:
                self.tokenizer = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf") # Not sure how the multimodial tokenizers work
            else:
                self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
                # turn off adding special tokens automatically
                self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
                self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
                self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
    
    # for LLava tokenizer
    def process(self, text: str, max_obs_length: int = 0) -> list[int]:
        if max_obs_length > 0: 
            return self.tokenizer.tokenizer(text, truncation = True, max_length = max_obs_length)
        else: 
            return self.tokenizer.tokenizer(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    # This breaks LLava Tokenizer!
    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
