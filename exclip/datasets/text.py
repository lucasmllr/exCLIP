from typing import List, Optional

import torch
from torch import Tensor


class Sentence:

    def __init__(
        self,
        text: str,
        tokens: Optional[List[str]] = None,
        token_ids: Optional[List[int]] = None,
        model_input: Optional[Tensor] = None,
    ):
        self.text = text
        self.tokens = tokens
        self.token_ids = token_ids
        self.model_input = model_input

    def to(self, device: torch.device):
        assert self.model_input is not None
        assert isinstance(self.model_input, Tensor)
        self.model_input = self.model_input.to(device)

    def print_tokens(self):
        assert isinstance(self.tokens, List)
        s = ""
        for i, t in enumerate(self.tokens):
            s += f"{i+1}-{t} "
        return s.strip()


class SentencePair:

    def __init__(self, sentences: List[Sentence], label: Optional[float] = None):
        self.sentences = sentences
        self.label = torch.tensor(label)

    def __getitem__(self, idx: int):
        assert idx < len(self.sentences)
        return self.sentences[idx]

    def to(self, device: torch.device):
        for s in self.sentences:
            s.to(device)
        if self.label is not None:
            self.label = self.label.to(device)
