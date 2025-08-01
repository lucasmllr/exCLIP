import csv
import gzip
import os
from os import PathLike

from torch.utils.data import Dataset

from ..models.tokenization import ClipTokenizer
from .text import Sentence, SentencePair


class STSDataset(Dataset):

    def __init__(self, path: PathLike, split: str = "train", mode: str = "train"):
        assert os.path.exists(path)
        assert split in ["train", "dev", "test"]
        assert mode in ["train", "eval"]
        self.mode = mode
        self.tokenizer = ClipTokenizer()
        self.pairs = self._load_data(path, split=split)

    def _load_data(self, path: PathLike, split: str):
        pairs = []
        with gzip.open(path, "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == split:
                    label = (
                        float(row["score"]) / 5.0
                    )  # Normalize score to range 0 ... 1
                    texts = [row["sentence1"], row["sentence2"]]
                    sentences = [
                        Sentence(
                            text=t,
                            tokens=self.tokenizer.get_tokens(t),
                            token_ids=self.tokenizer.get_token_ids(t),
                            model_input=self.tokenizer.tokenize(t),
                        )
                        for t in texts
                    ]
                    pairs.append(SentencePair(sentences, label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        if self.mode == "train":
            return (
                pair[0].model_input.squeeze(0),
                pair[1].model_input.squeeze(0),
                pair.label,
            )
        else:
            return pair
