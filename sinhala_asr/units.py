"""CTC unit inventory over Gemma tokenizer ids (QUAN-style): blank=0, unk=1, then one unit per Gemma id."""

import collections
import json

from transformers import AutoTokenizer

BLANK, UNK = 0, 1


class Units:
    def __init__(self, model_id: str, gemma_ids: list[int]):
        self.model_id = model_id
        self.gemma_ids = gemma_ids
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.to_unit = {g: i + 2 for i, g in enumerate(gemma_ids)}

    def __len__(self):
        return len(self.gemma_ids) + 2

    def gemma_encode(self, text: str) -> list[int]:
        # leading space so every word, including the first, starts with "▁"
        return self.tok(" " + text, add_special_tokens=False)["input_ids"]

    def encode(self, text: str) -> list[int]:
        return [self.to_unit.get(g, UNK) for g in self.gemma_encode(text)]

    def decode(self, units: list[int]) -> str:
        ids = [self.gemma_ids[u - 2] for u in units if u >= 2]
        return self.tok.decode(ids).strip()

    @classmethod
    def build(cls, model_id: str, texts, min_count: int = 1) -> "Units":
        tmp = cls(model_id, [])
        counts = collections.Counter(g for t in texts for g in tmp.gemma_encode(t))
        ids = sorted(g for g, c in counts.items() if c >= min_count)
        return cls(model_id, ids)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"model_id": self.model_id, "gemma_ids": self.gemma_ids}, f)

    @classmethod
    def load(cls, path: str) -> "Units":
        with open(path) as f:
            d = json.load(f)
        return cls(d["model_id"], d["gemma_ids"])
