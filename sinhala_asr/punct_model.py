"""Transformer word tagger for punctuation restoration (labels in sinhala_asr.punct): the label of each word is
predicted from its first sub-word token. Long inputs are tagged in overlapping chunks of words."""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from sinhala_asr.punct import LABELS

MAX_TOKENS = 320


def encode(tok, batch_words, batch_labels=None, max_tokens=MAX_TOKENS):
    enc = tok(batch_words, is_split_into_words=True, truncation=True, max_length=max_tokens, padding=True,
              return_tensors="pt")
    first = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)  # word index at first sub-word, else -100
    for b in range(len(batch_words)):
        prev = None
        for t, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid != prev:
                first[b, t] = wid
            prev = wid
    if batch_labels is not None:
        lab = torch.full_like(first, -100)
        for b, labels in enumerate(batch_labels):
            m = first[b] >= 0
            lab[b, m] = torch.tensor([int(labels[i]) for i in first[b, m].tolist()])
        enc["labels"] = lab
    return enc, first


class PunctTagger:
    def __init__(self, path: str, device: str = "cpu", dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path, dtype=dtype).to(device).eval()
        self.device = device

    @classmethod
    def from_model(cls, model, tok, device):
        self = cls.__new__(cls)
        self.tok, self.model, self.device = tok, model, device
        return self

    @torch.no_grad()
    def _tag(self, batch_words):
        enc, first = encode(self.tok, batch_words)
        logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.float().cpu()
        out = []
        for b, words in enumerate(batch_words):
            labs = [0] * len(words)
            m = first[b] >= 0
            for wid, lab in zip(first[b, m].tolist(), logits[b, m].argmax(-1).tolist()):
                labs[wid] = lab
            out.append(labs)
        return out

    def predict(self, texts: list[list[str]], batch_size: int = 32, chunk: int = 96, overlap: int = 24):
        """Labels for each word list. Lists longer than `chunk` words are cut into overlapping chunks; a word takes
        its label from the chunk where it is farthest from a cut (so every word sees >= overlap/2 words of context)."""
        pieces = []  # (text index, start, words)
        for ti, words in enumerate(texts):
            if len(words) <= chunk:
                pieces.append((ti, 0, words))
                continue
            s = 0
            while True:
                pieces.append((ti, s, words[s:s + chunk]))
                if s + chunk >= len(words):
                    break
                s += chunk - overlap
        tagged = []
        for i in range(0, len(pieces), batch_size):
            tagged += self._tag([p[2] for p in pieces[i:i + batch_size]])
        out = [[0] * len(w) for w in texts]
        best = [[-1] * len(w) for w in texts]
        for (ti, s, words), labs in zip(pieces, tagged):
            n_total = len(texts[ti])
            for k, lab in enumerate(labs):
                left = k if s > 0 else 10**6
                right = len(words) - 1 - k if s + len(words) < n_total else 10**6
                margin = min(left, right)
                if margin > best[ti][s + k]:
                    best[ti][s + k], out[ti][s + k] = margin, lab
        return out


def new_model(base: str):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    return model, tok
