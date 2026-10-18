"""CTC unit inventories. blank=0, unk=1 for every kind; `load_units(path)` picks the class from units.json.

  gemma     one unit per Gemma tokenizer id (QUAN-style; the original runs). units.json: {model_id, gemma_ids}
  grapheme  one unit per grapheme cluster (sinhala_asr.graphemes), plus a space unit "▁" (id 2)
  gbpe      grapheme-BPE: SentencePiece BPE trained on text where every grapheme is one symbol, so a piece is always
            a run of whole graphemes; word-initial pieces start with "▁" (scripts/13_build_units.py)

Every kind exposes the same interface:
  encode(text) -> unit ids; decode(ids) -> text; len(units)
  pieces        text of each unit (index = unit id; "" for blank/unk); "▁" starts a word, as QUAN's KenLMDecoder expects
  init_ids      Gemma token ids per unit (list[int] for gemma, list[list[int]] otherwise): the CTC head row of a
                unit starts from the (mean) Gemma text embedding of these ids
Only gemma checkpoints load with QUAN's own inference code; the others need this module.
"""

import base64
import collections
import json
import re

from transformers import AutoTokenizer

BLANK, UNK = 0, 1
WORD_START = "▁"
MODEL_ID = "google/embeddinggemma-2"

# Gemma splits Sinhala vowel signs into their own pieces, so CTC can emit impossible sequences (ිි, ෙේ). Real text
# never has two of these marks in a row: keep one (the later one if they differ). ~0.9% of words, -0.25 WER on v6 test.
_MARKS = "්ා-ෟෲෳ"
_dup = re.compile(f"([{_MARKS}])\\1+")
_two = re.compile(f"([{_MARKS}])([{_MARKS}])")
_ws = re.compile(r"\s+")


def clean_marks(text: str) -> str:
    return _two.sub(r"\2", _dup.sub(r"\1", text))


def _is_byte_piece(t: str) -> bool:
    return t.startswith("<0x") and t.endswith(">")


class Units:
    """Gemma tokenizer ids (kind "gemma")."""

    kind = "gemma"

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
        return clean_marks(self.tok.decode(ids).strip())

    @property
    def pieces(self) -> list[str]:
        if getattr(self, "_pieces", None) is None:
            toks = self.tok.convert_ids_to_tokens(self.gemma_ids)
            self._pieces = ["", ""] + ["" if _is_byte_piece(t) else t for t in toks]
        return self._pieces

    @property
    def init_ids(self) -> list[int]:
        return self.gemma_ids

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


class _TextUnits:
    """Shared parts of the grapheme and gbpe kinds: piece texts, decoding, Gemma ids for the head init."""

    kind = ""
    model_id = MODEL_ID

    def __init__(self, pieces: list[str]):
        self._pieces = ["", ""] + pieces
        self._tok = None

    def __len__(self):
        return len(self._pieces)

    @property
    def pieces(self) -> list[str]:
        return self._pieces

    @property
    def tok(self):
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
        return self._tok

    def decode(self, units: list[int]) -> str:
        from .text import normalize

        text = "".join(self._pieces[u] for u in units if u >= 2).replace(WORD_START, " ")
        # normalize() puts back the ZWJ in ්‍ය/්‍ර when the model emits e.g. ද් + ය as two units
        return normalize(_ws.sub(" ", text).strip())

    @property
    def init_ids(self) -> list[list[int]]:
        """Gemma ids of each unit's text: "▁x" -> ids of " x" (word-initial Gemma pieces), "▁" -> the "▁" token."""
        out = []
        space = self.tok.convert_tokens_to_ids(WORD_START)
        for p in self._pieces[2:]:
            if p == WORD_START:
                out.append([space])
            elif p.startswith(WORD_START):
                out.append(self.tok(" " + p[1:], add_special_tokens=False)["input_ids"] or [space])
            else:
                out.append(self.tok(p, add_special_tokens=False)["input_ids"] or [space])
        return out


class GraphemeUnits(_TextUnits):
    """One unit per grapheme cluster; unit 2 is the word separator "▁"."""

    kind = "grapheme"

    def __init__(self, graphemes: list[str]):
        super().__init__([WORD_START] + graphemes)
        self.graphemes = graphemes
        self.to_unit = {g: i + 3 for i, g in enumerate(graphemes)}
        self.to_unit[" "] = 2
        from .graphemes import WordCache

        self._split = WordCache()

    def encode(self, text: str) -> list[int]:
        return [self.to_unit.get(g, UNK) for g in self._split.sentence(text)]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": self.kind, "model_id": self.model_id, "graphemes": self.graphemes}, f, ensure_ascii=False)


class PieceUnits(_TextUnits):
    """Grapheme-BPE pieces. The SentencePiece model works on private-use symbols (one per grapheme of the alphabet);
    unit id = SentencePiece id + 1, so SentencePiece's <unk> (id 0) lands on our UNK (1)."""

    kind = "gbpe"
    UNSEEN = ""  # a symbol no grapheme maps to -> SentencePiece <unk>

    def __init__(self, sp_model: bytes, alphabet: list[str]):
        import sentencepiece as spm

        self.sp_model, self.alphabet = sp_model, alphabet
        self.sp = spm.SentencePieceProcessor(model_proto=sp_model)
        self.to_sym = {g: chr(0xE000 + i) for i, g in enumerate(alphabet)}
        to_g = {s: g for g, s in self.to_sym.items()}
        assert self.sp.unk_id() == 0 and len(alphabet) < 6400 - 1
        pieces = []
        for i in range(1, self.sp.get_piece_size()):
            p = self.sp.id_to_piece(i)
            pieces.append("" if self.sp.is_control(i) or self.sp.is_unknown(i) else "".join(to_g.get(c, c) for c in p))
        super().__init__(pieces)
        from .graphemes import WordCache

        self._split = WordCache()

    def symbols(self, text: str) -> str:
        """Text -> SentencePiece input: every grapheme becomes one private-use symbol; spaces stay."""
        out = []
        for k, w in enumerate(text.split(" ")):
            if k:
                out.append(" ")
            out.extend(self.to_sym.get(g, self.UNSEEN) for g in self._split.word(w))
        return "".join(out)

    def encode(self, text: str) -> list[int]:
        return [i + 1 for i in self.sp.encode(self.symbols(text))]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": self.kind, "model_id": self.model_id, "alphabet": self.alphabet,
                       "sp_model": base64.b64encode(self.sp_model).decode()}, f, ensure_ascii=False)


def load_units(path: str):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    kind = d.get("kind", "gemma")
    if kind == "gemma":
        return Units(d["model_id"], d["gemma_ids"])
    if kind == "grapheme":
        return GraphemeUnits(d["graphemes"])
    if kind == "gbpe":
        return PieceUnits(base64.b64decode(d["sp_model"]), d["alphabet"])
    raise ValueError(f"unknown unit kind {kind!r} in {path}")
