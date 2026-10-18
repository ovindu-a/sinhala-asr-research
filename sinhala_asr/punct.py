"""Punctuation restoration for Sinhala ASR output: labels, data extraction, baselines and scoring.

Task: tag every word of normalized (punctuation-free) ASR text with the mark that follows it:
  0 none · 1 comma (, ; :) · 2 period (. ! ෴ ।) · 3 question (?)
Labels come from ordinary punctuated text: split it into words with the ASR normalizer (sinhala_asr.text.normalize)
and record the mark after each word, so the tagger sees exactly what the ASR emits.
"""

import re
import unicodedata

from sinhala_asr.text import normalize

LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
MARKS = ["", ",", ".", "?"]
O, COMMA, PERIOD, QUESTION = range(4)
_MARK = {",": COMMA, ";": COMMA, ":": COMMA, ".": PERIOD, "!": PERIOD, "෴": PERIOD, "।": PERIOD, "…": PERIOD,
         "?": QUESTION}
_CLOSERS = set("\"'”’»)]}")
_tok = re.compile(r"\S+")


def words_and_labels(text: str) -> tuple[list[str], list[int]]:
    """Normalized words of `text` and the punctuation label of each. A mark counts only at the end of a
    whitespace token (so abbreviations like එල්.ටී.ටී.ඊ and decimals keep their inner dots unlabelled); a token
    that is only punctuation (" . ") labels the word before it. Strongest mark wins: ? > . > ,"""
    words, labels = [], []
    for tok in _tok.findall(unicodedata.normalize("NFC", text)):
        end = len(tok)
        while end and (tok[end - 1] in _MARK or tok[end - 1] in _CLOSERS):
            end -= 1
        lab = max((_MARK[c] for c in tok[end:] if c in _MARK), default=O)
        ws = normalize(tok[:end]).split()
        words += ws
        labels += [O] * len(ws)
        if lab and labels:
            labels[-1] = max(labels[-1], lab)
    return words, labels


def sentence_ends(labels: list[int]) -> list[int]:
    return [i for i, lab in enumerate(labels) if lab in (PERIOD, QUESTION)]


def render(words: list[str], labels: list[int]) -> str:
    return " ".join(w + MARKS[lab] for w, lab in zip(words, labels))


# --- baselines -----------------------------------------------------------------------------------------------------

_Q = re.compile(r"(ද|දැයි|නේ|නේද|ඇයි)$")


def question_rule(word: str) -> bool:
    """Sinhala yes/no and wh- questions end in the particle ද (කොහොමද, මොකක්ද, ගියාද...) or a tag (නේ, නේද)."""
    return bool(_Q.search(word))


def end_only(words: list[str]) -> list[int]:
    """B0: nothing inside, and the transcript ends with ? if its last word looks like a question, else a period."""
    labs = [O] * len(words)
    if words:
        labs[-1] = QUESTION if question_rule(words[-1]) else PERIOD
    return labs


class WordStats:
    """B1: most frequent label after each word in training text, backing off to the word's last 3/2/1 letters.
    The last word of a transcript always gets . or ? (its best sentence-end label)."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.counts: dict[str, list[int]] = {}

    def _keys(self, w: str):
        return [w] + [f"~{w[-k:]}" for k in (3, 2, 1) if len(w) > k]

    def add(self, words, labels):
        for w, lab in zip(words, labels):
            for k in self._keys(w):
                c = self.counts.setdefault(k, [0, 0, 0, 0])
                c[lab] += 1

    def _dist(self, w):
        for k in self._keys(w):
            c = self.counts.get(k)
            if c and sum(c) >= self.min_count:
                return c
        return [1, 0, 0, 0]

    def predict(self, words):
        labs = []
        for w in words:
            c = self._dist(w)
            labs.append(max(range(4), key=lambda i: c[i]))
        if words:
            c = self._dist(words[-1])
            labs[-1] = QUESTION if c[QUESTION] > c[PERIOD] or question_rule(words[-1]) and c[QUESTION] else PERIOD
        return labs


# --- scoring -------------------------------------------------------------------------------------------------------


class Scores:
    """Per-class precision/recall/F1 over word slots, plus micro-F1 over the three marks and 'sentence-end' F1
    (period and question merged: was a boundary placed at the right word?)."""

    def __init__(self):
        self.tp, self.fp, self.fn = [0] * 4, [0] * 4, [0] * 4
        self.end = [0, 0, 0]  # tp fp fn for {PERIOD, QUESTION} as one class
        self.n_words = 0

    def add(self, ref, hyp):
        assert len(ref) == len(hyp)
        self.n_words += len(ref)
        for r, h in zip(ref, hyp):
            if r == h:
                self.tp[r] += 1
            else:
                self.fp[h] += 1
                self.fn[r] += 1
            re_, he = r in (PERIOD, QUESTION), h in (PERIOD, QUESTION)
            self.end[0] += re_ and he
            self.end[1] += he and not re_
            self.end[2] += re_ and not he

    @staticmethod
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    def summary(self) -> dict:
        out: dict = {"words": self.n_words}
        for i in (COMMA, PERIOD, QUESTION):
            p, r, f = self._prf(self.tp[i], self.fp[i], self.fn[i])
            out[LABELS[i]] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4), "n": self.tp[i] + self.fn[i]}
        p, r, f = self._prf(sum(self.tp[1:]), sum(self.fp[1:]), sum(self.fn[1:]))
        out["micro"] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        p, r, f = self._prf(*self.end)
        out["sentence_end"] = {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        return out


def transfer_labels(ref_words, ref_labels, hyp_words) -> list[int]:
    """Word-level Levenshtein alignment of ASR output to the reference; each hyp word takes the label of the ref
    word it is aligned to (match or substitution), and a deleted ref word passes its label to the hyp word before
    it, so 'reference punctuation' is defined on the ASR's own words."""
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]))
    out = [O] * m
    i, j = n, m
    pending = O  # label of ref words deleted after the current hyp position
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]):
            out[j - 1] = max(ref_labels[i - 1], pending)
            pending = O
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            pending = max(pending, ref_labels[i - 1])
            i -= 1
        else:
            out[j - 1] = max(out[j - 1], pending)
            pending = O
            j -= 1
    return out
