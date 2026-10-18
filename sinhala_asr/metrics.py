"""ASR metrics, including word error rates that ignore where the spaces are.

Sinhala spacing is inconsistent (කළහැකි / කළ හැකි, නොහැකි වීමය / නොහැකිවීමය), so plain WER charges a word error
for speech that was recognised correctly. Reported metrics:

  wer          standard WER
  cer          standard CER (spaces count as characters)
  cer_nospace  CER with all spaces removed from both sides
  wer_seg      spacing-agnostic WER: the hypothesis characters are aligned to the reference characters (spaces
               removed from both), the hypothesis is re-split at the reference's word boundaries, then WER is
               computed. A word counts as wrong only if its letters are wrong.
  ger          grapheme error rate: CER over grapheme clusters (sinhala_asr.graphemes; spaces count, like cer)
"""

import jiwer
from rapidfuzz.distance import Levenshtein

from .graphemes import WordCache

_words = WordCache()


def ger(refs: list[str], hyps: list[str]) -> float:
    errs = n = 0
    for r, h in zip(refs, hyps):
        rg, hg = _words.sentence(r), _words.sentence(h) if h != "<empty>" else []
        errs += Levenshtein.distance(rg, hg)
        n += len(rg)
    return errs / max(1, n)


def resegment(ref: str, hyp: str) -> str:
    """Re-insert spaces into hyp at the positions aligned to ref's word boundaries."""
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    if not r or not h:
        return h
    bounds, k = set(), 0  # indices into r that start a new word (except the first)
    for w in ref.split()[:-1]:
        k += len(w)
        bounds.add(k)
    # ref index -> hyp index of the aligned (or next) hyp character
    to_h = [0] * (len(r) + 1)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(r, h):
        n = i2 - i1
        for t in range(n):
            if tag in ("equal", "replace") and j2 > j1:
                to_h[i1 + t] = j1 + min(j2 - j1 - 1, t * (j2 - j1) // n)
            else:  # deletion: ref char has no hyp char; boundary falls before the next hyp char
                to_h[i1 + t] = j1
    to_h[len(r)] = len(h)
    cuts = sorted({to_h[b] for b in bounds if 0 < to_h[b] < len(h)})
    out, prev = [], 0
    for c in cuts:
        out.append(h[prev:c])
        prev = c
    out.append(h[prev:])
    return " ".join(x for x in out if x)


def score(refs: list[str], hyps: list[str]) -> dict:
    hyps = [h if h.strip() else "<empty>" for h in hyps]
    ns = lambda xs: [x.replace(" ", "") or "<empty>" for x in xs]  # noqa: E731
    seg = [resegment(r, h) or "<empty>" for r, h in zip(refs, hyps)]
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "cer_nospace": jiwer.cer(ns(refs), ns(hyps)),
        "wer_seg": jiwer.wer(refs, seg),
        "ger": ger(refs, hyps),
        "utts": len(refs),
    }
