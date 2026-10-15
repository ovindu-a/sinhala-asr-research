"""Differential error analysis of two systems on the same test set (word level, per reference word).

Each reference word gets a status per system: correct, or one error category:
  spacing       letters right, only the word boundary differs (right after re-segmentation, wrong in plain WER)
  orthographic  same word up to ල/ළ, න/ණ, short/long vowel, ZWJ (sounds identical)
  one_letter    one letter added/dropped/changed (other than the above)
  near_miss     a few letters wrong (letter edit rate <= 1/3)
  wrong_word    a different word
  missed_word   deleted
  digits        reference word contains digits (spoken numbers vs digits)
  english       reference word is Latin script (code-mixed English)
Insertions (extra words) do not belong to a reference word and are counted per system, with "runaway"
utterances (>= 4 extra words, e.g. repetition loops) counted separately.

  uv run scripts/compare_errors.py runs/v6_top6_test.hyps.jsonl runs/whisper_test.hyps.jsonl --names ours whisper
"""

import argparse
import collections
import json
import re

import jiwer
from rapidfuzz.distance import Levenshtein

from sinhala_asr.metrics import resegment

PAIRS = [("ළ", "ල"), ("ණ", "න"), ("ී", "ි"), ("ූ", "ු"), ("ේ", "ෙ"), ("ෝ", "ො"), ("ෑ", "ැ"), ("‍", "")]
CATS = ["spacing", "orthographic", "one_letter", "near_miss", "wrong_word", "missed_word", "digits", "english"]


def canon(w):
    for a, b in PAIRS:
        w = w.replace(a, b)
    return w


def word_status(ref, hyp):
    """Per reference word: 'ok' or (category, hyp_word); plus number of inserted words."""
    rw = ref.split()
    hyp = hyp or ""

    def align(h):
        st, ins = ["ok"] * len(rw), 0
        if not h.strip():
            return [("del", None)] * len(rw), 0
        hw = h.split()
        for c in jiwer.process_words(ref, h).alignments[0]:
            if c.type == "substitute":
                for i, j in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx)):
                    st[i] = ("sub", hw[j])
            elif c.type == "delete":
                for i in range(c.ref_start_idx, c.ref_end_idx):
                    st[i] = ("del", None)
            elif c.type == "insert":
                ins += c.hyp_end_idx - c.hyp_start_idx
        return st, ins

    raw, ins = align(hyp)
    seg, _ = align(resegment(ref, hyp))
    out = []
    for k, w in enumerate(rw):
        if raw[k] == "ok":
            out.append("ok")
        elif seg[k] == "ok":
            out.append(("spacing", None))
        elif re.search(r"\d", w):
            out.append(("digits", seg[k][1]))
        elif re.search(r"[a-z]", w):
            out.append(("english", seg[k][1]))
        elif seg[k][0] == "del":
            out.append(("missed_word", None))
        else:
            h = seg[k][1]
            if canon(h) == canon(w):
                out.append(("orthographic", h))
            elif Levenshtein.distance(w, h) == 1:
                out.append(("one_letter", h))
            elif Levenshtein.distance(w, h) / max(len(w), 1) <= 1 / 3:
                out.append(("near_miss", h))
            else:
                out.append(("wrong_word", h))
    return out, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["ours", "whisper"])
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()
    A, B = args.names
    load = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]  # noqa: E731
    ra, rb = load(args.a), load(args.b)
    by_id = {r["id"]: r for r in rb}
    rb = [by_id[r["id"]] for r in ra]  # files may list utterances in different orders
    assert [r["ref"] for r in ra] == [r["ref"] for r in rb]

    only = {A: collections.Counter(), B: collections.Counter()}
    both = collections.Counter()
    tot = {A: collections.Counter(), B: collections.Counter()}
    ins = {A: 0, B: 0}
    runaway = {A: 0, B: 0}
    ex = {A: collections.defaultdict(list), B: collections.defaultdict(list)}
    n_words = 0
    for x, y in zip(ra, rb):
        sa, ia = word_status(x["ref"], x["hyp"])
        sb, ib = word_status(y["ref"], y["hyp"])
        ins[A] += ia
        ins[B] += ib
        runaway[A] += ia >= 4
        runaway[B] += ib >= 4
        rw = x["ref"].split()
        n_words += len(rw)
        for k, (p, q) in enumerate(zip(sa, sb)):
            if p != "ok":
                tot[A][p[0]] += 1
            if q != "ok":
                tot[B][q[0]] += 1
            if p != "ok" and q == "ok":
                only[A][p[0]] += 1
                if len(ex[A][p[0]]) < args.examples:
                    ex[A][p[0]].append((rw[k], p[1], q))
            elif q != "ok" and p == "ok":
                only[B][q[0]] += 1
                if len(ex[B][q[0]]) < args.examples:
                    ex[B][q[0]].append((rw[k], q[1], p))
            elif p != "ok" and q != "ok":
                both["same output" if p[1] == q[1] and p[0] == q[0] else "different outputs"] += 1

    pct = lambda c: f"{c:6,d}  ({100 * c / n_words:5.2f}% of words)"  # noqa: E731
    print(f"{n_words:,} reference words, {len(ra):,} utterances\n")
    print(f"{'category':14s} | {'total ' + A:>26s} | {'total ' + B:>26s} | {'only ' + A:>26s} | {'only ' + B:>26s}")
    for c in CATS:
        print(f"{c:14s} | {pct(tot[A][c]):>26s} | {pct(tot[B][c]):>26s} | {pct(only[A][c]):>26s} | {pct(only[B][c]):>26s}")
    sa_, sb_ = sum(tot[A].values()), sum(tot[B].values())
    print(f"{'ALL word errs':14s} | {pct(sa_):>26s} | {pct(sb_):>26s} | {pct(sum(only[A].values())):>26s} | "
          f"{pct(sum(only[B].values())):>26s}")
    print(f"\nwords both get wrong: {sum(both.values()):,} ({dict(both)})")
    print(f"extra (inserted) words: {A} {ins[A]:,}, {B} {ins[B]:,}; "
          f"runaway utterances (>=4 extra words): {A} {runaway[A]}, {B} {runaway[B]}")
    for name in (A, B):
        print(f"\nexamples of errors only {name} makes (ref -> {name}):")
        for c in CATS:
            if ex[name][c]:
                print(f"  {c:13s}", "; ".join(f"{r} -> {h or '∅'}" for r, h, _ in ex[name][c]))


if __name__ == "__main__":
    main()
