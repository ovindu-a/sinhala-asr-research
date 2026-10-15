"""Break down where the word errors come from, for one or more *.hyps.jsonl files (ref/hyp per utterance).

  uv run scripts/error_analysis.py runs/v6_top6_test.hyps.jsonl runs/whisper_test.hyps.jsonl
"""

import collections
import json
import sys
import unicodedata

import jiwer
from rapidfuzz.distance import Levenshtein

from sinhala_asr.metrics import resegment


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cname(c):
    return {" ": "<space>"}.get(c, f"{c} ({unicodedata.name(c, '?').replace('SINHALA ', '').lower()})")


def analyse(path, top=12):
    rows = load(path)
    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] or "<empty>" for r in rows]
    w = jiwer.process_words(refs, hyps)
    n = sum(len(r.split()) for r in refs)
    seg = [resegment(r, h) or "<empty>" for r, h in zip(refs, hyps)]
    ws = jiwer.process_words(refs, seg)
    print(f"\n=== {path}  ({len(rows):,} utts, {n:,} ref words)")
    print(f"WER {w.wer:.2%} = subs {w.substitutions / n:.2%} + dels {w.deletions / n:.2%} + ins {w.insertions / n:.2%}")
    print(f"WER_seg {ws.wer:.2%}  -> spacing alone accounts for {w.wer - ws.wer:.2%} absolute "
          f"({(w.wer - ws.wer) / w.wer:.0%} of all word errors)")

    # utterance-level
    per = [(Levenshtein.distance(r.replace(' ', ''), h.replace(' ', '')) / max(1, len(r.replace(' ', ''))), r, h)
           for r, h in zip(refs, hyps)]
    perfect = sum(1 for e, _, _ in per if e == 0)
    print(f"utterances with every letter right: {perfect / len(per):.1%}; with >50% letters wrong: "
          f"{sum(1 for e, _, _ in per if e > 0.5) / len(per):.1%}")

    # word confusions after re-segmentation (real letter errors only)
    conf = collections.Counter()
    out = ws
    for k, al in enumerate(out.alignments):
        rw, hw = refs[k].split(), seg[k].split()
        for c in al:
            if c.type == "substitute":
                for i, j in zip(range(c.ref_start_idx, c.ref_end_idx), range(c.hyp_start_idx, c.hyp_end_idx)):
                    conf[(rw[i], hw[j])] += 1
    print(f"top word substitutions (ref -> hyp), after removing spacing effects:")
    for (a, b), c in conf.most_common(top):
        print(f"   {c:4d}  {a}  ->  {b}")

    # character confusions
    cc, dels, ins = collections.Counter(), collections.Counter(), collections.Counter()
    for r, h in zip(refs, hyps):
        r, h = r.replace(" ", ""), h.replace(" ", "")
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(r, h):
            if tag == "replace" and i2 - i1 == j2 - j1:
                for a, b in zip(r[i1:i2], h[j1:j2]):
                    cc[(a, b)] += 1
            elif tag == "delete" or (tag == "replace" and i2 - i1 > j2 - j1):
                dels.update(r[i1:i2])
            if tag == "insert" or (tag == "replace" and j2 - j1 > i2 - i1):
                ins.update(h[j1:j2])
    print("top letter substitutions (ref -> hyp):")
    for (a, b), c in cc.most_common(top):
        print(f"   {c:5d}  {cname(a)}  ->  {cname(b)}")
    print("most-deleted letters:", ", ".join(f"{cname(c)} {k}" for c, k in dels.most_common(6)))
    print("most-inserted letters:", ", ".join(f"{cname(c)} {k}" for c, k in ins.most_common(6)))

    # by reference length and code-mixing
    buckets = collections.defaultdict(lambda: ([], []))
    for r, h in zip(refs, hyps):
        L = len(r.split())
        b = "1 word" if L == 1 else "2-3 words" if L <= 3 else "4-6 words" if L <= 6 else "7+ words"
        buckets[b][0].append(r)
        buckets[b][1].append(h)
    print("WER by reference length:", ", ".join(f"{b} {jiwer.wer(*v):.1%} (n={len(v[0])})" for b, v in sorted(buckets.items())))
    latin = [(r, h) for r, h in zip(refs, hyps) if any("a" <= ch <= "z" for ch in r)]
    if latin:
        print(f"utterances with English words in the reference: {len(latin)}; WER {jiwer.wer(*map(list, zip(*latin))):.1%}")
    worst = sorted(per, reverse=True)[:3]
    print("examples of the worst utterances:")
    for e, r, h in worst:
        print(f"   ref: {r}\n   hyp: {h}")
    return {k: (r, h) for k, (r, h) in enumerate(zip(refs, hyps))}


if __name__ == "__main__":
    res = [analyse(p) for p in sys.argv[1:]]
    if len(res) == 2:
        a, b = sys.argv[1:]
        ra, rb = load(a), load(b)
        better = worse = 0
        for x, y in zip(ra, rb):
            ex = Levenshtein.distance(x["ref"].replace(" ", ""), (x["hyp"] or "").replace(" ", ""))
            ey = Levenshtein.distance(y["ref"].replace(" ", ""), (y["hyp"] or "").replace(" ", ""))
            better += ex < ey
            worse += ex > ey
        print(f"\nper utterance (letter errors): {a} better on {better:,}, worse on {worse:,}, tied on {len(ra) - better - worse:,}")
