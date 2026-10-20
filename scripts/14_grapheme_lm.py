"""Grapheme n-gram KenLM for CTC shallow fusion with grapheme units (B): the word LM's text, one token per grapheme.

Each corpus sentence (the exact si_4gram text, 07_build_lm.py --corpus-only) is split into graphemes
(sinhala_asr.graphemes) and written space-separated, with "|" for the word break:
    අම්මා ගෙදර  ->  අ ම් මා | ගෙ ද ර
so the LM scores every grapheme as it is emitted, including where the spaces go, and handles words it never saw.
Decode with: 08_lm_decode.py decode --lm <out> --lm-kind grapheme (grapheme units only).

writes <out>/lm.bin, <out>/vocab.txt (graphemes + "|", most frequent first), <out>/lm_config.json
Order > 6 needs KenLM built with KENLM_MAX_ORDER >= order (lmplz) and the Python module with MAX_ORDER >= order.

  python scripts/14_grapheme_lm.py --corpus /data/lm/si_corpus/corpus.txt --out /data/lm/si_graph8 --order 8
"""

import argparse
import collections
import json
import os
import subprocess
import time

from sinhala_asr.graphemes import WordCache

SEP = "|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--prune", default="0 0 1", help="lmplz --prune thresholds; last value repeats for higher orders")
    ap.add_argument("--max-words", type=int, default=0, help="stop after this many corpus words (0 = all)")
    ap.add_argument("--memory", default="80%", help="lmplz -S")
    ap.add_argument("--tmp", default="/tmp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t = time.time()
    cache, counts = WordCache(), collections.Counter()
    text = os.path.join(args.tmp, "graphemes.txt")
    n_words = n_lines = 0
    with open(args.corpus, encoding="utf-8") as f, open(text, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = [SEP if g == " " else g for g in cache.sentence(line)]
            counts.update(toks)
            out.write(" ".join(toks) + "\n")
            n_lines += 1
            n_words += line.count(" ") + 1
            if n_lines % 2_000_000 == 0:
                print(f"  {n_lines:,} sentences, {n_words:,} words ({time.time() - t:.0f}s)", flush=True)
            if args.max_words and n_words >= args.max_words:
                break
    n_tok = sum(counts.values())
    print(f"grapheme text: {n_lines:,} sentences, {n_words:,} words, {n_tok:,} tokens, {len(counts):,} types "
          f"({time.time() - t:.0f}s)", flush=True)

    arpa, binary = os.path.join(args.tmp, "lm.arpa"), os.path.join(args.out, "lm.bin")
    subprocess.run(f"lmplz -o {args.order} -S {args.memory} -T {args.tmp} --prune {args.prune} "
                   f"< {text} > {arpa}", shell=True, check=True)
    os.remove(text)
    subprocess.run(f"build_binary -q 8 -b 8 trie {arpa} {binary}", shell=True, check=True)
    os.remove(arpa)
    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8") as f:
        for g, _ in counts.most_common():
            f.write(g + "\n")
    with open(os.path.join(args.out, "lm_config.json"), "w") as f:
        json.dump({"kind": "grapheme", "order": args.order, "prune": args.prune, "separator": SEP,
                   "corpus": args.corpus, "sentences": n_lines, "words": n_words, "tokens": n_tok,
                   "types": len(counts), "max_words": args.max_words}, f, indent=1)
    print(f"lm: {binary} {os.path.getsize(binary) / 1e9:.2f} GB ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
