"""Grapheme alphabet + word counts for the grapheme / grapheme-BPE units, from the KenLM text corpus.

The corpus is the exact text the word 4-gram KenLM was built from (07_build_lm.py --corpus-only: Wikipedia si +
FineWeb-2 sin_Sinh + v6/SLR-52 training transcripts, dev/test sentences removed), already normalized.

Alphabet = every *valid* grapheme (sinhala_asr.graphemes.is_valid) that occurs in the speech training transcripts or
at least --min-count times in the corpus, plus a-z and 0-9. Invalid clusters (typos like දිි, ලේ්, stray marks) are
left out: in training they become <unk>, and corpus words containing them are not used for the tokenizer.

writes <out>/graphemes.json  {alphabet, corpus_counts, speech_counts, stats}
       <out>/words.tsv       word<TAB>count for every corpus word seen >= --min-word-count times whose graphemes are
                             all in the alphabet (SentencePiece input for 13_build_units.py)

  python scripts/12_grapheme_inventory.py --corpus /data/lm/si_corpus/corpus.txt \
      --train-manifests /data/manifests/v6_plus_slr52/train.jsonl --out /data/units

C redo (tokenizer fitted to speech, see GRAPHEME_PLAN.md section 16): --speech-alphabet keeps only graphemes seen in
the speech transcripts (+ a-z, 0-9), and --transcript-weight W adds the transcripts' word counts W more times to
words.tsv, so BPE merges follow spoken words instead of web text:
  ... --out /data/units_speech --speech-alphabet --transcript-weight 50
"""

import argparse
import collections
import json
import os
import time

from sinhala_asr.graphemes import WordCache, is_valid

LATIN_DIGITS = list("abcdefghijklmnopqrstuvwxyz0123456789")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--train-manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-count", type=int, default=50, help="corpus count for a grapheme not seen in speech text")
    ap.add_argument("--min-word-count", type=int, default=2, help="drop corpus hapaxes (mostly typos) from words.tsv")
    ap.add_argument("--speech-alphabet", action="store_true", help="alphabet = graphemes of the speech transcripts only")
    ap.add_argument("--transcript-weight", type=int, default=0, help="add transcript word counts this many times")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = WordCache()

    t = time.time()
    words = collections.Counter()
    with open(args.corpus, encoding="utf-8") as f:
        for k, line in enumerate(f):
            words.update(line.split())
            if k % 2_000_000 == 0:
                print(f"  {k:,} lines, {len(words):,} distinct words ({time.time() - t:.0f}s)", flush=True)
    n_tokens = sum(words.values())
    print(f"corpus: {n_tokens:,} words, {len(words):,} distinct ({time.time() - t:.0f}s)")

    corpus_g = collections.Counter()
    for w, c in words.items():
        for g in cache.word(w):
            corpus_g[g] += c
    speech_g, speech_w = collections.Counter(), collections.Counter()
    for path in args.train_manifests:
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                speech_g.update(g for g in cache.sentence(text) if g != " ")
                speech_w.update(text.split())
    if args.transcript_weight:  # the corpus already holds the transcripts once
        for w, c in speech_w.items():
            words[w] += c * args.transcript_weight
        print(f"transcripts x{args.transcript_weight}: {sum(speech_w.values()):,} words, {len(speech_w):,} distinct")
    print(f"graphemes: corpus {len(corpus_g):,} distinct, speech transcripts {len(speech_g):,} distinct "
          f"({time.time() - t:.0f}s)")

    keep = set() if args.speech_alphabet else {g for g in corpus_g if is_valid(g) and corpus_g[g] >= args.min_count}
    keep |= {g for g in speech_g if is_valid(g)}
    keep |= set(LATIN_DIGITS)
    alphabet = sorted(keep, key=lambda g: (-(corpus_g[g] + speech_g[g]), g))

    invalid = collections.Counter({g: c for g, c in corpus_g.items() if not is_valid(g)})
    rare_valid = collections.Counter({g: c for g, c in corpus_g.items() if is_valid(g) and g not in keep})
    tot = sum(corpus_g.values())
    stats = {
        "corpus_words": n_tokens, "corpus_distinct_words": len(words),
        "corpus_distinct_graphemes": len(corpus_g), "speech_distinct_graphemes": len(speech_g),
        "alphabet": len(alphabet), "min_count": args.min_count, "speech_alphabet": args.speech_alphabet,
        "transcript_weight": args.transcript_weight,
        "corpus_grapheme_tokens": tot,
        "corpus_tokens_covered": sum(corpus_g[g] for g in keep) / tot,
        "invalid_distinct": len(invalid), "invalid_token_share": sum(invalid.values()) / tot,
        "invalid_top": invalid.most_common(30),
        "rare_valid_distinct": len(rare_valid), "rare_valid_token_share": sum(rare_valid.values()) / tot,
        "speech_invalid": sorted(g for g in speech_g if not is_valid(g)),
    }

    kept_words = dropped_tok = 0
    with open(os.path.join(args.out, "words.tsv"), "w", encoding="utf-8") as f:
        for w, c in words.most_common():
            if c < args.min_word_count:
                break
            if all(g in keep for g in cache.word(w)):
                f.write(f"{w}\t{c}\n")
                kept_words += 1
            else:
                dropped_tok += c
    stats |= {"words_tsv": kept_words, "min_word_count": args.min_word_count,
              "words_tsv_token_share": (sum(c for c in words.values() if c >= args.min_word_count) - dropped_tok)
              / sum(words.values())}  # incl. transcript weighting

    with open(os.path.join(args.out, "graphemes.json"), "w", encoding="utf-8") as f:
        json.dump({"alphabet": alphabet, "corpus_counts": {g: corpus_g[g] for g in alphabet},
                   "speech_counts": {g: speech_g[g] for g in alphabet}, "stats": stats}, f, ensure_ascii=False,
                  indent=1)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"done in {time.time() - t:.0f}s -> {args.out}/graphemes.json, words.tsv")


if __name__ == "__main__":
    main()
