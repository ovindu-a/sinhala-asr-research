"""Step 1a: how well does the Gemma tokenizer cover Sinhala?

Reports, on the OpenSLR-52 transcripts:
  - tokens per word (fertility), and the share of words that are a single token
  - whether pieces are real Sinhala subwords or byte-fallback (<0xE0> ...)
  - how many distinct Gemma ids the transcripts use (= the CTC unit inventory, cf. QUAN's 14,917)
  - word-level alternative: vocabulary size, coverage and OOV rate on held-out speakers
  - the effect of ZWJ/ZWNJ (common in Sinhala conjuncts) on tokenization

usage: uv run scripts/01_tokenizer_check.py [--tsv data/openslr52/utt_spk_text.tsv]
"""

import argparse
import collections
import random
import re
import unicodedata

from transformers import AutoTokenizer

SINHALA = re.compile(r"[඀-෿]")
ZW = re.compile("[‌‍]")
BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")


def load(tsv):
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[2].strip():
                rows.append((parts[0], parts[1], unicodedata.normalize("NFC", parts[2].strip())))
    return rows


def pct(a, b):
    return f"{100 * a / max(b, 1):.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/openslr52/utt_spk_text.tsv")
    ap.add_argument("--model", default="google/embeddinggemma-2")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = load(args.tsv)
    print(f"{len(rows):,} utterances, {len({r[1] for r in rows}):,} speakers\n")

    # --- Gemma vocab: how many pieces contain Sinhala script at all
    vocab = tok.get_vocab()
    si_pieces = [p for p in vocab if SINHALA.search(tok.convert_tokens_to_string([p]))]
    lens = collections.Counter(len(SINHALA.findall(tok.convert_tokens_to_string([p]))) for p in si_pieces)
    print(f"Gemma vocab {len(vocab):,} pieces; {len(si_pieces):,} contain Sinhala characters")
    print("  Sinhala chars per piece:", dict(sorted(lens.items())[:10]), "\n")

    # --- per-word fertility
    words = collections.Counter(w for _, _, t in rows for w in t.split())
    n_tokens_by_word = {}
    byte_words = 0
    for w in words:
        pieces = tok.tokenize(w)
        n_tokens_by_word[w] = len(pieces)
        if any(BYTE_PIECE.match(p) for p in pieces):
            byte_words += 1
    total = sum(words.values())
    fert = sum(n_tokens_by_word[w] * c for w, c in words.items()) / total
    single = sum(c for w, c in words.items() if n_tokens_by_word[w] == 1)
    dist = collections.Counter()
    for w, c in words.items():
        dist[min(n_tokens_by_word[w], 6)] += c
    print(f"words: {total:,} running, {len(words):,} distinct")
    print(f"tokens/word (running): {fert:.2f}   single-token words: {pct(single, total)}")
    print("  distribution (running words):", {(f"{k}" if k < 6 else "6+"): pct(v, total) for k, v in sorted(dist.items())})
    print(f"  distinct words needing byte-fallback pieces: {byte_words:,} ({pct(byte_words, len(words))})\n")

    # --- sentence-level tokenization (what the CTC targets would be) and unit inventory
    ids = collections.Counter()
    tok_per_utt = []
    for _, _, t in rows:
        e = tok(t, add_special_tokens=False)["input_ids"]
        ids.update(e)
        tok_per_utt.append(len(e))
    assert tok.decode(tok(rows[0][2], add_special_tokens=False)["input_ids"]) == rows[0][2], "round-trip failed"
    cum, need = 0, None
    for i, (_, c) in enumerate(ids.most_common(), 1):
        cum += c
        if need is None and cum / sum(ids.values()) >= 0.99:
            need = i
    print(f"distinct Gemma ids used by transcripts: {len(ids):,} (QUAN Vietnamese used 14,917)")
    print(f"  ids covering 99% of tokens: {need:,}; ids seen only once: {sum(1 for c in ids.values() if c == 1):,}")
    print(f"  mean tokens/utterance: {sum(tok_per_utt) / len(tok_per_utt):.1f}")
    print("  top pieces:", [tok.convert_ids_to_tokens(i) for i, _ in ids.most_common(25)], "\n")

    # --- CTC feasibility: encoder emits 25 frames/s; CTC needs frames >= targets (+ repeats)
    # OpenSLR has no durations in the tsv; ~4.3 s/utt is typical for this corpus (224 h / 185k utts).
    print(f"CTC check: ~25 frames/s; {sum(tok_per_utt) / len(tok_per_utt) / 4.3:.1f} tokens/s at ~4.3 s/utt "
          "(needs to stay well under 25)\n")

    # --- word-level units: speaker-held-out OOV
    spk = sorted({r[1] for r in rows})
    random.Random(0).shuffle(spk)
    test_spk = set(spk[: len(spk) // 10])
    train_w = collections.Counter(w for _, s, t in rows if s not in test_spk for w in t.split())
    test_w = collections.Counter(w for _, s, t in rows if s in test_spk for w in t.split())
    oov = sum(c for w, c in test_w.items() if w not in train_w)
    for k in (20_000, 50_000, 100_000):
        top = {w for w, _ in train_w.most_common(k)}
        o = sum(c for w, c in test_w.items() if w not in top)
        print(f"word units, top {k:>7,}: held-out-speaker OOV {pct(o, sum(test_w.values()))}")
    print(f"word units, all {len(train_w):>8,}: held-out-speaker OOV {pct(oov, sum(test_w.values()))}")
    print("  (OpenSLR prompts repeat across speakers, so this OOV is optimistic for real speech)\n")

    # --- ZWJ / ZWNJ
    zw_utts = [t for _, _, t in rows if ZW.search(t)]
    diff = sum(1 for t in zw_utts if tok(t)["input_ids"] != tok(ZW.sub("", t))["input_ids"])
    print(f"utterances containing ZWJ/ZWNJ: {pct(len(zw_utts), len(rows))}; tokenization changes if stripped: {pct(diff, len(zw_utts))}")
    if zw_utts:
        t = zw_utts[0]
        print("  example:", t)
        print("    with ZWJ:   ", tok.tokenize(t))
        print("    without ZWJ:", tok.tokenize(ZW.sub("", t)))

    # --- a few examples
    print("\nexamples:")
    for _, _, t in random.Random(1).sample(rows, 5):
        print(" ", t, "->", tok.tokenize(t))


if __name__ == "__main__":
    main()
