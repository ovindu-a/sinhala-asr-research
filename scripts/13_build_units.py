"""Build grapheme or grapheme-BPE (gbpe) CTC units from 12_grapheme_inventory.py's output, and report on them.

  grapheme  units = blank, unk, "▁" (space), then the alphabet           -> <out>/units.json
  gbpe      SentencePiece BPE over words.tsv (word<TAB>count, i.e. the whole corpus) with every grapheme replaced by
            one private-use symbol, so merges never split a grapheme. --vocab counts SentencePiece's pieces including
            its <unk>; our units add the CTC blank (len(units) = vocab + 1).          -> <out>/units.json, sp.model

The report (<out>/report.json, also printed) is the go/no-go check before any GPU time:
  per manifest: units per word, share of single-unit words, <unk> rate, round-trip failures (must be 0 for rows
  without <unk>), targets per second (CTC needs fewer targets than encoder frames)
  on the first manifest (speech training text): how many units occur 0 / < 20 times
  --compare-gemma: the same numbers for Gemma-token units built from the same training text

  python scripts/13_build_units.py --kind gbpe --vocab 3000 --inventory /data/units --out /data/units/gbpe3000 \
      --manifests /data/manifests/v6_plus_slr52/train.jsonl /data/manifests/ss_v6/test.jsonl --compare-gemma
"""

import argparse
import collections
import json
import os
import tempfile

import numpy as np

from sinhala_asr.graphemes import WordCache
from sinhala_asr.units import MODEL_ID, UNK, GraphemeUnits, PieceUnits, Units, load_units

RARE = 20


def train_gbpe(inv_dir: str, alphabet: list[str], vocab: int, out: str) -> PieceUnits:
    import sentencepiece as spm

    to_sym = {g: chr(0xE000 + i) for i, g in enumerate(alphabet)}
    cache = WordCache()
    with tempfile.TemporaryDirectory() as tmp:
        tsv = os.path.join(tmp, "words.sym.tsv")
        n = 0
        with open(os.path.join(inv_dir, "words.tsv"), encoding="utf-8") as f, open(tsv, "w", encoding="utf-8") as o:
            for line in f:
                w, c = line.rstrip("\n").split("\t")
                o.write("".join(to_sym[g] for g in cache.word(w)) + f"\t{c}\n")
                n += 1
            for s in to_sym.values():  # every alphabet grapheme must be in SentencePiece's alphabet
                o.write(f"{s}\t1\n")
        print(f"training SentencePiece BPE (vocab {vocab}) on {n:,} distinct words", flush=True)
        spm.SentencePieceTrainer.train(
            input=tsv, input_format="tsv", model_prefix=os.path.join(out, "sp"), model_type="bpe",
            vocab_size=vocab, character_coverage=1.0, byte_fallback=False, unk_id=0, bos_id=-1, eos_id=-1,
            pad_id=-1, normalization_rule_name="identity", remove_extra_whitespaces=True, add_dummy_prefix=True,
            split_by_unicode_script=False, split_by_number=False, split_by_whitespace=True, split_digits=False,
            max_sentencepiece_length=16, num_threads=os.cpu_count(), minloglevel=1)
    with open(os.path.join(out, "sp.model"), "rb") as f:
        return PieceUnits(f.read(), alphabet)


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [r["text"] for r in rows], [r.get("dur") for r in rows]


def stats(units, manifests: list[str]) -> dict:
    cache = WordCache()
    res, usage = {}, collections.Counter()
    for k, path in enumerate(manifests):
        texts, durs = read_texts(path)
        n_words = n_units = n_space = single = unk = unk_rows = bad = n_g = 0
        rate, word_len = [], {}
        for t, d in zip(texts, durs):
            ids = units.encode(t)
            words = t.split()
            n_words += len(words)
            n_units += len(ids)
            n_space += sum(units.pieces[i] == "▁" for i in ids)
            n_g += sum(len(cache.word(w)) for w in words)
            for w in words:
                if w not in word_len:
                    word_len[w] = len(units.encode(w))
                single += word_len[w] == 1
            u = ids.count(UNK)
            unk += u
            if u:
                unk_rows += 1
            elif units.decode(ids) != t:
                bad += 1
            if d:
                rate.append(len(ids) / d)
            if k == 0:
                usage.update(ids)
        r = np.array(rate) if rate else np.zeros(1)
        res[os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)] = {
            "rows": len(texts), "words": n_words, "units": n_units, "units_per_word": n_units / max(1, n_words),
            "graphemes_per_unit_excl_space": n_g / max(1, n_units - n_space),
            "single_unit_words": single / max(1, n_words), "unk_rate": unk / max(1, n_units),
            "rows_with_unk": unk_rows, "roundtrip_failures": bad,
            "targets_per_s": {"mean": float(r.mean()), "p99": float(np.percentile(r, 99)), "max": float(r.max())},
        }
    real = range(2, len(units))
    # merged = units spanning 2+ graphemes (the BPE merges); single graphemes are shared with the grapheme units
    merged = [u for u in real if len(cache.word(units.pieces[u].replace("▁", ""))) >= 2]
    res["vocab"] = {
        "units": len(units),
        "unused_in_first_manifest": sum(usage[u] == 0 for u in real),
        f"rare_lt{RARE}_in_first_manifest": sum(usage[u] < RARE for u in real),
        f"rare_lt{RARE}_share": sum(usage[u] < RARE for u in real) / len(real),
        "merged_units": len(merged),
        f"merged_rare_lt{RARE}_share": sum(usage[u] < RARE for u in merged) / max(1, len(merged)),
    }
    return res


def show(name, st):
    print(f"\n== {name}: {st['vocab']}")
    for k, v in st.items():
        if k != "vocab":
            tp = v["targets_per_s"]
            print(f"  {k}: units/word {v['units_per_word']:.2f}, graphemes/unit {v['graphemes_per_unit_excl_space']:.2f}, "
                  f"single-unit words {v['single_unit_words']:.1%}, unk {v['unk_rate']:.3%} ({v['rows_with_unk']} rows), "
                  f"round-trip failures {v['roundtrip_failures']}, targets/s mean {tp['mean']:.1f} p99 {tp['p99']:.1f} "
                  f"max {tp['max']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["grapheme", "gbpe"])
    ap.add_argument("--inventory", required=True, help="dir with graphemes.json + words.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=3000)
    ap.add_argument("--manifests", nargs="+", required=True, help="first = speech training text (unit usage)")
    ap.add_argument("--compare-gemma", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.inventory, "graphemes.json"), encoding="utf-8") as f:
        alphabet = json.load(f)["alphabet"]

    units = GraphemeUnits(alphabet) if args.kind == "grapheme" else train_gbpe(args.inventory, alphabet, args.vocab, args.out)
    path = os.path.join(args.out, "units.json")
    units.save(path)
    units = load_units(path)  # what training will see
    print(f"{args.kind}: {len(units):,} units -> {path}")
    lens = collections.Counter(len(WordCache().word(p.replace("▁", ""))) for p in units.pieces[2:] if p)
    print("unit length in graphemes (excl. ▁):", dict(sorted(lens.items())))

    report = {"kind": args.kind, "vocab_arg": args.vocab, "units": len(units),
              "unit_len_hist": dict(sorted(lens.items())), "stats": stats(units, args.manifests)}
    show(args.kind, report["stats"])
    if args.compare_gemma:
        gem = Units.build(MODEL_ID, read_texts(args.manifests[0])[0])
        report["gemma"] = stats(gem, args.manifests)
        show("gemma (same training text)", report["gemma"])
    v = report["stats"]["vocab"]
    print(f"\nunits seen < {RARE} times in the speech training text: {v[f'rare_lt{RARE}_share']:.1%} of all units")
    if args.kind == "gbpe":
        rare = v[f"merged_rare_lt{RARE}_share"]
        print(f"merged pieces (2+ graphemes) seen < {RARE} times: {rare:.1%} of {v['merged_units']} "
              f"-> {'OK' if rare <= 0.20 else 'over 20%: use a smaller --vocab (2000)'}")
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
