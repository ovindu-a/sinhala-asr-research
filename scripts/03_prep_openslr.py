"""Build train/dev manifests from OpenSLR-52 (whatever shards are unzipped under --root).

Dev is speaker-disjoint from train, and every train utterance whose text also appears in dev is dropped
(OpenSLR prompts repeat across speakers, which otherwise inflates dev scores). Utterances with digits are
dropped until numbers are verbalized.

usage: uv run scripts/03_prep_openslr.py --root data/openslr52 --out data/manifests/openslr [--train-hours 20]
"""

import argparse
import glob
import json
import os
import random

import soundfile as sf
from tqdm import tqdm

from sinhala_asr.text import has_digits, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/openslr52")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--out", default="data/manifests/openslr")
    ap.add_argument("--dev-speaker-frac", type=float, default=0.08)
    ap.add_argument("--dev-max-utts", type=int, default=2000)
    ap.add_argument("--train-hours", type=float, default=None, help="cap train hours (random subset)")
    ap.add_argument("--max-dur", type=float, default=20.0)
    args = ap.parse_args()

    tsv = args.tsv or os.path.join(args.root, "utt_spk_text.tsv")
    meta = {}
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            u, s, t = line.rstrip("\n").split("\t")
            meta[u] = (s, t)

    rows, dropped = [], {"no_meta": 0, "digits": 0, "empty": 0, "too_long": 0}
    for p in tqdm(sorted(glob.glob(os.path.join(args.root, "**", "*.flac"), recursive=True)), desc="scan"):
        u = os.path.basename(p)[:-5]
        if u not in meta:
            dropped["no_meta"] += 1
            continue
        spk, raw = meta[u]
        if has_digits(raw):
            dropped["digits"] += 1
            continue
        text = normalize(raw)
        if not text:
            dropped["empty"] += 1
            continue
        dur = sf.info(p).duration
        if dur > args.max_dur:
            dropped["too_long"] += 1
            continue
        rows.append({"id": u, "audio": os.path.abspath(p), "text": text, "dur": round(dur, 3), "spk": spk, "src": "openslr52"})

    spk = sorted({r["spk"] for r in rows})
    random.Random(0).shuffle(spk)
    dev_spk = set(spk[: max(1, int(len(spk) * args.dev_speaker_frac))])
    dev = [r for r in rows if r["spk"] in dev_spk]
    random.Random(0).shuffle(dev)
    dev = dev[: args.dev_max_utts]
    dev_texts = {r["text"] for r in dev}
    train = [r for r in rows if r["spk"] not in dev_spk and r["text"] not in dev_texts]
    text_overlap_dropped = sum(1 for r in rows if r["spk"] not in dev_spk and r["text"] in dev_texts)
    if args.train_hours:
        random.Random(0).shuffle(train)
        keep, tot = [], 0.0
        for r in train:
            if tot >= args.train_hours * 3600:
                break
            keep.append(r)
            tot += r["dur"]
        train = keep

    os.makedirs(args.out, exist_ok=True)
    for name, split in (("train", train), ("dev", dev)):
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        h = sum(r["dur"] for r in split) / 3600
        print(f"{name}: {len(split):,} utts, {h:.2f} h, {len({r['spk'] for r in split})} speakers")
    print("dropped:", dropped, "| train utts dropped for sharing text with dev:", text_overlap_dropped)


if __name__ == "__main__":
    main()
