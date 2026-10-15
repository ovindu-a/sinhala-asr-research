"""Pack an HF parquet split straight into one raw int16 PCM file + manifest (no per-utterance files).

Writes:
  <out>.pcm    int16 mono 16 kHz, all utterances back to back
  <out>.json   {"ids": [...], "offsets": [...]}  (sample offsets, len = n + 1)
  <manifest>   jsonl rows {id, text, raw_text, dur, src, spk}

Defaults are for SinhaSpeech/sinhala-asr-data stratified_v6 (columns audio/text/source_dataset; speaker-disjoint
splits). --drop-digits drops utterances whose reference has digits (train only: numbers are not verbalized yet;
keep them in test so scores stay comparable).

usage: python scripts/05_pack_parquet.py --file data/stratified_v6/train.parquet --out /data/packed/ss_v6/train \
          --manifest /data/manifests/ss_v6/train.jsonl --drop-digits
"""

import argparse
import io
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from sinhala_asr.text import has_digits, normalize

SR = 16_000


def to_pcm16(b: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR)
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SinhaSpeech/sinhala-asr-data")
    ap.add_argument("--file", required=True, help="parquet path inside the dataset repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--src-col", default="source_dataset")
    ap.add_argument("--drop-digits", action="store_true")
    ap.add_argument("--subsample", type=int, default=0, help="keep a random subset of N rows (e.g. a dev subset)")
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--cache", default="/tmp/hf_parquet")
    args = ap.parse_args()

    path = hf_hub_download(args.repo, args.file, repo_type="dataset", local_dir=args.cache)
    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    keep_idx = None
    if args.subsample:
        keep_idx = set(random.Random(0).sample(range(n_total), min(args.subsample, n_total)))
    split = os.path.splitext(os.path.basename(args.file))[0]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    ids, offsets, pos, i = [], [0], 0, 0
    dropped = {"digits": 0, "empty": 0, "dur": 0, "decode": 0}
    cols = [args.audio_col, args.text_col] + ([args.src_col] if args.src_col else [])
    with open(args.out + ".pcm", "wb") as pcm, open(args.manifest, "w", encoding="utf-8") as man:
        bar = tqdm(total=n_total, mininterval=15, desc=split)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=cols).to_pylist()
            for r in t:
                k, i = i, i + 1
                bar.update(1)
                if keep_idx is not None and k not in keep_idx:
                    continue
                raw = r[args.text_col] or ""
                if args.drop_digits and has_digits(raw):
                    dropped["digits"] += 1
                    continue
                text = normalize(raw)
                if not text:
                    dropped["empty"] += 1
                    continue
                a = r[args.audio_col]
                try:
                    w = to_pcm16(a["bytes"] if isinstance(a, dict) else a)
                except Exception:
                    dropped["decode"] += 1
                    continue
                dur = len(w) / SR
                if not (args.min_dur <= dur <= args.max_dur):
                    dropped["dur"] += 1
                    continue
                uid = f"{split}_{k}"
                pcm.write(w.tobytes())
                pos += len(w)
                offsets.append(pos)
                ids.append(uid)
                man.write(json.dumps({"id": uid, "text": text, "raw_text": raw, "dur": round(dur, 3),
                                      "src": r.get(args.src_col) if args.src_col else None, "spk": None},
                                     ensure_ascii=False) + "\n")
        bar.close()
    with open(args.out + ".json", "w") as f:
        json.dump({"ids": ids, "offsets": offsets}, f)
    print(f"{split}: kept {len(ids):,} / {n_total:,} rows, {pos / SR / 3600:.2f} h, {pos * 2 / 1e9:.2f} GB; dropped {dropped}")


if __name__ == "__main__":
    main()
