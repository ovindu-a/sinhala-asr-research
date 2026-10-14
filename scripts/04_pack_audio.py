"""Pack a manifest's audio into one int16 array + offsets, so training reads one file instead of thousands.

Reading ~140 small FLACs per step from a network volume starved the GPU (5 s/step). The packed split is loaded
into RAM by train.py (--packed-dir); 20 h of 16 kHz int16 is ~2.3 GB.

usage: python scripts/04_pack_audio.py --manifest /data/manifests/x/train.jsonl --out /data/packed/x/train
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import soxr
from tqdm import tqdm

SR = 16_000


def load(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR)
    return (np.clip(wav, -1, 1) * 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="prefix; writes <out>.npy and <out>.json")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with ThreadPoolExecutor(args.threads) as ex:
        wavs = list(tqdm(ex.map(load, (r["audio"] for r in rows)), total=len(rows), mininterval=10))
    lens = np.array([len(w) for w in wavs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out + ".npy", np.concatenate(wavs))
    with open(args.out + ".json", "w") as f:
        json.dump({"ids": [r["id"] for r in rows], "offsets": offsets.tolist()}, f)
    print(f"packed {len(rows):,} utts, {offsets[-1] / SR / 3600:.2f} h, {offsets[-1] * 2 / 1e9:.2f} GB -> {args.out}.npy")


if __name__ == "__main__":
    main()
