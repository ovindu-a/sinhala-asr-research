"""Build OpenSLR-52 manifests from the Hugging Face parquet mirror (SPEAK-ASR/openslr-sinhala-asr).

openslr.org is slow (~0.6 MB/s); the HF mirror has the same audio + file_id/speaker_id/text/duration. Audio bytes
are written out as files so the manifests look exactly like scripts/03_prep_openslr.py's. The mirror's own
train/validation/test splits are ignored: speakers are re-split here (dev speaker-disjoint, train utterances that
share text with dev dropped, digits dropped).

usage: python scripts/03b_prep_openslr_hf.py --files 4 --audio-dir /data/openslr52_hf --out /data/manifests/x
"""

import argparse
import io
import json
import os
import random

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from sinhala_asr.text import has_digits, normalize

REPO = "SPEAK-ASR/openslr-sinhala-asr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=0, help="number of parquet files to use (0 = all)")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-speaker-frac", type=float, default=0.08)
    ap.add_argument("--dev-max-utts", type=int, default=2000)
    ap.add_argument("--train-hours", type=float, default=None)
    ap.add_argument("--max-dur", type=float, default=20.0)
    args = ap.parse_args()

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    random.Random(0).shuffle(files)
    if args.files:
        files = files[: args.files]
    os.makedirs(args.audio_dir, exist_ok=True)

    rows, dropped, seen = [], {"digits": 0, "empty": 0, "too_long": 0, "dup": 0}, set()
    for fn in files:
        path = hf_hub_download(REPO, fn, repo_type="dataset")
        table = pq.read_table(path)
        for r in tqdm(table.to_pylist(), desc=os.path.basename(fn), mininterval=10):
            u = r["file_id"]
            if u in seen:
                dropped["dup"] += 1
                continue
            seen.add(u)
            if has_digits(r["text"]):
                dropped["digits"] += 1
                continue
            text = normalize(r["text"])
            if not text:
                dropped["empty"] += 1
                continue
            a = r["audio"]
            out = os.path.join(args.audio_dir, u[:2], u + ".flac")
            if not os.path.exists(out):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                sf.write(out, wav, sr, format="FLAC")
            else:
                wav, sr = sf.read(out, dtype="float32")
            dur = len(wav) / sr
            if dur > args.max_dur:
                dropped["too_long"] += 1
                continue
            rows.append({"id": u, "audio": out, "text": text, "dur": round(dur, 3), "spk": r["speaker_id"], "src": "openslr52"})

    spk = sorted({r["spk"] for r in rows})
    random.Random(0).shuffle(spk)
    dev_spk = set(spk[: max(1, int(len(spk) * args.dev_speaker_frac))])
    dev = [r for r in rows if r["spk"] in dev_spk]
    random.Random(0).shuffle(dev)
    dev = dev[: args.dev_max_utts]
    dev_texts = {r["text"] for r in dev}
    train = [r for r in rows if r["spk"] not in dev_spk and r["text"] not in dev_texts]
    overlap = sum(1 for r in rows if r["spk"] not in dev_spk and r["text"] in dev_texts)
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
        print(f"{name}: {len(split):,} utts, {sum(r['dur'] for r in split) / 3600:.2f} h, {len({r['spk'] for r in split})} speakers")
    print("dropped:", dropped, "| train utts dropped for sharing text with dev:", overlap)


if __name__ == "__main__":
    main()
