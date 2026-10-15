"""Find the OpenSLR-52 utterances that SinhaSpeech v6 left out, and add them to training without leaking test speakers.

v6 is ~97% SLR-52 but has no file/speaker ids. 27% of SLR-52 utterances have a sentence only one speaker read, so
v6 val/test clips with such a sentence identify their SLR-52 speaker. Extra training data = SLR-52 utterances
  - whose speaker is NOT a v6 val/test speaker, and
  - whose (normalized) sentence does not occur anywhere in v6 (so nothing in v6 is duplicated or leaked).
Then the extra audio is packed (from the HF mirror SPEAK-ASR/openslr-sinhala-asr) and merged after v6 train:
  /data/packed/<out>/train.{pcm,json} + /data/manifests/<out>/train.jsonl  (= v6 train + extra)

usage: python scripts/06_slr52_extra.py --v6-train-manifest /data/manifests/ss_v6/train.jsonl \
          --v6-packed /data/packed/ss_v6/train --out v6_plus_slr52 [--dry-run]
"""

import argparse
import collections
import io
import json
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from sinhala_asr.text import has_digits, normalize

V6 = "SinhaSpeech/sinhala-asr-data"
SLR = "SPEAK-ASR/openslr-sinhala-asr"
SR = 16_000


def read_cols(repo, path, cols):
    # a plain download is far faster than HfFileSystem range reads on Modal
    local = hf_hub_download(repo, path, repo_type="dataset", local_dir="/tmp/hf_cols")
    rows = pq.read_table(local, columns=cols).to_pylist()
    os.remove(local)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6-train-manifest", required=True)
    ap.add_argument("--v6-packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--slr-tsv", default="/data/openslr52/utt_spk_text.tsv")
    args = ap.parse_args()

    # v6 texts (all splits) and source tags
    v6 = {}
    for split in ("train", "validation", "test"):
        v6[split] = read_cols(V6, f"data/stratified_v6/{split}.parquet", ["text", "source_dataset"])
        print(f"v6 {split}: {len(v6[split]):,} rows, {collections.Counter(r['source_dataset'] for r in v6[split])}")
    v6_texts = {normalize(r["text"]) for s in v6.values() for r in s}

    # SLR-52 speaker/text index (official tsv, uploaded to the volume); durations come later from the audio
    slr = []
    with open(args.slr_tsv, encoding="utf-8") as f:
        for line in f:
            u, s, t = line.rstrip("\n").split("\t")
            slr.append({"file_id": u, "speaker_id": s, "text": t, "norm": normalize(t)})
    by_text = collections.defaultdict(list)
    for r in slr:
        by_text[r["norm"]].append(r)
    print(f"SLR-52: {len(slr):,} utts, {len({r['speaker_id'] for r in slr})} speakers")

    def speakers_of(split):
        spk, hits = collections.Counter(), 0
        for r in v6[split]:
            if r["source_dataset"] != "openslr":
                continue
            c = by_text.get(normalize(r["text"]), [])
            if len(c) == 1:
                spk[c[0]["speaker_id"]] += 1
                hits += 1
        return spk, hits

    held, report = set(), {}
    for split in ("validation", "test", "train"):
        spk, hits = speakers_of(split)
        report[split] = spk
        print(f"v6 {split}: {hits:,} openslr clips matched by unique sentence -> {len(spk)} speakers")
    held = set(report["validation"]) | set(report["test"])
    overlap = held & set(report["train"])
    # a speaker seen once could be a sentence collision; require >=3 matches to call it a train speaker
    strong_overlap = {s for s in overlap if report["train"][s] >= 3 and (report["test"][s] + report["validation"][s]) >= 3}
    print(f"held-out (val+test) speakers: {len(held)}; also matched in v6 train: {len(overlap)} "
          f"(>=3 matches on both sides: {len(strong_overlap)})")

    extra = [r for r in slr if r["speaker_id"] not in held and r["norm"] not in v6_texts and r["norm"]
             and not has_digits(r["text"])]
    excluded_spk = sum(1 for r in slr if r["speaker_id"] in held)
    print(f"SLR-52 utts from held-out speakers (excluded): {excluded_spk:,}")
    print(f"EXTRA: {len(extra):,} utts (~{len(extra) * 4.3 / 3600:.0f} h at SLR-52's ~4.3 s/utt), "
          f"{len({r['speaker_id'] for r in extra})} speakers")
    if args.dry_run:
        return

    # pack extra audio from the HF mirror (every file may hold some), then merge after v6 train
    files = sorted(f for f in HfApi().list_repo_files(SLR, repo_type="dataset") if f.endswith(".parquet"))
    want = {fn: None for fn in files}
    meta = {r["file_id"]: r for r in extra}
    tmp = "/tmp/extra.pcm"
    ids, offs, pos, rows, seen = [], [0], 0, [], set()
    with open(tmp, "wb") as pcm:
        for fn in tqdm(sorted(want), desc="pack extra"):
            path = hf_hub_download(SLR, fn, repo_type="dataset", local_dir="/tmp/hf_slr")
            for r in pq.read_table(path, columns=["file_id", "audio"]).to_pylist():
                if r["file_id"] not in meta or "slr52_" + r["file_id"] in seen:
                    continue
                seen.add("slr52_" + r["file_id"])
                w, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                assert sr == SR
                w = (np.clip(w.mean(1), -1, 1) * 32767).astype(np.int16)
                if len(w) / SR > args.max_dur:
                    continue
                pcm.write(w.tobytes())
                pos += len(w)
                offs.append(pos)
                m = meta[r["file_id"]]
                uid = "slr52_" + r["file_id"]
                ids.append(uid)
                rows.append({"id": uid, "text": m["norm"], "raw_text": m["text"], "dur": round(len(w) / SR, 3),
                             "src": "openslr_extra", "spk": m["speaker_id"]})
            os.remove(path)

    with open(args.v6_packed + ".json") as f:
        base = json.load(f)
    with open(args.v6_train_manifest, encoding="utf-8") as f:
        base_rows = [json.loads(line) for line in f]
    out_p, out_m = f"/data/packed/{args.out}", f"/data/manifests/{args.out}"
    os.makedirs(out_p, exist_ok=True)
    os.makedirs(out_m, exist_ok=True)
    with open(f"{out_p}/train.pcm", "wb") as dst:
        for src in (args.v6_packed + ".pcm", tmp):
            with open(src, "rb") as s:
                shutil.copyfileobj(s, dst, 64 << 20)
    shift = base["offsets"][-1]
    with open(f"{out_p}/train.json", "w") as f:
        json.dump({"ids": base["ids"] + ids, "offsets": base["offsets"] + [shift + o for o in offs[1:]]}, f)
    with open(f"{out_m}/train.jsonl", "w", encoding="utf-8") as f:
        for r in base_rows + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name in ("dev2k", "test"):  # same eval sets
        for ext in (".pcm", ".json"):
            dst = f"{out_p}/{name}{ext}"
            if not os.path.exists(dst):
                os.symlink(f"/data/packed/ss_v6/{name}{ext}", dst)
        if not os.path.exists(f"{out_m}/{name}.jsonl"):
            shutil.copy(f"/data/manifests/ss_v6/{name}.jsonl", f"{out_m}/{name}.jsonl")
    h = (shift + pos) / SR / 3600
    print(f"wrote {args.out}: {len(base_rows) + len(rows):,} utts, {h:.1f} h (v6 {shift / SR / 3600:.1f} h + extra {pos / SR / 3600:.1f} h)")


if __name__ == "__main__":
    main()
