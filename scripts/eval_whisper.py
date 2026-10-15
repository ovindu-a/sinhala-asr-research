"""Baseline: score a Whisper checkpoint on a packed test set with exactly our normalization and metrics.

  python scripts/eval_whisper.py --model SinhaSpeech/whisper-small-sinhala-run12-best \
      --manifest /data/manifests/ss_v6/test.jsonl --packed /data/packed/ss_v6/test --out /runs/baselines/whisper_small_v6
"""

import argparse
import collections
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sinhala_asr.metrics import score
from sinhala_asr.text import normalize


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SinhaSpeech/whisper-small-sinhala-run12-best")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beams", type=int, default=1)
    args = ap.parse_args(argv)

    dev = "cuda"
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).to(dev).eval()
    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with open(args.packed + ".json") as f:
        off = json.load(f)["offsets"]
    pcm = np.fromfile(args.packed + ".pcm", dtype=np.int16)

    order = sorted(range(len(rows)), key=lambda i: rows[i]["dur"])
    hyps = [None] * len(rows)
    for s in tqdm(range(0, len(order), args.batch), mininterval=20):
        idx = order[s : s + args.batch]
        wavs = [pcm[off[i] : off[i + 1]].astype(np.float32) / 32767 for i in idx]
        feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            out = model.generate(feats.to(dev, torch.float16), language="si", task="transcribe",
                                 num_beams=args.beams, max_new_tokens=200)
        for i, t in zip(idx, proc.batch_decode(out, skip_special_tokens=True)):
            hyps[i] = normalize(t)

    refs = [r["text"] for r in rows]
    by_src = collections.defaultdict(lambda: ([], []))
    for r, h in zip(rows, hyps):
        by_src[r.get("src") or "all"][0].append(r["text"])
        by_src[r.get("src") or "all"][1].append(h)
    res = {"model": args.model, "overall": score(refs, hyps),
           "by_source": {s: score(a, b) for s, (a, b) in sorted(by_src.items())}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    with open(args.out + ".hyps.jsonl", "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r["id"], "src": r.get("src"), "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
    for s, m in [("overall", res["overall"])] + list(res["by_source"].items()):
        print(f"{s:10s} WER {m['wer']:.2%}  WER_seg {m['wer_seg']:.2%}  CER {m['cer']:.2%}  "
              f"CER_nospace {m['cer_nospace']:.2%}  ({m['utts']} utts)")


if __name__ == "__main__":
    main()
