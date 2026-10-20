"""Score a checkpoint on a packed test set: overall and per source, with WER, spacing-agnostic WER_seg, CER.

  python scripts/evaluate.py --ckpt /runs/x/best --manifest /data/manifests/ss_v6/test.jsonl \
      --packed /data/packed/ss_v6/test --out /runs/x/eval_ss_v6_test
writes <out>.json (metrics) and <out>.hyps.jsonl (ref/hyp per utterance)
"""

import argparse
import collections
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import AudioSet, DurationBatches, collate, evaluate, load_weights, read_jsonl  # noqa: E402

from sinhala_asr.metrics import score  # noqa: E402
from sinhala_asr.model import Config, CTCModel, load_full  # noqa: E402
from sinhala_asr.units import load_units  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--packed", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-seconds", type=float, default=1200)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device.type == "cuda"
    with open(os.path.join(args.ckpt, "config.json")) as f:
        cfg = Config(**{k: v for k, v in json.load(f).items() if k in Config.__dataclass_fields__})
    units = load_units(os.path.join(args.ckpt, "units.json"))
    model = CTCModel(cfg, load_full(cfg.encoder_id), units.init_ids, units.pieces)
    load_weights(model, args.ckpt)
    model = model.to(device).eval()

    rows = read_jsonl(args.manifest)
    ds = AudioSet(rows, units, args.packed)
    dl = DataLoader(ds, batch_sampler=DurationBatches([r["dur"] for r in rows], args.batch_seconds, shuffle=False),
                    collate_fn=collate, num_workers=args.workers)
    ev = evaluate(model, dl, units, device, amp, return_hyps=True)

    by_src = collections.defaultdict(lambda: ([], []))
    with open(args.out + ".hyps.jsonl", "w", encoding="utf-8") as f:
        for i, ref, hyp in ev["hyps"]:
            r = rows[i]
            by_src[r.get("src") or "all"][0].append(ref)
            by_src[r.get("src") or "all"][1].append(hyp)
            f.write(json.dumps({"id": r["id"], "src": r.get("src"), "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
    res = {"ckpt": args.ckpt, "manifest": args.manifest,
           "overall": {k: v for k, v in ev.items() if k not in ("examples", "hyps")},
           "by_source": {s: score(r, h) for s, (r, h) in sorted(by_src.items())}}
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    fmt = lambda m: (f"WER {m['wer']:.2%}  WER_seg {m['wer_seg']:.2%}  CER {m['cer']:.2%}  GER {m['ger']:.2%}  "  # noqa: E731
                     f"CER_nospace {m['cer_nospace']:.2%}  ({m['utts']} utts)")
    print("overall   ", fmt(res["overall"]))
    for s, m in res["by_source"].items():
        print(f"{s:10s}", fmt(m))
    return res


if __name__ == "__main__":
    main()
