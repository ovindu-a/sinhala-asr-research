"""CTC + KenLM beam search (QUAN's KenLMDecoder) on a checkpoint, with alpha/beta tuned on dev.

Two phases so the GPU is used only briefly:
  dump:   GPU forward pass, keep the top-k log-probs per frame  -> <out>/<split>.logits.pt
  decode: CPU beam search over many processes, grid over alpha/beta on dev, best setting on test

  python scripts/08_lm_decode.py dump   --ckpt C --split dev2k --split test --manifests M --packed P --out O
  python scripts/08_lm_decode.py decode --ckpt C --lm L --out O --tune-split dev2k --eval-split test --manifests M
"""

import argparse
import collections
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinhala_asr.metrics import score  # noqa: E402
from sinhala_asr.units import load_units  # noqa: E402

TOPK = 16


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dump(args):
    from torch.utils.data import DataLoader
    from train import AudioSet, DurationBatches, collate, load_weights

    from sinhala_asr.model import Config, CTCModel, load_full

    with open(os.path.join(args.ckpt, "config.json")) as f:
        cfg = Config(**{k: v for k, v in json.load(f).items() if k in Config.__dataclass_fields__})
    units = load_units(os.path.join(args.ckpt, "units.json"))
    model = CTCModel(cfg, load_full(cfg.encoder_id), units.init_ids)
    load_weights(model, args.ckpt)
    model = model.cuda().eval()
    os.makedirs(args.out, exist_ok=True)
    for split in args.split:
        rows = read_jsonl(f"{args.manifests}/{split}.jsonl")
        ds = AudioSet(rows, units, f"{args.packed}/{split}")
        dl = DataLoader(ds, batch_sampler=DurationBatches([r["dur"] for r in rows], 1200, shuffle=False),
                        collate_fn=collate, num_workers=6)
        out = [None] * len(rows)
        with torch.no_grad():
            for wav, wl, _, _, idx in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logp, ol = model(wav.cuda(), wl.cuda())
                v, i = logp.float().topk(TOPK, dim=-1)
                for b, k in enumerate(idx):
                    n = int(ol[b])
                    out[k] = (v[b, :n].half().cpu(), i[b, :n].short().cpu())
        torch.save({"rows": rows, "topk": out, "vocab": len(units)}, os.path.join(args.out, f"{split}.logits.pt"))
        print(f"dumped {split}: {len(rows):,} utts")


_G = {}


def _init(lm, pieces, unigrams, alpha, beta, vocab):
    from quan_transcribe.lm import KenLMDecoder

    _G["dec"] = KenLMDecoder(lm, pieces, unigrams, alpha=alpha, beta=beta)
    _G["V"] = vocab
    # units with no text (<unk>, byte pieces) are skipped by QUAN's beam search; if one is the only candidate in a
    # frame, every beam dies (IndexError). They add nothing to the transcript, so mask them out.
    _G["empty"] = np.array([k for k, p in enumerate(pieces) if k != 0 and not p])


def _decode_one(item):
    v, i = item
    lp = np.full((v.shape[0], _G["V"]), -1e4, dtype=np.float32)
    np.put_along_axis(lp, i.long().numpy(), v.float().numpy(), axis=1)
    if len(_G["empty"]):
        lp[:, _G["empty"]] = -1e4
    try:
        return _G["dec"].decode(lp, beam_width=_G.get("beam", 32))
    except IndexError:  # fall back to greedy CTC for this utterance
        best = lp.argmax(-1)
        keep = np.r_[True, best[1:] != best[:-1]]
        return [int(u) for u in best[keep] if u != 0]


def run(data, units, lm, pieces, unigrams, alpha, beta, procs, limit=None):
    items = data["topk"][:limit] if limit else data["topk"]
    with Pool(procs, initializer=_init, initargs=(lm, pieces, unigrams, alpha, beta, data["vocab"])) as pool:
        seqs = pool.map(_decode_one, items, chunksize=16)
    return [units.decode(s) for s in seqs]


def decode(args):
    units = load_units(os.path.join(args.ckpt, "units.json"))
    pieces = units.pieces  # "" for blank/unk/byte pieces; "▁" starts a word (grapheme units: the space unit)
    with open(os.path.join(args.lm, "unigrams.txt"), encoding="utf-8") as f:
        unigrams = [w.strip() for w in f if w.strip()]
    lm = os.path.join(args.lm, "lm.bin")
    procs = os.cpu_count()
    tune = torch.load(os.path.join(args.out, f"{args.tune_split}.logits.pt"), weights_only=False)
    refs = [r["text"] for r in tune["rows"]]
    lim = args.tune_limit
    grid = []
    for a, b in itertools.product(args.alphas, args.betas):
        hyps = run(tune, units, lm, pieces, unigrams, a, b, procs, lim)
        m = score(refs[: len(hyps)], hyps)
        grid.append({"alpha": a, "beta": b} | m)
        print(f"alpha {a:.2f} beta {b:.1f}: WER {m['wer']:.2%} WER_seg {m['wer_seg']:.2%} CER {m['cer']:.2%} "
              f"GER {m['ger']:.2%}", flush=True)
    best = min(grid, key=lambda g: g["wer"])
    print("best on", args.tune_split, best)

    ev = torch.load(os.path.join(args.out, f"{args.eval_split}.logits.pt"), weights_only=False)
    rows = ev["rows"]
    hyps = run(ev, units, lm, pieces, unigrams, best["alpha"], best["beta"], procs)
    by_src = collections.defaultdict(lambda: ([], []))
    for r, h in zip(rows, hyps):
        by_src[r.get("src") or "all"][0].append(r["text"])
        by_src[r.get("src") or "all"][1].append(h)
    res = {"lm": args.lm, "alpha": best["alpha"], "beta": best["beta"], "grid": grid,
           "overall": score([r["text"] for r in rows], hyps),
           "by_source": {s: score(a, b) for s, (a, b) in sorted(by_src.items())}}
    tag = f"eval_lm_{args.eval_split}"
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, tag + ".hyps.jsonl"), "w", encoding="utf-8") as f:
        for r, h in zip(rows, hyps):
            f.write(json.dumps({"id": r["id"], "src": r.get("src"), "ref": r["text"], "hyp": h}, ensure_ascii=False) + "\n")
    for s, m in [("overall", res["overall"])] + list(res["by_source"].items()):
        print(f"{s:10s} WER {m['wer']:.2%}  WER_seg {m['wer_seg']:.2%}  CER {m['cer']:.2%}  "
              f"CER_nospace {m['cer_nospace']:.2%}  GER {m['ger']:.2%}  ({m['utts']} utts)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dump", "decode"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--manifests")
    ap.add_argument("--packed")
    ap.add_argument("--lm")
    ap.add_argument("--tune-split", default="dev2k")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--tune-limit", type=int, default=1000)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.35, 0.5, 0.7])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    args = ap.parse_args(argv)
    dump(args) if args.phase == "dump" else decode(args)


if __name__ == "__main__":
    main()
