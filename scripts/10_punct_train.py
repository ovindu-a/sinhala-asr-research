"""Fine-tune a multilingual encoder (default XLM-R base) as a punctuation tagger on scripts/09_punct_data.py output.

Saves <out>/best (highest dev micro-F1 over , . ?) and <out>/metrics.jsonl.
usage: python scripts/10_punct_train.py --data /data/punct --out /runs/punct/xlmr_base --steps 8000
"""

import argparse
import json
import math
import os
import random
import time

import torch

from sinhala_asr.punct import Scores
from sinhala_asr.punct_model import PunctTagger, encode, new_model


def load_lines(path):
    with open(path, "rb") as f:
        return f.readlines()


def evaluate(tagger, rows):
    s = Scores()
    for r, hyp in zip(rows, tagger.predict([r["w"] for r in rows], batch_size=64)):
        s.add([int(c) for c in r["l"]], hyp)
    return s.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--dev-windows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_lines(os.path.join(args.data, "train.jsonl"))
    dev = [json.loads(x) for x in load_lines(os.path.join(args.data, "dev.jsonl"))]
    dev = random.Random(0).sample(dev, min(args.dev_windows, len(dev)))
    print(f"train windows {len(train):,}, dev windows {len(dev):,}", flush=True)

    model, tok = new_model(args.base)
    model.to(dev_type)
    tagger = PunctTagger.from_model(model, tok, dev_type)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = int(args.warmup * args.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (args.steps - s) / max(1, args.steps - warm)))

    order, pos = list(range(len(train))), len(train)
    best, t0, run_loss = -1.0, time.time(), 0.0
    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(order):
            random.shuffle(order)
            pos = 0
        rows = [json.loads(train[i]) for i in order[pos:pos + args.batch]]
        pos += args.batch
        enc, _ = encode(tok, [r["w"] for r in rows], [r["l"] for r in rows])
        model.train()
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
            loss = model(**{k: v.to(dev_type) for k, v in enc.items()}).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {run_loss / 100:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.3f}s/step", flush=True)
            run_loss = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
                m = evaluate(tagger, dev)
            m["step"] = step
            log.write(json.dumps(m) + "\n")
            log.flush()
            f1 = m["micro"]["f1"]
            print(f"dev step {step}: micro-F1 {f1:.4f} | , {m['COMMA']['f1']:.3f} . {m['PERIOD']['f1']:.3f} "
                  f"? {m['QUESTION']['f1']:.3f}", flush=True)
            if f1 > best and not math.isnan(f1):
                best = f1
                model.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
                with open(os.path.join(args.out, "best", "dev.json"), "w") as f:
                    json.dump(m, f, indent=1)
    print(f"best dev micro-F1 {best:.4f}")


if __name__ == "__main__":
    main()
