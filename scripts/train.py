"""Train the CTC reader (single GPU / MPS). Writes metrics.jsonl like QUAN's, plus QUAN-loadable checkpoints.

Probe (step 1b), encoder frozen:
  uv run scripts/train.py --train data/manifests/openslr/train.jsonl --dev data/manifests/openslr/dev.jsonl \
      --out runs/probe_frozen --unfreeze-top-k 0 --steps 3000
"""

import argparse
import json
import math
import os
import random
import sys
import time

import jiwer
import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Sampler

from sinhala_asr.model import Config, CTCModel, greedy, load_full
from sinhala_asr.units import Units

SR = 16_000


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class AudioSet(Dataset):
    def __init__(self, rows, units: Units, packed: str | None = None):
        """packed: prefix written by scripts/04_pack_audio.py; the whole split is read into RAM once."""
        self.rows = rows
        self.targets = [units.encode(r["text"]) for r in rows]
        self.pcm = self.offsets = None
        if packed:
            with open(packed + ".json") as f:
                meta = json.load(f)
            assert meta["ids"] == [r["id"] for r in rows], f"{packed} does not match the manifest"
            t = time.time()
            self.pcm = np.load(packed + ".npy")
            self.offsets = np.asarray(meta["offsets"])
            print(f"loaded {packed}.npy ({self.pcm.nbytes / 1e9:.2f} GB) in {time.time() - t:.0f}s")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.pcm is not None:
            wav = self.pcm[self.offsets[i] : self.offsets[i + 1]].astype(np.float32) / 32767
            return torch.from_numpy(wav), torch.tensor(self.targets[i]), i
        wav, sr = sf.read(self.rows[i]["audio"], dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if sr != SR:
            wav = soxr.resample(wav, sr, SR)
        return torch.from_numpy(np.ascontiguousarray(wav)), torch.tensor(self.targets[i]), i


def collate(batch):
    wavs, tgts, idx = zip(*batch)
    wl = torch.tensor([len(w) for w in wavs])
    tl = torch.tensor([len(t) for t in tgts])
    W = torch.zeros(len(wavs), int(wl.max()))
    for k, w in enumerate(wavs):
        W[k, : len(w)] = w
    return W, wl, torch.cat(tgts), tl, list(idx)


class DurationBatches(Sampler):
    """Batches of similar duration, capped at max_seconds of audio; reshuffled every epoch."""

    def __init__(self, durs, max_seconds, shuffle=True, seed=0):
        self.durs, self.max_seconds, self.shuffle, self.seed, self.epoch = durs, max_seconds, shuffle, seed, 0

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.durs)))
        if self.shuffle:
            rng.shuffle(idx)
        out = []
        for s in range(0, len(idx), 2000):  # sort within pools so padding stays small
            pool = sorted(idx[s : s + 2000], key=lambda i: self.durs[i])
            cur, longest = [], 0.0
            for i in pool:
                longest2 = max(longest, self.durs[i])
                if cur and longest2 * (len(cur) + 1) > self.max_seconds:
                    out.append(cur)
                    cur, longest2 = [], self.durs[i]
                cur.append(i)
                longest = longest2
            if cur:
                out.append(cur)
        if self.shuffle:
            rng.shuffle(out)
        return out

    def __iter__(self):
        b = self._batches()
        self.epoch += 1
        return iter(b)


@torch.no_grad()
def evaluate(model, loader, units, device, amp, max_batches=None):
    model.eval()
    hyps, refs, n = [], [], 0
    for k, (wav, wl, _, _, idx) in enumerate(loader):
        if max_batches and k >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logp, ol = model(wav.to(device), wl.to(device))
        for seq, i in zip(greedy(logp, ol), idx):
            hyps.append(units.decode(seq))
            refs.append(loader.dataset.rows[i]["text"])
        n += len(idx)
    model.train()
    hyps = [h if h else "<empty>" for h in hyps]
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "utts": n, "examples": list(zip(refs[:3], hyps[:3]))}


def save(model, cfg, units, out, tag):
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    save_file(model.trainable_state_dict(), os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=1)
    units.save(os.path.join(d, "units.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", default=None, help="existing units.json; default: build from --train")
    ap.add_argument("--unfreeze-top-k", type=int, default=0)
    ap.add_argument("--head-init", default="text", choices=["text", "random"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--enc-lr-mult", type=float, default=0.1)
    ap.add_argument("--layer-lr-mult", type=float, default=50.0)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch-seconds", type=float, default=600)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-dynamic-chunk", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--packed-dir", default=None, help="dir with train.npy/json, dev.npy/json from 04_pack_audio.py")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device.type == "cuda"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args) | {"argv": sys.argv}, f, indent=1)

    train_rows, dev_rows = read_jsonl(args.train), read_jsonl(args.dev)
    units = Units.load(args.units) if args.units else Units.build("google/embeddinggemma-2", [r["text"] for r in train_rows])
    print(f"train {len(train_rows):,} utts / {sum(r['dur'] for r in train_rows) / 3600:.1f} h; "
          f"dev {len(dev_rows):,}; units {len(units):,}")

    cfg = Config(vocab_size=len(units), unfreeze_top_k=args.unfreeze_top_k, head_init=args.head_init,
                 dynamic_chunk=not args.no_dynamic_chunk)
    model = CTCModel(cfg, load_full(cfg.encoder_id), units.gemma_ids).to(device)
    model.train()
    model.encoder.eval()  # frozen encoder layers: keep their dropout (if any) off
    for layer in model.encoder.layers[len(model.encoder.layers) - cfg.unfreeze_top_k :] if cfg.unfreeze_top_k else []:
        layer.train()

    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("encoder.") and n != "layer_logits"]
    decay = lambda p: p.ndim >= 2  # noqa: E731
    groups = [
        {"params": [p for p in other if decay(p)], "weight_decay": args.wd, "lr_mult": 1.0},
        {"params": [p for p in other if not decay(p)], "weight_decay": 0.0, "lr_mult": 1.0},
    ]
    if model.layer_logits is not None:  # 13 scalars; at the base lr they barely move
        groups.append({"params": [model.layer_logits], "weight_decay": 0.0, "lr_mult": args.layer_lr_mult})
    if enc_params:
        groups.append({"params": enc_params, "weight_decay": args.wd, "lr_mult": args.enc_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_train / 1e6:.1f}M (encoder top {cfg.unfreeze_top_k}: "
          f"{sum(p.numel() for p in enc_params) / 1e6:.1f}M)")

    warm = int(args.warmup * args.steps)

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        p = (s - warm) / max(1, args.steps - warm)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    pk = (lambda n: os.path.join(args.packed_dir, n)) if args.packed_dir else (lambda n: None)
    train_set, dev_set = AudioSet(train_rows, units, pk("train")), AudioSet(dev_rows, units, pk("dev"))
    unk_rate = sum(t.count(1) for t in dev_set.targets) / max(1, sum(len(t) for t in dev_set.targets))
    print(f"dev target tokens mapped to <unk>: {unk_rate:.2%}")
    sampler = DurationBatches([r["dur"] for r in train_rows], args.batch_seconds, seed=args.seed)
    kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=amp)
    train_dl = DataLoader(train_set, batch_sampler=sampler, **kw)
    dev_dl = DataLoader(dev_set, batch_sampler=DurationBatches([r["dur"] for r in dev_rows], args.batch_seconds, shuffle=False), **kw)

    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    step, t0, audio_s, best = 0, time.time(), 0.0, float("inf")
    run_loss, skipped, data_s = [], 0, 0.0
    while step < args.steps:
        t_data = time.time()
        for wav, wl, tgt, tl, _ in train_dl:
            data_s += time.time() - t_data
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step) * g["lr_mult"]
            wav, wl = wav.to(device, non_blocking=True), wl.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logp, ol = model(wav, wl)
            ctc_dev = device if device.type == "cuda" else torch.device("cpu")  # no CTC kernel on MPS
            loss = F.ctc_loss(logp.transpose(0, 1).float().to(ctc_dev), tgt.to(ctc_dev), ol.to(ctc_dev), tl.to(ctc_dev),
                              blank=0, reduction="sum", zero_infinity=True) / tl.sum().clamp(min=1)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            step += 1
            run_loss.append(loss.item())
            audio_s += wl.sum().item() / SR
            t_data = time.time()

            if step % args.log_every == 0:
                el = time.time() - t0
                rec = {"step": step, "loss": float(np.mean(run_loss)), "lr": lr_at(step), "grad_norm": gn.item(),
                       "audio_hours_per_s": audio_s / 3600 / el, "s_per_step": el / args.log_every,
                       "skipped": skipped, "epoch": sampler.epoch, "data_wait_frac": data_s / el}
                if device.type == "cuda":
                    rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                if model.layer_logits is not None:
                    rec["layer_w"] = [round(x, 3) for x in torch.softmax(model.layer_logits.detach().float(), 0).tolist()]
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss, t0, audio_s, data_s = [], time.time(), 0.0, 0.0

            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, dev_dl, units, device, amp, args.eval_batches)
                print(f"[eval step {step}] WER {ev['wer']:.2%} CER {ev['cer']:.2%} ({ev['utts']} utts)")
                for r, h in ev["examples"]:
                    print(f"   ref: {r}\n   hyp: {h}")
                log.write(json.dumps({"step": step, "eval": {"dev": {k: ev[k] for k in ("wer", "cer", "utts")}}}) + "\n")
                log.flush()
                if ev["cer"] < best:
                    best = ev["cer"]
                    save(model, cfg, units, args.out, "best")
                t0, t_data = time.time(), time.time()
    save(model, cfg, units, args.out, "final")
    print(f"done; best dev CER {best:.2%}")


if __name__ == "__main__":
    main()
