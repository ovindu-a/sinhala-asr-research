"""Speed: our CTC model (greedy, and + KenLM beam search) vs Whisper-small, same GPU, same test clips.

End to end from 16 kHz waveforms in memory to text (feature extraction included). Reports RTFx = audio seconds
processed per wall-clock second, for batched throughput and for one-clip-at-a-time latency.

  python scripts/benchmark_speed.py --ckpt /runs/v6x_top8/best --lm /data/lm/si_4gram \
      --manifest /data/manifests/ss_v6/test.jsonl --packed /data/packed/ss_v6/test --n 200 --n-latency 20
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_weights  # noqa: E402

from sinhala_asr.model import Config, CTCModel, greedy, load_full  # noqa: E402
from sinhala_asr.units import load_units  # noqa: E402

SR = 16_000


def sync():
    torch.cuda.synchronize()


def batches(wavs, max_seconds):
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    cur, out = [], []
    for i in order:
        if cur and len(wavs[i]) / SR * (len(cur) + 1) > max_seconds:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + ([cur] if cur else [])


def timeit(fn, reps=1):
    fn()  # warm-up
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--whisper", default="SinhaSpeech/whisper-small-sinhala-run12-best")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-latency", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    off = json.load(open(args.packed + ".json"))["offsets"]
    pcm = np.fromfile(args.packed + ".pcm", dtype=np.int16)
    pick = random.Random(0).sample(range(len(rows)), args.n)
    wavs = [pcm[off[i] : off[i + 1]].astype(np.float32) / 32767 for i in pick]
    audio_s = sum(len(w) for w in wavs) / SR
    lat_idx = list(range(args.n_latency))
    lat_audio = sum(len(wavs[i]) for i in lat_idx) / SR
    gpu = torch.cuda.get_device_name()
    print(f"{gpu}; {args.n} clips, {audio_s:.0f} s audio (mean {audio_s / args.n:.1f} s); latency set {args.n_latency} clips")
    res = {"gpu": gpu, "clips": args.n, "audio_s": audio_s}

    # ---------------- ours
    cfg = Config(**{k: v for k, v in json.load(open(os.path.join(args.ckpt, "config.json"))).items()
                    if k in Config.__dataclass_fields__})
    units = load_units(os.path.join(args.ckpt, "units.json"))
    model = CTCModel(cfg, load_full(cfg.encoder_id), units.init_ids, units.pieces)
    load_weights(model, args.ckpt)
    model = model.cuda().eval()
    n_ours = sum(p.numel() for p in model.parameters())

    def ours_run(idx_batches, keep_logits=None):
        texts = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in idx_batches:
                L = max(len(wavs[i]) for i in b)
                x = torch.zeros(len(b), L)
                for k, i in enumerate(b):
                    x[k, : len(wavs[i])] = torch.from_numpy(wavs[i])
                lens = torch.tensor([len(wavs[i]) for i in b])
                logp, ol = model(x.cuda(non_blocking=True), lens.cuda())
                if keep_logits is not None:
                    for k, i in enumerate(b):
                        keep_logits[i] = logp[k, : int(ol[k])].float().cpu().numpy()
                texts += [units.decode(s) for s in greedy(logp, ol)]
        return texts

    t = timeit(lambda: ours_run(batches(wavs, 600)), reps=3)
    res["ours_batched_s"] = t
    t1 = timeit(lambda: ours_run([[i] for i in lat_idx]), reps=3)
    res["ours_single_s"] = t1
    print(f"ours greedy   batched: {t:7.3f} s  RTFx {audio_s / t:8.0f} | one-at-a-time: {t1 / len(lat_idx) * 1000:6.1f} ms/clip"
          f"  RTFx {lat_audio / t1:6.0f}")

    if args.lm:  # beam search with KenLM: CPU, one process, per clip
        from quan_transcribe.lm import KenLMDecoder

        logits = {}
        ours_run(batches(wavs, 600), keep_logits=logits)
        pieces = units.pieces
        uni = [w.strip() for w in open(os.path.join(args.lm, "unigrams.txt"), encoding="utf-8") if w.strip()]
        dec = KenLMDecoder(os.path.join(args.lm, "lm.bin"), pieces, uni, alpha=0.35, beta=1.0)
        empty = [k for k, p in enumerate(pieces) if k != 0 and not p]
        t = time.perf_counter()
        for i in range(args.n):
            lp = logits[i].copy()
            lp[:, empty] = -1e4
            dec.decode(lp, beam_width=32)
        tb = time.perf_counter() - t
        res["lm_beam_cpu_1core_s"] = tb
        print(f"ours KenLM beam search (CPU, 1 core): {tb:7.2f} s total, {tb / args.n * 1000:6.1f} ms/clip, "
              f"RTFx {audio_s / tb:6.0f} (add to the GPU time; parallelises across cores)")
    del model
    torch.cuda.empty_cache()

    # ---------------- whisper
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(args.whisper)
    wm = WhisperForConditionalGeneration.from_pretrained(args.whisper, dtype=torch.float16).cuda().eval()
    n_wh = sum(p.numel() for p in wm.parameters())

    def wh_run(idx_batches, bs_cap=64):
        texts = []
        for b in idx_batches:
            for s in range(0, len(b), bs_cap):
                bb = b[s : s + bs_cap]
                feats = proc.feature_extractor([wavs[i] for i in bb], sampling_rate=SR, return_tensors="pt").input_features
                with torch.no_grad():
                    out = wm.generate(feats.cuda(non_blocking=True).half(), language="si", task="transcribe",
                                      num_beams=1, max_new_tokens=200)
                texts += proc.batch_decode(out, skip_special_tokens=True)
        return texts

    t = timeit(lambda: wh_run(batches(wavs, 600)), reps=1)
    res["whisper_batched_s"] = t
    t1 = timeit(lambda: wh_run([[i] for i in lat_idx]), reps=1)
    res["whisper_single_s"] = t1
    print(f"whisper greedy batched: {t:7.3f} s  RTFx {audio_s / t:8.0f} | one-at-a-time: {t1 / len(lat_idx) * 1000:6.1f} ms/clip"
          f"  RTFx {lat_audio / t1:6.0f}")

    res |= {"ours_params": n_ours, "whisper_params": n_wh,
            "speedup_batched": res["whisper_batched_s"] / res["ours_batched_s"],
            "speedup_single": res["whisper_single_s"] / res["ours_single_s"]}
    print(f"\nparams: ours {n_ours / 1e6:.1f}M, whisper {n_wh / 1e6:.1f}M")
    print(f"speed-up (whisper time / ours greedy): batched {res['speedup_batched']:.1f}x, "
          f"one-at-a-time {res['speedup_single']:.1f}x")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
