"""Step 1b (free, local): does EmbeddingGemma 2 already put Sinhala speech near its transcript?

Audio->text retrieval among N clips with their own transcripts. If the shared space works for Sinhala, the
correct transcript ranks near the top; chance R@1 is 1/N. An English set is run the same way as a reference.

usage: uv run scripts/02_zeroshot_probe.py --n 300
"""

import argparse
import glob
import os
import random
import unicodedata

import numpy as np
import soundfile as sf
import soxr
import torch
from sentence_transformers import SentenceTransformer

SR = 16_000


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    return soxr.resample(wav, sr, SR) if sr != SR else wav


def sinhala_set(n, root="data/openslr52", tsv="data/openslr52/utt_spk_text.tsv"):
    text = {}
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            u, _, t = line.rstrip("\n").split("\t")
            text[u] = unicodedata.normalize("NFC", t)
    files = sorted(glob.glob(os.path.join(root, "asr_sinhala", "data", "*", "*.flac")))
    random.Random(0).shuffle(files)
    # distinct transcripts only, so retrieval is not ambiguous
    out, seen = [], set()
    for p in files:
        t = text.get(os.path.basename(p)[:-5])
        if t and t not in seen and len(t.split()) >= 3:
            seen.add(t)
            out.append((load_wav(p), t))
        if len(out) == n:
            break
    return out


def english_set(n):
    import io

    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))  # decode with soundfile, not torchcodec
    out = []
    for ex in ds.select(range(min(n, len(ds)))):
        a = ex["audio"]
        wav, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"], dtype="float32", always_2d=True)
        wav = wav.mean(1)
        out.append((soxr.resample(wav, sr, SR) if sr != SR else wav, ex["text"].lower()))
    return out


def evaluate(model, pairs, name, prompts):
    audio = [{"audio": {"array": w, "sampling_rate": SR}} for w, _ in pairs]
    a = model.encode(audio, batch_size=8, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    n = len(pairs)
    for label, prompt in prompts.items():
        t = model.encode([t for _, t in pairs], prompt=prompt, batch_size=32, normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=False)
        sim = (a @ t.T).float().cpu()
        rank = (sim > sim.diag()[:, None]).sum(1)  # 0 = correct transcript is top
        r1, r5 = (rank < 1).float().mean().item(), (rank < 5).float().mean().item()
        mrr = (1.0 / (rank + 1).float()).mean().item()
        gap = (sim.diag().mean() - sim.mean()).item()
        print(f"{name:8s} N={n:4d}  text prompt={label:9s}  R@1 {r1:6.1%}  R@5 {r5:6.1%}  MRR {mrr:.3f}  "
              f"(chance R@1 {1 / n:.1%}; pos-vs-mean cos gap {gap:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("google/embeddinggemma-2", device=device,
                                model_kwargs={"dtype": torch.float32}, config_kwargs={"vision_config": None})
    prompts = {"none": "", "document": "title: none | text: "}
    en = english_set(args.n)
    si = sinhala_set(args.n)
    evaluate(model, en, "english", prompts)
    evaluate(model, si, "sinhala", prompts)
    # same N for a fair comparison
    evaluate(model, si[: len(en)], "sinhala", prompts)


if __name__ == "__main__":
    main()
