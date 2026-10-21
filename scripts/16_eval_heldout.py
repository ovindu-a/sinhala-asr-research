"""Score every system on a held-out set (15_slr30_heldout.py): greedy CTC, then KenLM decoding with the LM weights
each system was tuned to on v6 dev2k (nothing is tuned on the held-out set), plus Whisper-small as an outside baseline.

Runs on CUDA, Apple MPS or CPU. --audio-tower uses app/prepare_models.py's audio_tower.safetensors instead of the
whole EmbeddingGemma 2 (identical audio features; the text embeddings are only a head initialiser that every
checkpoint overwrites), so a laptop needs no 1.5 GB model download.

  python scripts/16_eval_heldout.py --data data/slr30/heldout --ckpt-root runs/heldout/ckpt \
      --a-ckpt <dir with A's config/units/model> --word-lm <si_4gram dir> --graph-lm <si_graph8 dir> \
      --whisper <whisper dir or HF id> --audio-tower <.../audio_tower.safetensors> --out runs/heldout/results
writes <out>/<system>.hyps.jsonl per system + <out>/summary.json, and prints the table.
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402

from train import AudioSet, DurationBatches, collate, load_weights, read_jsonl  # noqa: E402

import sinhala_asr.model as M  # noqa: E402
from sinhala_asr.metrics import score  # noqa: E402
from sinhala_asr.text import normalize  # noqa: E402
from sinhala_asr.units import load_units  # noqa: E402

LMD = importlib.import_module("08_lm_decode")
TOPK = LMD.TOPK

# (name, checkpoint run, [(label, lm kind, alpha, beta, extra)]): weights from each system's v6 dev2k grid
SYSTEMS = [
    ("A (Gemma units)", "A", [("word 4-gram", "word", 0.35, 1.0, None)]),
    ("A longer", "v6x_top8_long", [("word 4-gram", "word", 0.5, 1.0, None)]),
    ("B (graphemes)", "gr_v6x_top8", [("word 4-gram", "word", 0.35, 1.0, None),
                                      ("grapheme 8-gram", "grapheme", 0.5, 1.0, None),
                                      ("word + grapheme", "dual", 0.2, 1.0, {"g_alpha": 0.3, "g_beta": 1.0})]),
    ("B longer", "gr_v6x_top8_long", [("grapheme 8-gram", "grapheme", 0.7, 2.0, None)]),
    ("B longer + head parts", "gr_v6x_top8_long_parts", [("grapheme 8-gram", "grapheme", 0.7, 1.0, None)]),
    ("C (grapheme-BPE, web-fitted)", "gb3k_v6x_top8", [("word 4-gram", "word", 0.35, 1.0, None)]),
    # C redo was never LM-tuned: C's weights
    ("C redo (speech-fitted)", "gbs3k_v6x_top8", [("word 4-gram", "word", 0.35, 1.0, None)]),
]


def slim_full(tower_path):
    """load_full() stand-in: the audio tower from audio_tower.safetensors + a placeholder text embedding."""
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

    def load(encoder_id, **kw):
        cfg = AutoConfig.from_pretrained(encoder_id)
        enc = Gemma4AudioModel._from_config(cfg.audio_config, dtype=torch.bfloat16).eval()
        own = {**dict(enc.named_parameters()), **dict(enc.named_buffers())}
        state = load_file(tower_path)
        assert set(own) == set(state), "audio_tower.safetensors does not match the encoder"
        with torch.no_grad():
            for k, v in state.items():
                own[k].data = v
        emb = types.SimpleNamespace(weight=torch.zeros(cfg.text_config.vocab_size, cfg.text_config.hidden_size))
        return types.SimpleNamespace(audio_tower=enc, language_model=types.SimpleNamespace(embed_tokens=emb))

    return load


@torch.no_grad()
def greedy_and_topk(ckpt, rows, packed, device):
    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = M.Config(**{k: v for k, v in json.load(f).items() if k in M.Config.__dataclass_fields__})
    units = load_units(os.path.join(ckpt, "units.json"))
    model = M.CTCModel(cfg, M.load_full(cfg.encoder_id), units.init_ids, units.pieces)
    load_weights(model, ckpt)
    model = model.to(device).eval()
    ds = AudioSet(rows, units, packed)
    hyps, topk = [None] * len(rows), [None] * len(rows)
    for wav, wl, _, _, idx in torch.utils.data.DataLoader(
            ds, batch_sampler=DurationBatches([r["dur"] for r in rows], 300, shuffle=False), collate_fn=collate):
        logp, ol = model(wav.to(device), wl.to(device))
        for b, (seq, k) in enumerate(zip(M.greedy(logp, ol), idx)):
            hyps[k] = units.decode(seq)
            n = int(ol[b])
            v, i = logp[b, :n].float().topk(TOPK, dim=-1)
            topk[k] = (v.half().cpu(), i.short().cpu())
    del model
    return units, hyps, {"rows": rows, "topk": topk, "vocab": len(units)}


def lm_decode(units, data, kind, alpha, beta, extra, word_lm, graph_lm, procs):
    pieces = units.pieces
    if kind == "word":
        lm, dual = os.path.join(word_lm, "lm.bin"), None
        vocab = os.path.join(word_lm, "unigrams.txt")
    elif kind == "grapheme":
        pieces = [p if not p else "▁|" if p == "▁" else "▁" + p for p in pieces]
        lm, dual = os.path.join(graph_lm, "lm.bin"), None
        vocab = os.path.join(graph_lm, "vocab.txt")
    else:  # dual: word LM at word ends + grapheme LM per unit
        lm, vocab = os.path.join(word_lm, "lm.bin"), os.path.join(word_lm, "unigrams.txt")
        dual = {"glm": os.path.join(graph_lm, "lm.bin")} | extra
    with open(vocab, encoding="utf-8") as f:
        unigrams = [w.strip() for w in f if w.strip()]
    return LMD.run(data, units, lm, pieces, unigrams, alpha, beta, procs, dual=dual)


@torch.no_grad()
def whisper(path, rows, packed, device, batch=16):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float32).to(device).eval()
    with open(packed + ".json") as f:
        off = json.load(f)["offsets"]
    pcm = np.memmap(packed + ".pcm", dtype=np.int16, mode="r")
    hyps = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["dur"])
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        wavs = [pcm[off[i] : off[i + 1]].astype(np.float32) / 32767 for i in idx]
        feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        out = model.generate(feats.to(device), language="si", task="transcribe", max_new_tokens=200)
        for i, t in zip(idx, proc.batch_decode(out, skip_special_tokens=True)):
            hyps[i] = normalize(t)
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with test.jsonl + test.{pcm,json}")
    ap.add_argument("--ckpt-root", required=True, help="<root>/<run>/{config,units}.json + model.safetensors")
    ap.add_argument("--a-ckpt", required=True, help="A's checkpoint dir (v6x_top8)")
    ap.add_argument("--word-lm", required=True)
    ap.add_argument("--graph-lm", required=True)
    ap.add_argument("--whisper", default="SinhaSpeech/whisper-small-sinhala-run12-best")
    ap.add_argument("--audio-tower", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="system names to run (default: all + Whisper)")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.audio_tower:
        M.load_full = slim_full(args.audio_tower)
    rows = read_jsonl(os.path.join(args.data, "test.jsonl"))
    refs = [r["text"] for r in rows]
    packed = os.path.join(args.data, "test")
    summary_path = os.path.join(args.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    def save(key, hyps, extra=None):
        m = score(refs, hyps)
        srcs = sorted({r["src"] for r in rows})
        by_src = {s: score([r["text"] for r in rows if r["src"] == s], [h for r, h in zip(rows, hyps) if r["src"] == s])
                  for s in srcs} if len(srcs) > 1 else {}
        summary[key] = m | {"by_source": by_src} | (extra or {})
        with open(os.path.join(args.out, key.replace(" ", "_").replace("/", "-") + ".hyps.jsonl"), "w",
                  encoding="utf-8") as f:
            for r, h in zip(rows, hyps):
                f.write(json.dumps({"id": r["id"], "src": r["src"], "ref": r["text"], "hyp": h}, ensure_ascii=False)
                        + "\n")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"{key:60s} WER {m['wer']:.2%}  WER_seg {m['wer_seg']:.2%}  CER {m['cer']:.2%}  GER {m['ger']:.2%}",
              flush=True)

    print(f"{len(rows)} utts, {sum(r['dur'] for r in rows) / 3600:.2f} h, device {device}", flush=True)
    for name, run, lms in SYSTEMS:
        if args.only is not None and name not in args.only:
            continue
        ckpt = args.a_ckpt if run == "A" else os.path.join(args.ckpt_root, run)
        if not os.path.exists(os.path.join(ckpt, "model.safetensors")):
            print(f"skip {name}: no checkpoint in {ckpt}", flush=True)
            continue
        t = time.time()
        units, hyps, data = greedy_and_topk(ckpt, rows, packed, device)
        save(f"{name} | greedy", hyps, {"secs": time.time() - t})
        for label, kind, a, b, extra in lms:
            t = time.time()
            save(f"{name} | {label}", lm_decode(units, data, kind, a, b, extra, args.word_lm, args.graph_lm,
                                                args.procs), {"alpha": a, "beta": b, "secs": time.time() - t})
    if args.only is None or "Whisper-small" in args.only:
        t = time.time()
        save("Whisper-small (SinhaSpeech) | greedy", whisper(args.whisper, rows, packed, device),
             {"secs": time.time() - t})


if __name__ == "__main__":
    main()
