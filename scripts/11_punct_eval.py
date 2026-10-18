"""Score punctuation restoration: B0 (end-of-transcript rule), B1 (per-word label statistics) and, with --model,
the trained tagger, on every test set from scripts/09_punct_data.py plus ASR output.

ASR sets: --hyps files from scripts/evaluate.py / 08_lm_decode.py (id, src, ref, hyp) on v6 test. Reference marks
come from the raw v6 transcript and are moved onto the ASR words by alignment (sinhala_asr.punct.transfer_labels).

usage: python scripts/11_punct_eval.py --data /data/punct --v6 /data/manifests/ss_v6 \
          --hyps ctc_lm=/root/app/runs/v6x_top8_lm_test.hyps.jsonl --model /runs/punct/xlmr_base/best --out eval.json
"""

import argparse
import json
import os

from sinhala_asr.punct import Scores, WordStats, end_only, render, transfer_labels, words_and_labels

TESTS = ["test_wiki", "test_web", "test_spoken", "test_slr"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--hyps", nargs="*", default=[], help="name=path")
    ap.add_argument("--model", default=None)
    ap.add_argument("--stats-words", type=int, default=5_000_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sets = {}
    for t in TESTS:
        rows = read(os.path.join(args.data, f"{t}.jsonl"))
        sets[t] = [(r["w"], [int(c) for c in r["l"]]) for r in rows]
    raw = {r["id"]: r.get("raw_text") or r["text"] for r in read(os.path.join(args.v6, "test.jsonl"))}
    for spec in args.hyps:
        name, path = spec.split("=", 1)
        all_, spoken = [], []
        for h in read(path):
            hw = h["hyp"].split()
            if not hw:
                continue
            rw, rl = words_and_labels(raw[h["id"]])
            item = (hw, transfer_labels(rw, rl, hw))
            all_.append(item)
            if h.get("src") in ("youtube", "bizbrains", "linga"):
                spoken.append(item)
        sets[f"asr_{name}"] = all_
        sets[f"asr_{name}_spoken"] = spoken

    ws, n = WordStats(), 0
    with open(os.path.join(args.data, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ws.add(r["w"], [int(c) for c in r["l"]])
            n += len(r["w"])
            if n >= args.stats_words:
                break
    systems = {"B0_end_rule": end_only, "B1_word_stats": ws.predict}
    tagger = None
    if args.model:
        import torch

        from sinhala_asr.punct_model import PunctTagger

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tagger = PunctTagger(args.model, dev, torch.bfloat16 if dev == "cuda" else torch.float32)

    results, samples = {}, {}
    for name, items in sets.items():
        words = [w for w, _ in items]
        preds = {s: [fn(w) for w in words] for s, fn in systems.items()}
        if tagger:
            preds["model"] = tagger.predict(words, batch_size=64)
        results[name] = {}
        for s, ps in preds.items():
            sc = Scores()
            for (_, ref), hyp in zip(items, ps):
                sc.add(ref, hyp)
            results[name][s] = sc.summary()
        samples[name] = [{"ref": render(w, r), **{s: render(w, preds[s][i]) for s in preds}}
                         for i, (w, r) in enumerate(items[:25])]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": results, "samples": samples}, f, ensure_ascii=False, indent=1)
    print(f"| set | words | system | , F1 | . F1 | ? F1 | micro P / R / F1 | sentence-end F1 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, by_sys in results.items():
        for s, m in by_sys.items():
            print(f"| {name} | {m['words']:,} | {s} | {m['COMMA']['f1']:.3f} | {m['PERIOD']['f1']:.3f} | "
                  f"{m['QUESTION']['f1']:.3f} | {m['micro']['p']:.3f} / {m['micro']['r']:.3f} / "
                  f"{m['micro']['f1']:.3f} | {m['sentence_end']['f1']:.3f} |")


if __name__ == "__main__":
    main()
