"""Build a word 4-gram KenLM for CTC shallow fusion (QUAN-style), with test/dev sentences removed.

Text: Sinhala Wikipedia (wikimedia/wikipedia 20231101.si, CC-BY-SA) + FineWeb-2 sin_Sinh (ODC-By, capped) + our
training transcripts. Normalized like the ASR targets. OpenSLR prompts come from news/web text, so every external
sentence that contains a dev/test reference (>= 3 words) is dropped (Aho-Corasick over word boundaries).
Training transcripts are kept as-is (the acoustic model and the Whisper baseline saw them too).

writes <out>/lm.bin, <out>/unigrams.txt, <out>/lm_config.json
usage: python scripts/07_build_lm.py --out /data/lm/si_4gram --train-manifests /data/manifests/v6_plus_slr52/train.jsonl \
          --exclude /data/manifests/ss_v6/test.jsonl /data/manifests/ss_v6/dev2k.jsonl --max-web-words 150000000
"""

import argparse
import collections
import json
import os
import re
import subprocess

import ahocorasick
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from sinhala_asr.text import is_sinhala, normalize

SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


def sentences(doc: str):
    for s in SPLIT.split(doc):
        t = normalize(s)
        if len(t.split()) >= 3 and is_sinhala(t):
            yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-manifests", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="+", required=True)
    ap.add_argument("--max-web-words", type=int, default=150_000_000)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--unigrams", type=int, default=500_000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    A = ahocorasick.Automaton()
    n_ex = 0
    for path in args.exclude:
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                if len(t.split()) >= 3:
                    A.add_word(f" {t} ", t)
                    n_ex += 1
    A.make_automaton()
    print(f"exclusion patterns: {n_ex:,}")

    corpus = os.path.join(args.out, "corpus.txt")
    stats = collections.Counter()
    words = collections.Counter()
    with open(corpus, "w", encoding="utf-8") as out:

        def emit(t, src):
            if src != "train" and next(A.iter(f" {t} "), None) is not None:
                stats[f"{src}_dropped_leak"] += 1
                return 0
            out.write(t + "\n")
            ws = t.split()
            words.update(ws)
            stats[f"{src}_sent"] += 1
            stats[f"{src}_words"] += len(ws)
            return len(ws)

        for path in args.train_manifests:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    emit(json.loads(line)["text"], "train")

        wiki = hf_hub_download("wikimedia/wikipedia", "20231101.si/train-00000-of-00001.parquet",
                               repo_type="dataset", local_dir="/tmp/lmtext")
        for doc in tqdm(pq.read_table(wiki, columns=["text"]).column("text").to_pylist(), desc="wiki", mininterval=30):
            for t in sentences(doc):
                emit(t, "wiki")

        web = hf_hub_download("HuggingFaceFW/fineweb-2", "data/sin_Sinh/train/000_00000.parquet",
                              repo_type="dataset", local_dir="/tmp/lmtext")
        pf, n_web = pq.ParquetFile(web), 0
        for rg in tqdm(range(pf.metadata.num_row_groups), desc="fineweb2", mininterval=30):
            for doc in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                for t in sentences(doc):
                    n_web += emit(t, "web")
            if n_web >= args.max_web_words:
                break
    print("corpus:", dict(stats))

    arpa, binary = os.path.join(args.out, "lm.arpa"), os.path.join(args.out, "lm.bin")
    prune = " ".join(["0"] + ["1"] * (args.order - 1))  # drop singleton 2+-grams to keep the model small
    subprocess.run(f"lmplz -o {args.order} -S 60% -T /tmp --prune {prune} < {corpus} > {arpa}", shell=True, check=True)
    subprocess.run(f"build_binary -q 8 -b 8 trie {arpa} {binary}", shell=True, check=True)
    with open(os.path.join(args.out, "unigrams.txt"), "w", encoding="utf-8") as f:
        for w, _ in words.most_common(args.unigrams):
            f.write(w + "\n")
    with open(os.path.join(args.out, "lm_config.json"), "w") as f:
        json.dump({"order": args.order, "prune": prune, "stats": dict(stats), "exclusion_patterns": n_ex,
                   "text": "wikimedia/wikipedia 20231101.si + FineWeb-2 sin_Sinh (capped) + training transcripts; "
                           "external sentences containing any dev/test reference (>=3 words) removed"}, f, indent=1)
    for p in (arpa, corpus):
        os.remove(p)
    print("lm:", binary, f"{os.path.getsize(binary) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
