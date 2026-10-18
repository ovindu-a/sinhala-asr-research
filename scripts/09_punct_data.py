"""Build punctuation-restoration data: windows of normalized words with the mark after each word (sinhala_asr.punct).

Written text: Sinhala Wikipedia (20231101.si) + FineWeb-2 sin_Sinh, the same sources as the KenLM.
Spoken text: SinhaSpeech v6 raw transcripts (OpenSLR prompt fragments + YouTube/BizBrains/Linga utterances).

  train.jsonl       random-length windows (16-128 words, half of them cut at a sentence end) from wiki + web + v6 train
  dev.jsonl         held-out wiki/web windows + v6 dev2k
  test_wiki.jsonl   held-out Wikipedia articles (every 50th), runs of whole sentences of 40-150 words
  test_web.jsonl    held-out FineWeb-2 row groups (the last two), same shape
  test_spoken.jsonl v6 test YouTube/BizBrains/Linga utterances (conversational, punctuated inside)
  test_slr.jsonl    v6 test OpenSLR fragments (3,000 sampled)
Written windows containing any >= 5-word v6 dev2k/test reference are dropped (OpenSLR prompts come from web text).

usage: python scripts/09_punct_data.py --out /data/punct --v6 /data/manifests/ss_v6 --train-words 30000000
"""

import argparse
import collections
import json
import os
import random

import ahocorasick
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from sinhala_asr.punct import PERIOD, QUESTION, words_and_labels
from sinhala_asr.text import is_sinhala

SPOKEN_SRC = {"youtube", "bizbrains", "linga"}


def doc_ok(words, labels) -> bool:
    if len(words) < 20:
        return False
    if sum(map(is_sinhala, words)) < 0.7 * len(words):
        return False
    ends = sum(lab in (PERIOD, QUESTION) for lab in labels)
    return ends >= 1 and len(words) / ends <= 60  # lists, menus and unpunctuated dumps teach "no punctuation"


def train_windows(words, labels, rng):
    i, n = 0, len(words)
    while i < n:
        j = min(n, i + rng.randint(16, 128))
        if rng.random() < 0.5:
            ends = [k + 1 for k in range(i + 8, j) if labels[k] in (PERIOD, QUESTION)]
            if ends:
                j = ends[-1]
        yield words[i:j], labels[i:j]
        i = j


def test_windows(words, labels, lo=40, hi=150):
    start = 0
    for k, lab in enumerate(labels):
        if lab not in (PERIOD, QUESTION):
            continue
        if k + 1 - start > hi:
            start = k + 1
        elif k + 1 - start >= lo:
            yield words[start:k + 1], labels[start:k + 1]
            start = k + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--train-words", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    stats = collections.Counter()

    A = ahocorasick.Automaton()
    for split in ("test", "dev2k"):
        with open(os.path.join(args.v6, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)["text"]
                if len(t.split()) >= 5:
                    A.add_word(f" {t} ", t)
    A.make_automaton()

    files = {k: open(os.path.join(args.out, f"{k}.jsonl"), "w", encoding="utf-8")
             for k in ("train", "dev", "test_wiki", "test_web", "test_spoken", "test_slr")}

    def write(split, src, words, labels):
        if split != "test_slr" and split != "test_spoken" and src != "spoken" \
                and next(A.iter(f" {' '.join(words)} "), None) is not None:
            stats[f"{split}_{src}_dropped_leak"] += 1
            return 0
        files[split].write(json.dumps({"src": src, "w": words, "l": "".join(map(str, labels))}, ensure_ascii=False)
                           + "\n")
        stats[f"{split}_{src}_windows"] += 1
        stats[f"{split}_{src}_words"] += len(words)
        return len(words)

    # spoken: v6 raw transcripts
    for split, out in (("train", "train"), ("dev2k", "dev")):
        with open(os.path.join(args.v6, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                w, lab = words_and_labels(r.get("raw_text") or r["text"])
                if w:
                    write(out, "spoken", w, lab)
    slr = []
    with open(os.path.join(args.v6, "test.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w, lab = words_and_labels(r.get("raw_text") or r["text"])
            if not w:
                continue
            if r["src"] in SPOKEN_SRC:
                write("test_spoken", r["src"], w, lab)
            else:
                slr.append((w, lab))
    for w, lab in random.Random(1).sample(slr, min(3000, len(slr))):
        write("test_slr", "openslr", w, lab)

    # Wikipedia: every 50th article is test, every 50th + 1 dev
    wiki = hf_hub_download("wikimedia/wikipedia", "20231101.si/train-00000-of-00001.parquet", repo_type="dataset")
    for i, doc in enumerate(pq.read_table(wiki, columns=["text"]).column("text").to_pylist()):
        w, lab = words_and_labels(doc)
        if not doc_ok(w, lab):
            stats["wiki_docs_skipped"] += 1
            continue
        if i % 50 == 0:
            for ww, ll in test_windows(w, lab):
                write("test_wiki", "wiki", ww, ll)
        else:
            for ww, ll in train_windows(w, lab, rng):
                write("dev" if i % 50 == 1 else "train", "wiki", ww, ll)
    print({k: v for k, v in stats.items() if "wiki" in k}, flush=True)

    # FineWeb-2: last two row groups held out (dev + test), the rest is training text up to --train-words
    web = hf_hub_download("HuggingFaceFW/fineweb-2", "data/sin_Sinh/train/000_00000.parquet", repo_type="dataset")
    pf = pq.ParquetFile(web)
    n_rg = pf.metadata.num_row_groups
    for split, rg in (("test_web", n_rg - 1), ("dev", n_rg - 2)):
        for j, doc in enumerate(pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()):
            w, lab = words_and_labels(doc)
            if not doc_ok(w, lab):
                continue
            if split == "test_web":
                for ww, ll in test_windows(w, lab):
                    write(split, "web", ww, ll)
            elif j % 4 == 0:
                for ww, ll in train_windows(w, lab, rng):
                    write(split, "web", ww, ll)
    n_train = 0
    for rg in range(n_rg - 2):
        for doc in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            w, lab = words_and_labels(doc)
            if not doc_ok(w, lab):
                stats["web_docs_skipped"] += 1
                continue
            for ww, ll in train_windows(w, lab, rng):
                n_train += write("train", "web", ww, ll)
        print(f"web row group {rg}: {n_train:,} train words", flush=True)
        if n_train >= args.train_words:
            break

    for f in files.values():
        f.close()
    with open(os.path.join(args.out, "data_stats.json"), "w") as f:
        json.dump(dict(stats), f, indent=1, sort_keys=True)
    print(json.dumps(dict(stats), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
