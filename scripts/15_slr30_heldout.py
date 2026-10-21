"""Held-out test set from OpenSLR-30 (Google's studio-recorded Sinhala TTS corpus, CC BY-SA 4.0, 12 speakers, 48 kHz).

The Hugging Face mirror keshan/multispeaker-tts-sinhala (data.tar.gz + file_index.tsv) has 5,364 utterances: 2,064
OpenSLR-30 ones (sin_<speaker>_*, 12 speakers) and 3,300 from one more speaker (pn_sin_01_*); reported separately. Not independent of the training data as a whole: ~15% of its utterances
are also SLR-52 / v6 sentences. Kept here are only the
sentences that occur in none of:
  - the training transcripts (--exclude: v6 + SLR-52 train) and the v6 dev / test sets,
  - the language models' text (--corpus: si_corpus = the si_4gram / si_graph8 text), as a whole word-bounded span,
so no acoustic model and no LM has seen the sentence. The speakers and the recording conditions are new too.

writes <out>/test.jsonl (manifest, normalized text), <out>/test.{pcm,json} (16 kHz int16, as 05_pack_parquet.py),
       <out>/stats.json

  python scripts/15_slr30_heldout.py --tar data/slr30/hf_data.tar.gz --index data/slr30/hf_file_index.tsv \
      --exclude reference/manifests/v6_plus_slr52/train.jsonl.gz data/manifests/ss_v6/{dev2k,test}.jsonl \
      --corpus data/lm_dl/corpus.txt --out data/slr30/heldout
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
import tarfile

import numpy as np
import soundfile as sf
import soxr

from sinhala_asr.text import has_digits, normalize

SR = 16_000


def read_jsonl(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="data.tar.gz (HF mirror) or si_lk.tar.gz (openslr.org/30)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", help="file_index.tsv (HF mirror: sentence<TAB>file_path)")
    src.add_argument("--lines", help="si_lk.lines.txt (openslr.org)")
    ap.add_argument("--exclude", nargs="+", required=True, help="manifests whose sentences are excluded (.jsonl[.gz])")
    ap.add_argument("--corpus", default=None, help="LM text; sentences found in it (word-bounded) are excluded")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    if args.index:
        with open(args.index, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows.append({"id": os.path.basename(r["file_path"])[:-4], "text": normalize(r["sentence"]),
                             "raw_text": r["sentence"]})
    else:
        with open(args.lines, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line.strip())
                if m:
                    rows.append({"id": m.group(1), "text": normalize(m.group(2)), "raw_text": m.group(2)})
    stats = {"slr30_utts": len(rows)}

    seen = set()
    for p in args.exclude:
        seen |= {r["text"] for r in read_jsonl(p)}
    rows_in = [r for r in rows if r["text"] and r["text"] not in seen and not has_digits(r["text"])]
    stats["after_transcript_exclusion"] = len(rows_in)

    if args.corpus:  # sentence occurs anywhere in the LM text, as a whole-word span
        import ahocorasick

        A = ahocorasick.Automaton()
        for k, r in enumerate(rows_in):
            A.add_word(f" {r['text']} ", k)
        A.make_automaton()
        hit = set()
        with open(args.corpus, encoding="utf-8") as f:
            for n, line in enumerate(f):
                for _, k in A.iter(f" {line.strip()} "):
                    hit.add(k)
                if n % 2_000_000 == 0:
                    print(f"  corpus {n:,} lines, {len(hit)} sentences found so far", flush=True)
        rows_in = [r for k, r in enumerate(rows_in) if k not in hit]
        stats["found_in_lm_text"] = len(hit)
    stats["heldout_utts"] = len(rows_in)
    keep = {r["id"]: r for r in rows_in}

    audio = {}
    with tarfile.open(args.tar) as t:
        for m in t:
            name = os.path.basename(m.name)
            if m.isfile() and name.endswith(".wav") and name[:-4] in keep:
                wav, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True)
                wav = wav.mean(1)
                if sr != SR:
                    wav = soxr.resample(wav, sr, SR)
                audio[name[:-4]] = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    missing = [i for i in keep if i not in audio]
    assert not missing, f"{len(missing)} held-out utterances have no audio in the tar, e.g. {missing[:3]}"

    ids = sorted(keep)
    offsets, total = [0], 0
    with open(os.path.join(args.out, "test.pcm"), "wb") as f, \
            open(os.path.join(args.out, "test.jsonl"), "w", encoding="utf-8") as man:
        for i in ids:
            a = audio[i]
            f.write(a.tobytes())
            total += len(a)
            offsets.append(total)
            # sin_<spk>_*: OpenSLR-30 proper (12 speakers); pn_sin_01_*: one more speaker in the HF mirror
            src, spk = ("slr30", i.split("_")[1]) if i.startswith("sin_") else ("pn_sin_01", "pn_sin_01")
            r = keep[i] | {"dur": round(len(a) / SR, 2), "src": src, "spk": spk}
            man.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump({"ids": ids, "offsets": offsets}, f)
    by_src = {}
    for i in ids:
        k = "slr30" if i.startswith("sin_") else "pn_sin_01"
        by_src[k] = by_src.get(k, 0) + 1
    stats |= {"hours": total / SR / 3600, "utts_by_source": by_src,
              "speakers": len({i.split("_")[1] if i.startswith("sin_") else "pn_sin_01" for i in ids})}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
