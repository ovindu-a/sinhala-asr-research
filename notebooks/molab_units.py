# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "transformers>=5.19",
#     "huggingface_hub[cli]>=1.25",
#     "numpy",
#     "jiwer",
#     "rapidfuzz",
#     "sentencepiece",
#     "grapheme-kit==0.1.1",
# ]
# ///
"""C redo, step 1 (CPU): a grapheme-BPE tokenizer fitted to speech instead of web text.

The first C (gbpe3000) took its alphabet from web text (1,266 graphemes, 455 never spoken), so 1,490 of its 3,000
pieces were single graphemes and its merges followed web vocabulary (1.99 units per spoken word). Here the alphabet is
the graphemes of the speech transcripts only, and the transcripts' word counts are added --transcript-weight times on
top of the 160 M-word KenLM text before BPE. Output goes to the bucket as units_speech/, which the
"C redo" presets of molab_train.py use. See docs/MOLAB.md.
"""

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import urllib.request
    import zipfile

    import marimo as mo

    return json, mo, os, shutil, subprocess, sys, urllib, zipfile


@app.cell
def _(mo):
    BUCKET = "ovindu-a/sinhala-grapheme-asr"
    REPO_ZIP = "https://codeload.github.com/ovindu-a/sinhala-asr-research/zip/refs/heads/grapheme-units"
    WORK = "/tmp/sinhala_units"
    weight = mo.ui.number(start=0, stop=1000, step=10, value=50, label="--transcript-weight")
    vocab = mo.ui.dropdown(options=["3000", "2000", "4000"], value="3000", label="BPE vocab")
    compare = mo.ui.checkbox(value=False, label="also report Gemma units (downloads the Gemma tokenizer)")
    hf_token = mo.ui.text(kind="password", label="HF_TOKEN (leave empty if set in the environment)")
    go = mo.ui.run_button(label="Build units (~15–30 min on 4 CPUs)")
    mo.vstack([weight, vocab, compare, hf_token, go])
    return BUCKET, REPO_ZIP, WORK, compare, go, hf_token, vocab, weight


@app.cell
def _(BUCKET, REPO_ZIP, WORK, compare, go, hf_token, mo, os, shutil, subprocess, sys, urllib, vocab, weight, zipfile):
    mo.stop(not go.value, mo.md("Set the options and press **Build units**."))
    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value
    os.makedirs(WORK, exist_ok=True)
    _zip = os.path.join(WORK, "code.zip")
    urllib.request.urlretrieve(REPO_ZIP, _zip)
    CODE = os.path.join(WORK, "code")
    shutil.rmtree(CODE, ignore_errors=True)
    with zipfile.ZipFile(_zip) as _z:
        _z.extractall(WORK)
    os.replace(os.path.join(WORK, "sinhala-asr-research-grapheme-units"), CODE)

    def _sh(cmd):
        print("+", cmd, flush=True)
        subprocess.run(cmd, shell=True, check=True, cwd=CODE, env=os.environ | {"PYTHONPATH": CODE})

    _m = os.path.join(WORK, "manifests")
    os.makedirs(_m, exist_ok=True)
    _sh(f"gunzip -c reference/manifests/v6_plus_slr52/train.jsonl.gz > {_m}/train.jsonl")
    _sh(f"cp reference/manifests/ss_v6/test.jsonl {_m}/test.jsonl")
    _sh(f"hf buckets sync hf://buckets/{BUCKET}/lm/si_corpus {WORK}/si_corpus --include corpus.txt")
    OUT = os.path.join(WORK, "units_speech")
    _sh(f"{sys.executable} scripts/12_grapheme_inventory.py --corpus {WORK}/si_corpus/corpus.txt "
        f"--train-manifests {_m}/train.jsonl --out {OUT} --speech-alphabet --transcript-weight {int(weight.value)}")
    _sh(f"{sys.executable} scripts/13_build_units.py --kind gbpe --vocab {vocab.value} --inventory {OUT} "
        f"--out {OUT}/gbpe{vocab.value} --manifests {_m}/train.jsonl {_m}/test.jsonl"
        + (" --compare-gemma" if compare.value else ""))
    mo.md(f"units in `{OUT}/gbpe{vocab.value}`")
    return CODE, OUT


@app.cell
def _(CODE, OUT, json, mo, os, vocab):
    with open(os.path.join(OUT, f"gbpe{vocab.value}", "report.json")) as _f:
        _r = json.load(_f)
    with open(os.path.join(CODE, "reference/grapheme/units/gbpe3000_report.json")) as _f:
        _old = json.load(_f)

    def _row(name, r):
        _tr = next(v for k, v in r["stats"].items() if k.endswith("train.jsonl"))
        _v = r["stats"]["vocab"]
        _single = r.get("unit_len_hist", {}).get("1", r.get("unit_len_hist", {}).get(1, "?"))
        return (f"| {name} | {r['units']:,} | {_tr['units_per_word']:.2f} | {_tr['graphemes_per_unit_excl_space']:.2f} | "
                f"{_tr['single_unit_words']:.1%} | {_single} | {_v['rare_lt20_share']:.1%} | "
                f"{_v.get('merged_rare_lt20_share', float('nan')):.1%} | {_tr['roundtrip_failures']} |")

    mo.md("| units | size | units/word (train) | graphemes/unit | single-unit words | 1-grapheme pieces | "
          "units < 20× | merged < 20× | round-trip fails |\n|---|---|---|---|---|---|---|---|---|\n"
          + _row("old C gbpe3000 (web-fitted)", _old) + "\n" + _row(f"new gbpe{vocab.value} (speech-fitted)", _r)
          + "\n\nGo on if round-trip fails = 0 and units/word drops well below 1.99. Then upload.")
    return


@app.cell
def _(mo):
    up = mo.ui.run_button(label="Upload to the bucket as units_speech/")
    up
    return (up,)


@app.cell
def _(BUCKET, OUT, mo, subprocess, up, vocab):
    mo.stop(not up.value)
    for _src, _dst in ((f"{OUT}/gbpe{vocab.value}", f"units_speech/gbpe{vocab.value}"), (OUT, "units_speech")):
        _flags = [] if _src != OUT else ["--include", "graphemes.json"]
        subprocess.run(["hf", "buckets", "sync", _src, f"hf://buckets/{BUCKET}/{_dst}", *_flags], check=True)
    mo.md(f"uploaded: `hf://buckets/{BUCKET}/units_speech/gbpe{vocab.value}` (+ graphemes.json). "
          f"The C presets in molab_train.py expect `units_speech/gbpe3000`; for another size edit the preset.")
    return


if __name__ == "__main__":
    app.run()
