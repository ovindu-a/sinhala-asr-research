# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers>=5.19",
#     "huggingface_hub[cli]>=1.25",
#     "numpy",
#     "jiwer",
#     "rapidfuzz",
#     "sentencepiece",
#     "grapheme-kit==0.1.1",
#     "quan-transcribe @ git+https://github.com/qnguyen3/QUAN-Transcribe",
# ]
# ///
"""KenLM beam-search decoding on molab (CPU only): a run's dumped log-probs + the grapheme 8-gram or the word 4-gram.

Inputs from the private bucket (runs/<run>/best/{units,config}.json + lm/*.logits.pt from molab_train.py step 4,
lm/si_graph8) or the public HF repo ovindu-a/sinhala-gemma-asr (lm/ = word 4-gram). Outputs go back to
runs/<run>/best/lm/eval_lm_test<tag>.* in the bucket. 4 CPUs: ~1 utterance / s / CPU, so decoding all 15,860 test
utterances once takes ~1 h; each grid setting adds 2,000 dev utterances (~8 min). See docs/MOLAB.md.
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
    import time
    import urllib.request
    import zipfile

    import marimo as mo

    return json, mo, os, shutil, subprocess, sys, time, urllib, zipfile


@app.cell
def _(mo):
    BUCKET = "ovindu-a/sinhala-grapheme-asr"
    REPO_ZIP = "https://codeload.github.com/ovindu-a/sinhala-asr-research/zip/refs/heads/grapheme-units"
    WORK = "/tmp/sinhala_decode"
    run = mo.ui.dropdown(
        options=["gr_v6x_top8_long", "gr_v6x_top8_long_parts", "gr_v6x_top8", "gbs3k_v6x_top8", "gb3k_v6x_top8"],
        value="gr_v6x_top8_long", label="run (bucket runs/<run>/best)")
    lm_kind = mo.ui.dropdown(options=["grapheme 8-gram (graphemes runs: gr_*)", "word 4-gram (any run)"],
                             value="grapheme 8-gram (graphemes runs: gr_*)", label="language model")
    mode = mo.ui.dropdown(
        options=["quick: B's best weights, test only (~1 h)", "grid on all 2,000 dev, then test"],
        value="quick: B's best weights, test only (~1 h)", label="mode")
    hf_token = mo.ui.text(kind="password", label="HF_TOKEN (leave empty if set in the environment)")
    mo.vstack([run, lm_kind, mode, hf_token])
    return BUCKET, REPO_ZIP, WORK, hf_token, lm_kind, mode, run


@app.cell
def _(lm_kind, mo, mode):
    _g = lm_kind.value.startswith("grapheme")
    _quick = mode.value.startswith("quick")
    # B's dev-tuned weights: grapheme 8-gram alpha 0.5 beta 1.0 (§13); word 4-gram alpha 0.35 beta 1.0
    alphas = mo.ui.text(value=("0.5" if _g else "0.35") if _quick else ("0.3 0.5 0.7" if _g else "0.2 0.35 0.5"),
                        label="alphas (LM weight)")
    betas = mo.ui.text(value="1.0" if _quick else ("0.5 1.0" if _g else "0.5 1.0 2.0"), label="betas (insertion bonus)")
    tag = mo.ui.text(value=("_glm8" if _g else "_wlm") + ("_quick" if _quick else ""), label="output tag")
    mo.vstack([alphas, betas, tag])
    return alphas, betas, tag


@app.cell
def _(alphas, betas, mo, mode):
    _n = len(alphas.value.split()) * len(betas.value.split())
    _dev = 200 if mode.value.startswith("quick") else 2000
    _min = (_n * _dev + 15860) / 4 / 60
    mo.md(f"{_n} setting(s) × {_dev} dev utts + 15,860 test utts ≈ **{_min:.0f} min** on 4 CPUs (rough)")
    return


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1. Set up: code + KenLM (order 10)")
    setup_btn
    return (setup_btn,)


@app.cell
def _(REPO_ZIP, WORK, hf_token, mo, os, setup_btn, shutil, subprocess, sys, urllib, zipfile):
    mo.stop(not setup_btn.value, mo.vstack([mo.md("**Step 1:** click to start"), setup_btn]))
    if hf_token.value:
        os.environ["HF_TOKEN"] = hf_token.value
    assert os.environ.get("HF_TOKEN"), "set HF_TOKEN (field above or the environment)"
    os.makedirs(WORK, exist_ok=True)
    _zip = os.path.join(WORK, "code.zip")
    urllib.request.urlretrieve(REPO_ZIP, _zip)
    CODE = os.path.join(WORK, "code")
    shutil.rmtree(CODE, ignore_errors=True)
    with zipfile.ZipFile(_zip) as _z:
        _z.extractall(WORK)
    os.replace(os.path.join(WORK, "sinhala-asr-research-grapheme-units"), CODE)
    # the stock kenlm wheel/build supports n-grams up to order 6; the grapheme LM is order 8
    _env = os.environ | {"MAX_ORDER": "10"}
    _url = "https://github.com/kpu/kenlm/archive/master.zip"
    _r = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", _url], env=_env)
    if _r.returncode != 0:  # venvs without pip
        _r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--reinstall", "--no-deps", _url],
                            env=_env)
    assert _r.returncode == 0, "kenlm build failed (needs a C++ compiler); decode on Modal instead (docs/MOLAB.md §3)"
    mo.md(f"code: `{CODE}` · kenlm built with MAX_ORDER=10 · CPUs: {os.cpu_count()}")
    return (CODE,)


@app.cell
def _(mo):
    data_btn = mo.ui.run_button(label="2. Download log-probs + LM")
    data_btn
    return (data_btn,)


@app.cell
def _(BUCKET, CODE, WORK, data_btn, lm_kind, mo, os, run, subprocess, sys):
    mo.stop(not data_btn.value, mo.vstack([mo.md("**Step 2** (after step 1 has finished):"), data_btn]))

    def _sync(src, dst, *flags):
        subprocess.run(["hf", "buckets", "sync", f"hf://buckets/{BUCKET}/{src}", dst, *flags], check=False)

    CK = f"{WORK}/runs/{run.value}/best"
    for _f in ("units.json", "config.json"):  # not model.safetensors: decoding only needs the log-probs
        _sync(f"runs/{run.value}/best", CK, "--include", _f)
    _sync(f"runs/{run.value}/best/lm", f"{CK}/lm", "--include", "*.logits.pt")
    for _f in ("units.json", "lm/dev2k.logits.pt", "lm/test.logits.pt"):
        assert os.path.exists(f"{CK}/{_f}"), (f"runs/{run.value}/best/{_f} not in the bucket: "
                                              "run molab_train.py step 4 for this run first")
    if lm_kind.value.startswith("grapheme"):
        LM = f"{WORK}/lm/si_graph8"
        _sync("lm/si_graph8", LM)
    else:
        from huggingface_hub import hf_hub_download

        LM = f"{WORK}/lm/si_4gram"
        os.makedirs(LM, exist_ok=True)
        for _f in ("lm.bin", "unigrams.txt", "lm_config.json"):
            os.replace(hf_hub_download("ovindu-a/sinhala-gemma-asr", f"lm/{_f}", local_dir=f"{WORK}/hf"), f"{LM}/{_f}")
    assert os.path.exists(f"{LM}/lm.bin"), "LM download failed"
    # loads the LM once: fails here (not after an hour) if kenlm was built without order 8
    subprocess.run([sys.executable, "-c", f"import kenlm; m = kenlm.Model('{LM}/lm.bin'); print('LM order', m.order)"],
                   check=True)
    _train = f"{WORK}/train.jsonl"
    if not os.path.exists(_train):
        subprocess.run(f"gunzip -c {CODE}/reference/manifests/ss_v6/train.jsonl.gz > {_train}", shell=True, check=True)
    mo.md(f"log-probs: `{CK}/lm` · LM: `{LM}` ({os.path.getsize(f'{LM}/lm.bin') / 1e9:.2f} GB)")
    return CK, LM


@app.cell
def _(mo):
    decode_btn = mo.ui.run_button(label="3. Decode (blocks this notebook)")
    decode_btn
    return (decode_btn,)


@app.cell
def _(BUCKET, CK, CODE, LM, alphas, betas, decode_btn, lm_kind, mo, mode, os, run, subprocess, sys, tag, time):
    mo.stop(not decode_btn.value, mo.vstack([mo.md("**Step 3** (after step 2 has finished):"), decode_btn]))
    _quick = mode.value.startswith("quick")
    _cmd = [sys.executable, "scripts/08_lm_decode.py", "decode", "--ckpt", CK, "--lm", LM, "--out", f"{CK}/lm",
            "--tune-limit", "200" if _quick else "2000", "--alphas", *alphas.value.split(),
            "--betas", *betas.value.split(), "--tag", tag.value, "--procs", str(os.cpu_count())]
    if lm_kind.value.startswith("grapheme"):
        _cmd += ["--lm-kind", "grapheme"]
    print(" ".join(_cmd), flush=True)
    _t = time.time()
    subprocess.run(_cmd, cwd=CODE, env=os.environ | {"PYTHONPATH": CODE}, check=True)
    subprocess.run(["hf", "buckets", "sync", f"{CK}/lm", f"hf://buckets/{BUCKET}/runs/{run.value}/best/lm",
                    "--include", f"eval_lm_test{tag.value}*"], check=True)
    mo.md(f"decoded in {(time.time() - _t) / 60:.0f} min; `eval_lm_test{tag.value}.*` synced to the bucket")
    return


@app.cell
def _(CK, CODE, WORK, json, mo, os, subprocess, sys, tag):
    _hyps = f"{CK}/lm/eval_lm_test{tag.value}.hyps.jsonl"
    mo.stop(not os.path.exists(_hyps), mo.md("Results appear here after step 3."))
    with open(f"{CK}/lm/eval_lm_test{tag.value}.json") as _f:
        _r = json.load(_f)
    _out = subprocess.run([sys.executable, "scripts/rescore.py", _hyps, "--train", f"{WORK}/train.jsonl"], cwd=CODE,
                          env=os.environ | {"PYTHONPATH": CODE}, capture_output=True, text=True).stdout
    _lines = [l for l in _out.splitlines() if any(f" {k} " in l for k in ("all", "seen", "unseen"))]
    mo.md(f"best on dev: α {_r['alpha']}, β {_r['beta']} ({_r.get('tune_utts')} dev utts)\n\n```\n"
          + "\n".join(_lines) + "\n```\n\nReference (test, all / unseen WER): B + grapheme 8-gram 14.26 / 19.23; "
          "A + word 4-gram 14.72 / 20.57.")
    return


if __name__ == "__main__":
    app.run()
