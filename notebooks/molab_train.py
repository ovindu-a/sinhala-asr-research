# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.9",
#     "transformers>=5.19",
#     "huggingface_hub[cli]>=1.25",
#     "safetensors",
#     "numpy",
#     "soundfile",
#     "soxr",
#     "jiwer",
#     "rapidfuzz",
#     "tqdm",
#     "pyarrow",
#     "sentencepiece",
#     "grapheme-kit==0.1.1",
#     "quan-transcribe @ git+https://github.com/qnguyen3/QUAN-Transcribe",
# ]
# ///
"""Train / continue a CTC reader on molab (GPU), with data and checkpoints in the private HF bucket.

Presets: B for longer (flat head or + head parts), C redo (speech-fitted grapheme-BPE) v6_top6 / v6x_top8.
See docs/MOLAB.md. Run the numbered buttons in order; every step can be rerun after a restart, and training
resumes from the bucket's runs/<name>/last.
"""

import marimo

__generated_with = "0.25.2"
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

    return json, mo, os, shutil, subprocess, sys, urllib, zipfile


@app.cell
def _(mo):
    mo.md("""
    # Sinhala CTC training on molab

    Data, starting checkpoints and results live in the private bucket `hf://buckets/ovindu-a/sinhala-grapheme-asr`
    (needs `HF_TOKEN` with read/write access). Checkpoints are synced back to the bucket after every save, so a
    session that ends can be resumed: run the same preset again.
    """)
    return


@app.cell
def _():
    BUCKET = "ovindu-a/sinhala-grapheme-asr"
    REPO_ZIP = "https://codeload.github.com/ovindu-a/sinhala-asr-research/zip/refs/heads/grapheme-units"
    WORK = "/tmp/sinhala"  # local disk of the session; nothing here survives it except what is synced
    PRESETS = {
        "B longer (flat head)": dict(
            name="gr_v6x_top8_long", manifests="v6_plus_slr52", init="gr_v6x_top8", units="", top_k=8,
            steps=6000, lr=5e-5, warmup=0.03, head_parts=False),
        "B longer + head parts": dict(
            name="gr_v6x_top8_long_parts", manifests="v6_plus_slr52", init="gr_v6x_top8", units="", top_k=8,
            steps=6000, lr=5e-5, warmup=0.03, head_parts=True),
        "C redo: v6_top6 (speech-fitted gbpe3000)": dict(
            name="gbs3k_v6_top6", manifests="ss_v6", init="", units="units_speech/gbpe3000", top_k=6,
            steps=7500, lr=2e-4, warmup=0.05, head_parts=False),
        "C redo: v6x_top8": dict(
            name="gbs3k_v6x_top8", manifests="v6_plus_slr52", init="gbs3k_v6_top6", units="", top_k=8,
            steps=6000, lr=1e-4, warmup=0.03, head_parts=False),
    }
    return BUCKET, PRESETS, REPO_ZIP, WORK


@app.cell
def _(PRESETS, mo):
    preset = mo.ui.dropdown(options=list(PRESETS), value="B longer (flat head)", label="Experiment")
    preset
    return (preset,)


@app.cell
def _(PRESETS, mo, preset):
    _p = PRESETS[preset.value]
    run_name = mo.ui.text(value=_p["name"], label="run name (bucket runs/<name>)")
    steps = mo.ui.number(start=100, stop=50000, step=100, value=_p["steps"], label="steps")
    lr = mo.ui.number(start=1e-6, stop=1e-3, step=1e-6, value=_p["lr"], label="peak lr")
    workers = mo.ui.number(start=0, stop=16, step=1, value=3, label="data workers (molab: 4 CPUs)")
    max_minutes = mo.ui.number(start=10, stop=700, step=10, value=660, label="stop + checkpoint after (min)")
    mo.vstack([run_name, steps, lr, workers, max_minutes,
               mo.md(f"init from: `{_p['init'] or '(scratch)'}` · units: `{_p['units'] or '(from init)'}` · "
                     f"data: `{_p['manifests']}` · unfrozen top {_p['top_k']} · head parts: {_p['head_parts']}")])
    return lr, max_minutes, run_name, steps, workers


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF_TOKEN (leave empty if set in the environment)")
    hf_token
    return (hf_token,)


@app.cell
def _(mo):
    setup_btn = mo.ui.run_button(label="1. Set up: code + GPU check")
    setup_btn
    return (setup_btn,)


@app.cell
def _(
    REPO_ZIP,
    WORK,
    hf_token,
    mo,
    os,
    setup_btn,
    shutil,
    sys,
    urllib,
    zipfile,
):
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
    for _p in (CODE, os.path.join(CODE, "scripts")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import torch

    assert torch.cuda.is_available(), "no GPU: attach one to this notebook"
    _x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    (_x @ _x).sum().item()  # fails here if this torch build has no kernels for the GPU
    _free = shutil.disk_usage(WORK).free / 1e9
    mo.md(f"code: `{CODE}` · GPU: **{torch.cuda.get_device_name()}** (sm_{''.join(map(str, torch.cuda.get_device_capability()))}, "
          f"torch {torch.__version__}) · free disk: **{_free:.0f} GB** (v6_plus_slr52 needs ~25 GB, ss_v6 ~16 GB)")
    return (CODE,)


@app.cell
def _(BUCKET, subprocess):
    def bucket_sync(src: str, dst: str, *flags: str):
        """hf buckets sync; remote paths are relative to the bucket root."""
        _fix = lambda p: p if p.startswith("/") else f"hf://buckets/{BUCKET}/{p}"  # noqa: E731
        cmd = ["hf", "buckets", "sync", _fix(src), _fix(dst), *flags]
        print("+", " ".join(cmd), flush=True)
        return subprocess.run(cmd, check=False).returncode == 0

    return (bucket_sync,)


@app.cell
def _(mo):
    data_btn = mo.ui.run_button(label="2. Download data + starting checkpoint")
    data_btn
    return (data_btn,)


@app.cell
def _(CODE, PRESETS, WORK, bucket_sync, data_btn, mo, os, preset, run_name):
    mo.stop(not data_btn.value, mo.vstack([mo.md("**Step 2** (after step 1 has finished):"), data_btn]))
    _p = PRESETS[preset.value]
    _m = _p["manifests"]
    D = os.path.join(WORK, "data")
    # training split of the preset; dev2k / test always from ss_v6 (v6_plus_slr52 uses the same ones)
    assert bucket_sync(f"data/packed/{_m}", f"{D}/packed/{_m}", "--include", "train.*")
    assert bucket_sync("data/packed/ss_v6", f"{D}/packed/ss_v6", "--include", "dev2k.*")
    assert bucket_sync("data/packed/ss_v6", f"{D}/packed/ss_v6", "--include", "test.*")
    assert bucket_sync(f"data/manifests/{_m}", f"{D}/manifests/{_m}", "--include", "train.jsonl")
    assert bucket_sync("data/manifests/ss_v6", f"{D}/manifests/ss_v6")
    RUNS = os.path.join(WORK, "runs")
    # a sync of a missing bucket folder copies nothing and still succeeds, so check the files themselves
    if _p["init"]:
        bucket_sync(f"runs/{_p['init']}/best", f"{RUNS}/{_p['init']}/best")
        assert os.path.exists(f"{RUNS}/{_p['init']}/best/model.safetensors"), (
            f"starting checkpoint runs/{_p['init']}/best is not in the bucket: run that preset (steps 1-3) first")
    if _p["units"]:
        bucket_sync(_p["units"], f"{D}/{_p['units']}")
        assert os.path.exists(f"{D}/{_p['units']}/units.json"), (
            f"{_p['units']}/units.json is not in the bucket: run notebooks/molab_units.py and press its Upload first")
    # resume: whatever this run already has in the bucket (last/, best/, metrics.jsonl)
    bucket_sync(f"runs/{run_name.value}", f"{RUNS}/{run_name.value}")
    _have = os.path.exists(f"{RUNS}/{run_name.value}/last/state.pt")
    _sizes = {f: os.path.getsize(os.path.join(dp, f)) / 1e9 for dp, _, fs in os.walk(D) for f in fs if f.endswith(".pcm")}
    mo.md(f"data in `{D}`: " + ", ".join(f"{k} {v:.1f} GB" for k, v in _sizes.items())
          + f" · resume: **{'yes, from bucket last/' if _have else 'no, fresh run'}** · code `{CODE}`")
    return D, RUNS


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="3. Train (blocks this notebook; checkpoints go to the bucket)")
    train_btn
    return (train_btn,)


@app.cell
def _(
    D,
    PRESETS,
    RUNS,
    bucket_sync,
    lr,
    max_minutes,
    mo,
    os,
    preset,
    run_name,
    steps,
    train_btn,
    workers,
):
    mo.stop(not train_btn.value, mo.vstack([mo.md("**Step 3** (after step 2 has finished):"), train_btn]))
    import train as _train_mod

    _p = PRESETS[preset.value]
    _m, _out = _p["manifests"], f"{RUNS}/{run_name.value}"
    _pk = os.path.join(D, "packed_link")  # train.py reads <packed-dir>/{train,dev}.{pcm,json}
    os.makedirs(_pk, exist_ok=True)
    for _split, _src in (("train", f"{D}/packed/{_m}/train"), ("dev", f"{D}/packed/ss_v6/dev2k")):
        for _ext in (".pcm", ".json"):
            _dst = f"{_pk}/{_split}{_ext}"
            if os.path.lexists(_dst):
                os.remove(_dst)
            os.symlink(_src + _ext, _dst)
    _argv = [
        "--train", f"{D}/manifests/{_m}/train.jsonl", "--dev", f"{D}/manifests/ss_v6/dev2k.jsonl",
        "--packed-dir", _pk, "--out", _out, "--unfreeze-top-k", str(_p["top_k"]),
        "--steps", str(int(steps.value)), "--lr", str(lr.value), "--warmup", str(_p["warmup"]),
        "--batch-seconds", "1000", "--eval-every", "1000", "--ckpt-every", "1000",
        "--workers", str(int(workers.value)), "--resume", "--max-minutes", str(max_minutes.value),
    ]
    if _p["init"]:
        _argv += ["--init-from", f"{RUNS}/{_p['init']}/best"]
    if _p["units"]:
        _argv += ["--units", f"{D}/{_p['units']}/units.json"]
    if _p["head_parts"]:
        _argv += ["--head-parts"]
    print("train.py", " ".join(_argv), flush=True)

    def _push():  # after every checkpoint / best / metrics flush
        bucket_sync(_out, f"runs/{run_name.value}")

    _train_mod.main(_argv, on_checkpoint=_push)
    _push()
    mo.md(f"training finished; `runs/{run_name.value}` synced to the bucket")
    return


@app.cell
def _(mo):
    eval_btn = mo.ui.run_button(label="4. Test eval (greedy) + log-prob dump for KenLM decoding")
    eval_btn
    return (eval_btn,)


@app.cell
def _(D, RUNS, bucket_sync, eval_btn, mo, run_name):
    mo.stop(not eval_btn.value, mo.vstack([mo.md("**Step 4** (after training):"), eval_btn]))
    import importlib as _importlib

    import evaluate as _eval_mod

    _ck = f"{RUNS}/{run_name.value}/best"
    _res = _eval_mod.main(["--ckpt", _ck, "--manifest", f"{D}/manifests/ss_v6/test.jsonl",
                          "--packed", f"{D}/packed/ss_v6/test", "--out", f"{_ck}/eval_ss_v6_test", "--workers", "3"])
    _lmd = _importlib.import_module("08_lm_decode")
    _lmd.main(["dump", "--ckpt", _ck, "--split", "dev2k", "--split", "test", "--manifests", f"{D}/manifests/ss_v6",
               "--packed", f"{D}/packed/ss_v6", "--out", f"{_ck}/lm"])
    bucket_sync(_ck, f"runs/{run_name.value}/best")
    _o = _res["overall"]
    mo.md(f"**{run_name.value}** greedy v6 test: WER {_o['wer']:.2%} · WER_seg {_o['wer_seg']:.2%} · "
          f"CER {_o['cer']:.2%} · GER {_o['ger']:.2%} ({_o['utts']} utts). Log-probs in `{_ck}/lm`, synced. "
          f"For KenLM decoding run `upload_hf_bucket.py --what pull --run {run_name.value}` and `lm_decode` on Modal "
          f"(docs/MOLAB.md). Reference, B greedy: WER 19.58 · WER_seg 11.71 · CER 4.32 · GER 5.40.")
    return


@app.cell
def _(CODE, RUNS, json, mo, os, run_name):
    def _curve(path):
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return {d["step"]: d["eval"]["dev"]["cer"] for d in map(json.loads, f) if "eval" in d}

    _new = _curve(f"{RUNS}/{run_name.value}/metrics.jsonl")
    _ref = _curve(f"{CODE}/reference/grapheme/training/gr_v6x_top8/metrics.jsonl")
    _rows = "\n".join(f"| {s} | {_ref.get(s, float('nan')):.2%} | {_new.get(s, float('nan')):.2%} |"
                      for s in sorted(set(_new) | set(_ref)))
    mo.md("Dev2k CER by step (rerun this cell to refresh)\n\n| step | B gr_v6x_top8 | this run |\n|---|---|---|\n" + _rows)
    return


if __name__ == "__main__":
    app.run()
