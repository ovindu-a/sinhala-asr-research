"""Run data prep and training on Modal.

  modal run modal_app.py::pack_v6                                  # SinhaSpeech v6 parquet -> packed PCM (CPU)
  modal run --detach modal_app.py::train --name v6_top2 --steps 8000  # resumes automatically from /runs/v6_top2/last
  modal run modal_app.py::evaluate --ckpt v6_top2/best --split test
  modal volume get sinhala-asr-runs v6_top2/metrics.jsonl runs/v6_top2/

(step-1 probe functions `prep`/`pack` built 20 h OpenSLR manifests from the HF mirror)
"""

import os
import subprocess

import modal

GPU = "H100"
data_vol = modal.Volume.from_name("sinhala-asr-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("sinhala-asr-runs", create_if_missing=True)
hf_vol = modal.Volume.from_name("sinhala-asr-hf-cache", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1", "unzip", "curl")
    .uv_pip_install(
        "torch>=2.6",
        "transformers>=5.19",
        "huggingface_hub",
        "safetensors",
        "numpy",
        "soundfile",
        "soxr",
        "jiwer",
        "tqdm",
        "pyarrow",
        "quan-transcribe @ git+https://github.com/qnguyen3/QUAN-Transcribe",
    )
    .env({"HF_HOME": "/hf", "PYTHONPATH": "/root/app", "PYTHONUNBUFFERED": "1"})
)
# local code must be the last layer of an image
image = base_image.add_local_python_source("sinhala_asr").add_local_dir("scripts", "/root/app/scripts")

app = modal.App("sinhala-asr", image=image)
VOLS = {"/data": data_vol, "/runs": runs_vol, "/hf": hf_vol}


def sh(cmd: str):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/root/app")


@app.function(volumes=VOLS, cpu=8, memory=32768, timeout=2 * 3600)
def prep(files: int = 4, train_hours: float = 20.0, out: str = "openslr_probe"):
    """OpenSLR-52 from the HF parquet mirror (openslr.org itself is too slow)."""
    hours = f"--train-hours {train_hours}" if train_hours else ""
    try:
        sh(f"python scripts/03b_prep_openslr_hf.py --files {files} --audio-dir /data/openslr52_hf "
           f"--out /data/manifests/{out} {hours}")
    finally:
        data_vol.commit()
        hf_vol.commit()


@app.function(volumes=VOLS, cpu=8, memory=16384, timeout=3600)
def pack(manifests: str = "openslr_probe"):
    """One int16 array per split, so the GPU job reads 2 files instead of ~18k small FLACs."""
    try:
        for split in ("train", "dev"):
            sh(f"python scripts/04_pack_audio.py --manifest /data/manifests/{manifests}/{split}.jsonl "
               f"--out /data/packed/{manifests}/{split}")
    finally:
        data_vol.commit()


@app.function(volumes=VOLS, cpu=8, memory=16384, timeout=2 * 3600)
def pack_v6(splits: str = "train,dev2k,test"):
    """SinhaSpeech stratified_v6 (speaker-disjoint) straight from parquet into packed PCM; no per-file audio."""
    spec = {
        "train": ("train", "--drop-digits"),
        "dev2k": ("validation", "--subsample 2000"),  # periodic eval during training
        "validation": ("validation", ""),
        "test": ("test", ""),  # keep every row (incl. digits) so scores compare to published ones
    }
    try:
        for name in splits.split(","):
            src, flags = spec[name]
            sh(f"python scripts/05_pack_parquet.py --file data/stratified_v6/{src}.parquet "
               f"--out /data/packed/ss_v6/{name} --manifest /data/manifests/ss_v6/{name}.jsonl {flags}")
            data_vol.commit()
    finally:
        data_vol.commit()


@app.function(volumes=VOLS, cpu=8, memory=32768, timeout=2 * 3600)
def slr52_extra(out: str = "v6_plus_slr52", dry_run: bool = False):
    """SLR-52 utterances missing from v6, minus v6 val/test speakers; merged after v6 train (see script)."""
    try:
        sh(f"python scripts/06_slr52_extra.py --v6-train-manifest /data/manifests/ss_v6/train.jsonl "
           f"--v6-packed /data/packed/ss_v6/train --out {out} {'--dry-run' if dry_run else ''}")
    finally:
        data_vol.commit()


def _scripts():
    import sys

    sys.path.insert(0, "/root/app/scripts")
    import evaluate as ev_mod
    import train as train_mod

    return train_mod, ev_mod


@app.function(volumes=VOLS, gpu=GPU, cpu=8, memory=49152, timeout=3 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def train(name: str, manifests: str = "ss_v6", dev: str = "dev2k", unfreeze_top_k: int = 2, steps: int = 8000,
          max_minutes: float = 165, extra: str = ""):
    """Always --resume: a preempted/retried container continues from /runs/<name>/last."""
    train_mod, _ = _scripts()
    m, p = f"/data/manifests/{manifests}", f"/data/packed/{manifests}"
    # train.py reads <packed>/train and <packed>/dev; point "dev" at the chosen dev split
    os.makedirs(f"/tmp/packed", exist_ok=True)
    for split, src in (("train", "train"), ("dev", dev)):
        for ext in (".pcm", ".json"):
            dst = f"/tmp/packed/{split}{ext}"
            if not os.path.exists(dst):
                os.symlink(f"{p}/{src}{ext}", dst)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    argv = (f"--train {m}/train.jsonl --dev {m}/{dev}.jsonl --packed-dir /tmp/packed --out /runs/{name} "
            f"--unfreeze-top-k {unfreeze_top_k} --steps {steps} --workers 7 --resume "
            f"--max-minutes {max_minutes} {extra}").split()
    try:
        train_mod.main(argv, on_checkpoint=runs_vol.commit)
    finally:
        runs_vol.commit()
        hf_vol.commit()


@app.function(volumes=VOLS, gpu=GPU, cpu=8, memory=32768, timeout=3600)
def evaluate(ckpt: str, split: str = "test", manifests: str = "ss_v6"):
    """ckpt: path on the runs volume, e.g. v6_top2/best. Writes /runs/<ckpt>/eval_<manifests>_<split>.{json,hyps.jsonl}."""
    _, ev_mod = _scripts()
    try:
        ev_mod.main([f"--ckpt", f"/runs/{ckpt}", "--manifest", f"/data/manifests/{manifests}/{split}.jsonl",
                     "--packed", f"/data/packed/{manifests}/{split}", "--out", f"/runs/{ckpt}/eval_{manifests}_{split}"])
    finally:
        runs_vol.commit()


@app.function(volumes=VOLS, gpu=GPU, cpu=8, memory=32768, timeout=3600)
def eval_whisper(model: str = "SinhaSpeech/whisper-small-sinhala-run12-best", split: str = "test"):
    """Published Whisper-small baseline, scored with our normalization + metrics on the same packed test set."""
    import sys

    sys.path.insert(0, "/root/app/scripts")
    import eval_whisper as ew

    try:
        ew.main(["--model", model, "--manifest", f"/data/manifests/ss_v6/{split}.jsonl",
                 "--packed", f"/data/packed/ss_v6/{split}", "--out", f"/runs/baselines/{model.split('/')[-1]}_{split}"])
    finally:
        runs_vol.commit()
        hf_vol.commit()


lm_image = (
    base_image.apt_install("build-essential", "cmake", "libboost-program-options-dev", "libboost-system-dev",
                      "libboost-thread-dev", "libboost-test-dev", "zlib1g-dev", "libbz2-dev", "liblzma-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/kpu/kenlm /opt/kenlm && cd /opt/kenlm && mkdir build && cd build "
        "&& cmake .. -DKENLM_MAX_ORDER=6 && make -j8 lmplz build_binary && cp bin/lmplz bin/build_binary /usr/local/bin/",
    )
    .uv_pip_install("https://github.com/kpu/kenlm/archive/master.zip", "pyahocorasick")
    .add_local_python_source("sinhala_asr")
    .add_local_dir("scripts", "/root/app/scripts")
)


@app.function(image=lm_image, volumes=VOLS, cpu=8, memory=65536, timeout=3 * 3600)
def build_lm(out: str = "si_4gram", train_manifests: str = "v6_plus_slr52", max_web_words: int = 150_000_000):
    try:
        sh(f"python scripts/07_build_lm.py --out /data/lm/{out} "
           f"--train-manifests /data/manifests/{train_manifests}/train.jsonl "
           f"--exclude /data/manifests/ss_v6/test.jsonl /data/manifests/ss_v6/dev2k.jsonl "
           f"--max-web-words {max_web_words}")
    finally:
        data_vol.commit()


@app.function(image=lm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=32768, timeout=3600)
def lm_dump(ckpt: str, manifests: str = "ss_v6"):
    try:
        sh(f"python scripts/08_lm_decode.py dump --ckpt /runs/{ckpt} --split dev2k --split test "
           f"--manifests /data/manifests/{manifests} --packed /data/packed/{manifests} --out /runs/{ckpt}/lm")
    finally:
        runs_vol.commit()


@app.function(image=lm_image, volumes=VOLS, cpu=32, memory=65536, timeout=3 * 3600)
def lm_decode(ckpt: str, lm: str = "si_4gram", extra: str = ""):
    try:
        sh(f"python scripts/08_lm_decode.py decode --ckpt /runs/{ckpt} --lm /data/lm/{lm} --out /runs/{ckpt}/lm {extra}")
    finally:
        runs_vol.commit()
