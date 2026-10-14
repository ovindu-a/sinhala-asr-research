"""Run data prep and training on Modal.

  modal run modal_app.py::prep --files 4 --train-hours 20         # OpenSLR-52 (HF mirror) -> volume, manifests
  modal run modal_app.py::pack                                     # pack audio for fast loading
  modal run modal_app.py::train --name probe_frozen --unfreeze-top-k 0 --steps 3000
  modal run modal_app.py::train --name probe_top2 --unfreeze-top-k 2 --steps 3000
  modal volume get sinhala-asr-runs probe_frozen ./runs/          # fetch metrics + checkpoints
"""

import subprocess

import modal

GPU = "A100-80GB"
data_vol = modal.Volume.from_name("sinhala-asr-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("sinhala-asr-runs", create_if_missing=True)
hf_vol = modal.Volume.from_name("sinhala-asr-hf-cache", create_if_missing=True)

image = (
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
    .add_local_python_source("sinhala_asr")
    .add_local_dir("scripts", "/root/app/scripts")
)

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


@app.function(volumes=VOLS, gpu=GPU, cpu=8, memory=32768, timeout=90 * 60)  # hard stop caps cost per run
def train(name: str, manifests: str = "openslr_probe", unfreeze_top_k: int = 0, steps: int = 3000,
          extra: str = ""):
    m = f"/data/manifests/{manifests}"
    try:
        sh(f"nvidia-smi --query-gpu=name,memory.total --format=csv && python scripts/train.py "
           f"--train {m}/train.jsonl --dev {m}/dev.jsonl --out /runs/{name} "
           f"--unfreeze-top-k {unfreeze_top_k} --steps {steps} --workers 7 "
           f"--packed-dir /data/packed/{manifests} {extra}")
    finally:
        runs_vol.commit()
        hf_vol.commit()
