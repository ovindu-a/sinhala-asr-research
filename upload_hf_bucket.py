"""Copy grapheme-unit checkpoints + logits + units from the Modal volumes to a private HF Storage Bucket.
  modal run upload_hf_bucket.py                 # dry run (lists what would be uploaded)
  modal run upload_hf_bucket.py --no-dry-run    # upload
"""
import subprocess

import modal

BUCKET = "ovindu-a/sinhala-grapheme-asr"
RUNS = ["gr_v6_top6", "gr_v6x_top8", "gb3k_v6_top6", "gb3k_v6x_top8"]

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub[cli]>=1.25")
app = modal.App("sinhala-hf-bucket", image=image)
runs = modal.Volume.from_name("sinhala-asr-runs")
data = modal.Volume.from_name("sinhala-asr-data")


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


@app.function(volumes={"/runs": runs, "/data": data}, secrets=[modal.Secret.from_name("huggingface")],
              cpu=4, memory=8192, timeout=3 * 3600)
def upload(dry_run: bool = True):
    sh("hf version && hf buckets --help")
    sh("hf auth whoami")
    sh(f"hf buckets create {BUCKET} --private --exist-ok")
    flag = "--dry-run" if dry_run else ""
    for r in RUNS:  # best/ only (weights, config, units, eval files, lm/ logits); never last/
        sh(f"hf buckets sync /runs/{r}/best hf://buckets/{BUCKET}/runs/{r}/best {flag}")
    # words.tsv (29 MB word list) excluded; graphemes.json, grapheme/, gbpe3000/ only
    sh(f"hf buckets sync /data/units hf://buckets/{BUCKET}/units --exclude 'words.tsv' {flag}")


@app.local_entrypoint()
def main(dry_run: bool = True):
    upload.remote(dry_run=dry_run)
