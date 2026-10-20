"""Copy grapheme-unit checkpoints + logits + units from the Modal volumes to a private HF Storage Bucket.
  modal run upload_hf_bucket.py                 # dry run (lists what would be uploaded)
  modal run upload_hf_bucket.py --no-dry-run    # upload
  modal run upload_hf_bucket.py --what glm8 [--no-dry-run]   # grapheme 8-gram, B's *_glm8* eval files, si_corpus text
  modal run upload_hf_bucket.py --what data [--no-dry-run]   # packed audio + manifests for molab training (~33 GB)
  modal run upload_hf_bucket.py --what pull --run gr_v6x_top8_long [--no-dry-run]
      # the other way: a run trained on molab (bucket runs/<run>/best) -> /runs/<run>/best, for lm_decode on Modal
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
def upload(dry_run: bool = True, what: str = "ckpts", run: str = ""):
    sh("hf version && hf buckets --help")
    sh("hf auth whoami")
    sh(f"hf buckets create {BUCKET} --private --exist-ok")
    flag = "--dry-run" if dry_run else ""
    if what == "data":  # docs/MOLAB.md: real files only (v6_plus_slr52 dev2k/test are symlinks to ss_v6)
        for split in ("train", "dev2k", "test"):
            sh(f"hf buckets sync /data/packed/ss_v6 hf://buckets/{BUCKET}/data/packed/ss_v6 --include '{split}.*' {flag}")
        sh(f"hf buckets sync /data/packed/v6_plus_slr52 hf://buckets/{BUCKET}/data/packed/v6_plus_slr52 "
           f"--include 'train.*' {flag}")
        sh(f"hf buckets sync /data/manifests/ss_v6 hf://buckets/{BUCKET}/data/manifests/ss_v6 {flag}")
        sh(f"hf buckets sync /data/manifests/v6_plus_slr52 hf://buckets/{BUCKET}/data/manifests/v6_plus_slr52 "
           f"--include 'train.jsonl' {flag}")
        return
    if what == "pull":  # bucket -> volume (best/ incl. lm/*.logits.pt from the molab notebook)
        assert run, "--run <name>"
        sh(f"hf buckets sync hf://buckets/{BUCKET}/runs/{run}/best /runs/{run}/best {flag}")
        if not dry_run:
            runs.commit()
        return
    if what == "glm8":  # GRAPHEME_PLAN §13: LM + decode results for molab, plus the corpus so it is never rebuilt
        sh(f"hf buckets sync /data/lm/si_graph8 hf://buckets/{BUCKET}/lm/si_graph8 {flag}")
        sh(f"hf buckets sync /runs/gr_v6x_top8/best/lm hf://buckets/{BUCKET}/runs/gr_v6x_top8/best/lm "
           f"--include '*_glm8*' {flag}")
        sh(f"hf buckets sync /data/lm/si_corpus hf://buckets/{BUCKET}/lm/si_corpus --include 'corpus.txt' {flag}")
        return
    for r in RUNS:  # best/ only (weights, config, units, eval files, lm/ logits); never last/
        sh(f"hf buckets sync /runs/{r}/best hf://buckets/{BUCKET}/runs/{r}/best {flag}")
    # words.tsv (29 MB word list) excluded; graphemes.json, grapheme/, gbpe3000/ only
    sh(f"hf buckets sync /data/units hf://buckets/{BUCKET}/units --exclude 'words.tsv' {flag}")


@app.local_entrypoint()
def main(dry_run: bool = True, what: str = "ckpts", run: str = ""):
    upload.remote(dry_run=dry_run, what=what, run=run)
