# Training on molab

Two marimo notebooks run the next grapheme experiments on [molab](https://molab.marimo.io) (free GPU: RTX Pro 6000,
4 CPUs, 32 GB RAM, sessions up to 12 h, ~1 GB kept between sessions). Data, starting checkpoints and results live in
the private HF bucket `hf://buckets/ovindu-a/sinhala-grapheme-asr`, so nothing has to survive a molab session.

| notebook | runs on | what |
|---|---|---|
| `notebooks/molab_train.py` | GPU | train / continue a run from a preset, checkpoints synced to the bucket, then greedy test eval + log-prob dump |
| `notebooks/molab_units.py` | CPU | C redo, step 1: grapheme-BPE tokenizer fitted to speech -> bucket `units_speech/` |
| `notebooks/molab_decode.py` | CPU | KenLM decoding (grapheme 8-gram / word 4-gram) of a run's log-probs, results to the bucket |

KenLM decoding runs on either: molab_decode.py (4 CPUs, ~1 h per test decode) or Modal (32 CPUs, for big grids).

## 0. One-time: put the training data in the bucket (Modal account `chatgptgroup321`)

```bash
modal run upload_hf_bucket.py --what data              # dry run: lists the files
modal run upload_hf_bucket.py --what data --no-dry-run # ~33 GB, a few minutes, < $0.10
```

Uploads `data/packed/ss_v6/{train,dev2k,test}.{pcm,json}`, `data/packed/v6_plus_slr52/train.{pcm,json}` and the
matching manifests. (v6_plus_slr52's dev2k / test are symlinks to ss_v6 and are not uploaded; the notebook reads
dev2k / test from ss_v6.) Check the bucket's storage limits / billing on Hugging Face first.

## 1. Experiments (presets in molab_train.py)

| preset | run name | starts from | data | steps, peak lr | GPU time (est.) |
|---|---|---|---|---|---|
| A longer (Gemma units) | `v6x_top8_long` | A's `v6x_top8/best` (public HF repo `ctc/`) | v6 + SLR-52, top 8 | 6,000, 5e-5 | ~1–1.5 h |
| **B longer (flat head)** | `gr_v6x_top8_long` | `gr_v6x_top8/best` | v6 + SLR-52, top 8 | 6,000, 5e-5 | ~1–1.5 h |
| B longer + head parts | `gr_v6x_top8_long_parts` | `gr_v6x_top8/best` | same | 6,000, 5e-5 | ~1–1.5 h |
| C redo: v6_top6 | `gbs3k_v6_top6` | scratch, `units_speech/gbpe3000` | v6, top 6 | 7,500, 2e-4 | ~1–1.5 h |
| C redo: v6x_top8 | `gbs3k_v6x_top8` | `gbs3k_v6_top6/best` | v6 + SLR-52, top 8 | 6,000, 1e-4 | ~1–1.5 h |

Why:
- **A longer**: the fair baseline for "B longer": the same 6,000-step continuation with the same schedule, from A's
  checkpoint (not in the bucket; the notebook downloads it from `ovindu-a/sinhala-gemma-asr`). A's dev CER had
  levelled off (6.53 @5000, 6.58 @6000), so it may gain less, but only this run shows it. Decode with the word
  4-gram (A's LM): `lm_decode --ckpt v6x_top8_long/best --lm si_4gram --extra "--tune-limit 2000"`.
- **B longer**: B's dev CER was still falling at the end of gr_v6x_top8 (5.88 -> 5.78 over the last 2,000 steps, LR
  already at its 1e-5 floor). A fresh cosine schedule from 5e-5 continues it.
- **+ head parts** (`--head-parts`, `sinhala_asr/model.py: grapheme_parts`): each grapheme's output row = its own row
  + shared rows for its base (consonant / conjunct / vowel, 164), vowel sign or al-lakuna (18) and ං/ඃ (3). The
  shared rows start at zero, so it starts exactly from B's checkpoint (checked: identical logits) and rare
  graphemes borrow from their parts. Compare with "B longer (flat)" (same steps), not with B.
- **C redo**: the first C took its alphabet from web text (1,266 graphemes, 455 never spoken), so 1,490 of its 3,000
  pieces were single graphemes and its merges followed web vocabulary (1.99 units per spoken word; 27% of units seen
  < 20 times). `molab_units.py` builds the alphabet from the speech transcripts only and adds the transcripts' word
  counts 50x (`12_grapheme_inventory.py --speech-alphabet --transcript-weight 50`) before BPE.

Order: B longer first (largest expected gain, directly usable with the grapheme LM). Then either of the others.

## 2. Running a preset

1. Open the notebook in molab, attach a GPU (molab_train.py), set `HF_TOKEN` (environment or the password field).
2. Pick the preset, check run name / steps / lr.
3. **1. Set up**: downloads this branch's code, checks the GPU actually runs bf16 matmuls with this torch build, shows
   free disk (needs ~25 GB for v6 + SLR-52, ~16 GB for v6).
4. **2. Download**: data + starting checkpoint (+ units) from the bucket, and whatever this run already has there
   (`runs/<name>/last` -> resumes).
5. **3. Train**: blocks the notebook. Every 1,000 steps it evaluates dev2k, checkpoints and syncs `runs/<name>/`
   (`last/`, `best/`, `metrics.jsonl`) to the bucket. It stops itself after 660 min (under the 12 h session) with a
   checkpoint. If the session ends anyway, open it again, same preset, press 1-2-3: it resumes from the bucket.
6. **4. Evaluate**: greedy v6 test (WER / WER_seg / CER / GER) + top-16 log-probs for dev2k and test, synced to
   `runs/<name>/best/` (incl. `lm/*.logits.pt`). The last cell compares the dev CER curve with B's.

For C redo: run `molab_units.py` first (CPU is enough; ~15–30 min), check its table (round-trip failures 0,
units/word clearly below the old 1.99), press Upload, then the two C presets in order.

## 3a. KenLM decoding on molab (CPU, slower)

`notebooks/molab_decode.py`: pick a run, the LM (grapheme 8-gram for `gr_*` runs, word 4-gram for any) and a mode,
then 1. Set up (builds KenLM's Python module with max order 10, which the 8-gram needs; needs a C++ compiler in the
session), 2. Download (the run's `units.json` + dumped log-probs, not the weights; the LM), 3. Decode. Results go to
`runs/<run>/best/lm/eval_lm_test<tag>.*` in the bucket, and the last cell shows all / seen / unseen.

- **quick**: B's dev-tuned weights (8-gram α 0.5 β 1.0; 4-gram α 0.35 β 1.0), 200 dev utts as a sanity check, then
  all 15,860 test utts: ~1 h on 4 CPUs (measured ~6 utts/s with 4 processes on a laptop).
- **grid**: every α × β on all 2,000 dev utts, then test with the best: ~8 min per setting + ~1 h.
- `08_lm_decode.py --eval-limit N` decodes only the first N test utts (timing checks).

## 3. KenLM decoding on Modal (VM agent, 32 CPUs: fast grids)

```bash
git pull --rebase origin grapheme-units
modal run upload_hf_bucket.py --what pull --run gr_v6x_top8_long --no-dry-run   # bucket -> /runs/<run>/best
# B-type runs: grapheme 8-gram (and dual, if §14 says it helps); same flags as GRAPHEME_PLAN.md §13/14
modal run --detach modal_app.py::lm_decode --ckpt gr_v6x_top8_long/best --lm si_graph8 \
    --extra "--lm-kind grapheme --tag _glm8 --tune-limit 2000 --alphas 0.3 0.5 0.7 --betas 0.5 1.0 2.0"
# C-type runs: word 4-gram, as for the first C
modal run --detach modal_app.py::lm_decode --ckpt gbs3k_v6x_top8/best --lm si_4gram --extra "--tune-limit 2000"
```

Reference (v6 test): B greedy 19.58 / 11.71 / 4.32 / 5.40 (WER / WER_seg / CER / GER); B + grapheme 8-gram
14.26 / 8.05 / 3.20 / 4.06; A + word LM 14.72 / 8.95 / 3.68 / 4.61; first C + word LM 14.73 / 8.99 / 3.74 / 4.58.

## 4. Troubleshooting

| symptom | fix |
|---|---|
| step 1 fails at the matmul ("no kernel image", sm_120) | the torch wheel lacks Blackwell kernels: install a newer one (`pip install -U torch --index-url https://download.pytorch.org/whl/cu128`) and restart the kernel |
| out of disk in step 2 | free space under `/tmp`, or point `WORK` (third cell) at a bigger disk |
| RAM pressure / killed workers | lower "data workers" to 1–2 (audio is memory-mapped, so RAM is mostly model + batches) |
| `hf buckets sync` auth errors | `HF_TOKEN` needs write access to the bucket (read is enough for steps 1–2) |
| dev CER of "B longer" jumps up at step 1,000 | peak lr too high for a continued run: rerun with a new run name and lr 2e-5 |
| training log shows `units N (gemma)` | wrong preset / missing units: stop, check `--units` in the printed `train.py` line |
