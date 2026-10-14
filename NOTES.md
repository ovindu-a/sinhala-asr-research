# Sinhala ASR: working notes

Goal: QUAN-style ASR (frozen EmbeddingGemma 2 audio encoder + CTC reader) for Sinhala.

## Budget
**Modal hard cap: $15 total** (user instruction, 2026-10-14). Log every Modal job below *before* launching
the next one. Approximate rates: A100-80GB ≈ $2.50/h, CPU ≈ $0.05/core-h (check modal.com/pricing).

| Date | Job | Resources | Wall time | Est. cost | Running total |
|---|---|---|---|---|---|
| 2026-10-14 | prep from openslr.org (stopped: server too slow) | 8 CPU, 16 GB | ~15 min | ~$0.15 | ~$0.15 |
| 2026-10-15 | prep 4 HF parquet files → 20 h train / 2.4 h dev | 8 CPU, 32 GB | ~10 min | ~$0.15 | ~$0.30 |
| 2026-10-15 | probe_frozen + probe_top2, 1st attempt: STOPPED at step ~75 (5.3 s/step, GPU starved by FLAC reads from volume) | 2× A100-80GB | ~6 min each | ~$0.65 | ~$0.95 |
| 2026-10-15 | pack audio (`modal_app.py::pack`) → /data/packed/openslr_probe | 8 CPU, 16 GB | ~20 min | ~$0.15 | ~$1.10 |
| 2026-10-15 | probe_frozen + probe_top2 (packed), 2000 steps each | 2× A100-80GB | ~12 / ~16 min | ~$1.20 | ~$2.30 |
| 2026-10-15 | **Checked `modal billing summary`: Ephemeral Apps $2.34 this month (all credits), billed $0.00** | | | | **$2.34 actual** |

Check actual spend any time: `modal billing summary` (this month) / `modal billing report --for today --show-resources`.

## Plan / status
- [x] 1a tokenizer check → `data/01_tokenizer_report.txt` (3.26 tok/word; units ≈ characters)
- [x] 1b zero-shot audio→text retrieval → `data/02_zeroshot_report.txt`: Sinhala R@1 82% of 300 (chance 0.3%); English 100% of 73
- [x] local smoke test of `scripts/train.py` (MPS, 30 steps); QUAN loader reads our checkpoint
- [x] 1b probe on Modal: 20 h OpenSLR, 2000 steps (17 epochs), dev = 2,000 utts / 38 unseen speakers, greedy CTC:
  | run | step 500 | 1000 | 1500 | 2000 (WER / CER) |
  |---|---|---|---|---|
  | probe_frozen (top-k 0) | 72.9 / 27.2 | 55.1 / 17.6 | 48.8 / 14.7 | **47.8 / 14.1** |
  | probe_top2 (top-k 2)   | 63.2 / 21.4 | 50.3 / 15.3 | 45.8 / 13.2 | **44.8 / 12.8** |
  Still improving when LR hit the floor. Many "word errors" are spacing/spelling variants (නොහැකි වීමය vs නොහැකිවීමය; ref typo උත්සහ).
  Checkpoints: volume `sinhala-asr-runs`, `<run>/best` (QUAN-loadable). Metrics copied to `runs/<run>/metrics.jsonl`.
- [ ] decide units (Gemma pieces vs words) and data scale-up

## Lessons
- Modal volume: reading thousands of small files is slow (~30 files/s). Always pack audio (04_pack_audio.py) and
  train with `--packed-dir`. For bigger sets, write packed arrays straight from parquet instead of per-file FLACs.
- zsh does not word-split `$var`; launch Modal runs with explicit arguments.

## Resume after an interrupted session
1. `modal app list` / `modal volume ls sinhala-asr-runs` to see what finished.
2. Check the budget table above before launching anything.
3. Continue from the first unchecked item.
