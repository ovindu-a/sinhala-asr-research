# Sinhala ASR: working notes

Goal: QUAN-style ASR (frozen EmbeddingGemma 2 audio encoder + CTC reader) for Sinhala.

## Budget
**Modal cap: $25 total** (raised from $15 by the user on 2026-10-15). Keep storage small: per run only `last/` + `best/`, overwritten in place. Log every Modal job below *before* launching
the next one. Approximate rates: A100-80GB ≈ $2.50/h, CPU ≈ $0.05/core-h (check modal.com/pricing).

| Date | Job | Resources | Wall time | Est. cost | Running total |
|---|---|---|---|---|---|
| 2026-10-14 | prep from openslr.org (stopped: server too slow) | 8 CPU, 16 GB | ~15 min | ~$0.15 | ~$0.15 |
| 2026-10-15 | prep 4 HF parquet files → 20 h train / 2.4 h dev | 8 CPU, 32 GB | ~10 min | ~$0.15 | ~$0.30 |
| 2026-10-15 | probe_frozen + probe_top2, 1st attempt: STOPPED at step ~75 (5.3 s/step, GPU starved by FLAC reads from volume) | 2× A100-80GB | ~6 min each | ~$0.65 | ~$0.95 |
| 2026-10-15 | pack audio (`modal_app.py::pack`) → /data/packed/openslr_probe | 8 CPU, 16 GB | ~20 min | ~$0.15 | ~$1.10 |
| 2026-10-15 | probe_frozen + probe_top2 (packed), 2000 steps each | 2× A100-80GB | ~12 / ~16 min | ~$1.20 | ~$2.30 |
| 2026-10-15 | **Checked `modal billing summary`: Ephemeral Apps $2.34 this month (all credits), billed $0.00** | | | | **$2.34 actual** |

| 2026-10-15 | deleted step-1 data/checkpoints from volumes (kept metrics) | – | – | – | – |
| 2026-10-15 | pack_v6 (SinhaSpeech stratified_v6 → packed PCM: train/dev2k/test) | 8 CPU, 16 GB | ~10 min | ~$0.10 | ~$2.45 |
| 2026-10-15 | eval_whisper baseline (whisper-small-sinhala-run12-best) on v6 test | H100 | ~15 min est. | ~$1.2 | ~$3.65 |
| 2026-10-15 | v6_top2 + v6_top6: 113 h v6 train, 7500 steps × 1000 s batch (~18 epochs) | 2× H100, 8 CPU, 48 GB | ~60–80 min each (max 165) | est. ~$9, max ~$25 if both hit timeout → watch | ~$12.7 |

| 2026-10-15 | `modal billing summary` after v6 runs + whisper + top6 test eval | | | | **$9.36 actual** |
| 2026-10-15 | slr52_extra analysis (1st try via HfFileSystem: too slow, stopped ~35 min) + fast rerun | 8 CPU | ~45 min | ~$0.5 | ~$9.9 |
| 2026-10-15 | slr52_extra pack (48.5k SLR-52 utts, ~58 h, non-held-out speakers) merged after v6 train | 8 CPU, 32 GB | ~40 min est. | ~$0.5 | ~$10.4 |

| 2026-10-15 | v6x_top8: init v6_top6/best, 174 h (v6 + SLR-52 extra), top-8 unfrozen, 6000 steps, lr 1e-4 | H100 | ~60 min est. | ~$5 | ~$15.5 |
| 2026-10-15 | build_lm (CPU) + lm_dump v6_top6 (GPU, short) | 8 CPU / H100 | ~40 / ~5 min | ~$1 | ~$16.5 |

| 2026-10-15 | **`modal billing summary` after v6x_top8 + build_lm + lm_dump(v6_top6)** | | | | **$16.48 actual** (≈$8.5 left of $25) |

| 2026-10-18 | **`modal billing summary` (profile ovindu-a)**: metered $20.90 (ephemeral $18.34, volumes $1.79 free tier, deployed $0.77) | | | | **≈$4 left of $25** |
| 2026-10-18 | punct_data: wiki + FineWeb-2 → /data/punct (30 M train words) | 8 CPU, 32 GB | ~20–40 min | ~$0.3 | |
| 2026-10-18 | punct baselines eval (CPU) + punct_train xlmr_base 8000 steps × 32 windows + eval | 8 CPU / L40S | ~5 / ≤75 min (timeout) | ~$0.1 + ≤$2.5 | |
| 2026-10-18 | **`modal billing summary` after punct_data + baselines + punct_train** (0.11 s/step, ~15 min train) | | | | **$21.74 actual** (≈$3.3 left of $25) |
| 2026-10-15 | evaluate v6x_top8/best on test + lm_dump v6x_top8/best | 2× H100 (short) | ~10 / ~5 min | ~$1 | ~$17.5 |
| 2026-10-15 | lm_decode v6x_top8 (α/β grid on 1k dev, then test) | 32 CPU | ~30–60 min est. | ~$1–2 | ~$19 |
| 2026-10-15 | lm_decode crashed on test (QUAN beam: all-<unk> frame); fixed, reran test only with α=0.35 β=1.0 | 32 CPU | | | |
| 2026-10-15 | **`modal billing summary`** | | | | **$16.71 actual** |
| 2026-10-15 | speed benchmark ours vs Whisper (200 test clips, H100) | H100 | ~10 min | ~$0.5 | ~$17.2 |
| 2026-10-15 | bench_gpus: same speed benchmark on L4 + T4 (serving cost/latency for a web app) | L4 + T4, 4 CPU each | ~10 min each | ~$0.4 | ~$17.6 |
| 2026-10-15 | web app deployed (`app/web/serve.py`): L4 class, scale-to-zero after 3 min, max 1 GPU container; CPU web front | L4 ≈ $0.96/h while awake | per use | ~$0.05/session | |
| 2026-10-15 | **`modal billing summary`** | | | | **$17.92 metered** ($7.08 left) |
| 2026-10-15 | eval_whispers: off-the-shelf openai/whisper-small + large-v3-turbo on v6 test (timing check before large-v2/v3/medium) | 2× H100 | ~8 min each (6 min decode) | ~$1 | ~$18.9 |
| 2026-10-15 | eval_whispers: openai/whisper-large-v3, large-v2, medium on v6 test | 3× H100, 45 min timeout | STOPPED by user at ~30/248 batches (~6 min), no results | ~$0.8 | ~$19.7 |

Check actual spend any time: `modal billing summary` (this month) / `modal billing report --for today --show-resources`.

## Web app hosting (2026-10-15)
Moved to Modal workspace **ovindu-22** (profile `ovindu-22`): https://ovindu-22--sinhala.modal.run (public, no key; rate limits per IP and per hour in serve.py, max 1 GPU).
Volume `sinhala-asr-web` holds models/ctc (v6x_top8/best), models/lm (si_4gram) and the HF cache; SHA-256 checked against
the old workspace. The old deployment in `ovindu-a` was left running (scales to zero, costs nothing idle); training
data/checkpoints stay on the `ovindu-a` volumes. Results and manifests were copied locally to runs/ and data/manifests/.

Also on Hugging Face Spaces (ZeroGPU, free): https://huggingface.co/spaces/ovindu-a/sinhala-transcribe, code in
app/hf_space/ (deploy: `.venv/bin/python app/hf_space/deploy.py`), weights from ovindu-a/sinhala-gemma-asr. Same page;
one ZeroGPU run per transcription (both models, results streamed). Visitor quotas only apply properly inside the
huggingface.co page (the Gradio client passes the visitor's identity); direct *.hf.space visits count as anonymous.

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
- [x] SinhaSpeech v6 (113 h train), 7500 steps × 1000 s, H100. dev2k CER: top2 7.67%, **top6 6.76%**.
  v6 TEST (15,860): ours v6_top6 WER 19.48 / WER_seg 14.00 / CER 5.09 vs Whisper-small run12-best (same scoring)
  WER 15.50 / WER_seg 10.70 / CER 4.43 (published 15.52 / 4.36). Per-source in runs/*eval*.json on the volume.
- [x] SLR-52 gap: 110 v6 val/test speakers identified via unique sentences; 48,547 extra utts (~58 h, 368 speakers) usable.
- [x] v6x_top8: init v6_top6/best → 174 h (v6 + SLR-52 extra), top-8, 6000 steps, lr 1e-4. dev2k best CER **6.53%** (step 5000; WER 22.15 / WER_seg 16.07), vs v6_top6 6.76%. Test eval not run yet.
- [x] KenLM: built `/data/lm/si_4gram/lm.bin` (0.39 GB; 160 M words: wiki 7.4 M + FineWeb-2 150 M + train 0.8 M; 134k leaking sentences dropped). Logits dumped for v6_top6/best. 
  **v6 TEST results (15,860 utts, same normalization/scoring):**
  | system | WER | WER_seg | CER |
  |---|---|---|---|
  | v6x_top8 greedy (+mark cleanup) | 19.27 | 13.47 | 4.91 |
  | **v6x_top8 + KenLM (α 0.35, β 1.0, beam 32)** | **14.72** | **8.95** | **3.68** |
  | Whisper-small run12-best | 15.50 | 10.70 | 4.43 |
  59% of test sentences occur verbatim in v6 train (OpenSLR prompts repeat across speakers; split is speaker-only).
  Sentences NOT in train (6,508): ours+LM 20.57/5.59 vs Whisper 21.80/6.68 (WER/CER); seen (9,352): 10.01/2.13 vs 10.43/2.59.
  Dev check (same 1,000 utts): greedy 23.59/7.92 → LM 17.88/6.53.

## What KenLM fixes (scripts/compare_errors.py greedy vs kenlm, v6x_top8, v6 test; runs/compare_greedy_vs_lm_test.txt)
Word errors (excl. insertions) 13,057 → 9,289 (−29%); unseen-sentence subset 7,274 → 5,768 (−21%).
Net change by category (all test): one_letter −2,003 · near_miss −721 · wrong_word −713 · spacing −423 ·
orthographic 0 · missed_word +88 · digits/english ~0; insertions +341 (1,462 → 1,803); runaway utts 18 → 10.

## Speed (benchmark_speed.py; Modal gave an H200 for gpu="H100"; 200 random test clips, 666 s audio, mean 3.3 s)
| | batched | one clip at a time |
|---|---|---|
| ours greedy (GPU) | 0.196 s → **RTFx 3,402** | 29.3 ms/clip (RTFx 118) |
| + KenLM beam (CPU, 1 core) | +0.53 s (2.6 ms/clip) → ~RTFx 900 end to end | +2.6 ms/clip |
| Whisper-small greedy, HF generate fp16 | 3.625 s → RTFx 184 | 325.1 ms/clip (RTFx 11) |
Ours greedy is 18.5× faster batched and 11.1× faster one-at-a-time. Caveat: Whisper pads every clip to 30 s and
HF generate is not the fastest Whisper runtime (faster-whisper/CTranslate2 would narrow the gap).

## Lessons
- Modal volume: reading thousands of small files is slow (~30 files/s). Always pack audio (04_pack_audio.py) and
  train with `--packed-dir`. For bigger sets, write packed arrays straight from parquet instead of per-file FLACs.
- zsh does not word-split `$var`; launch Modal runs with explicit arguments.

## Resume after an interrupted session
1. `modal app list` / `modal volume ls sinhala-asr-runs` to see what finished.
2. Check the budget table above before launching anything.
3. Continue from the first unchecked item.

## Mac app (app/): Sinhala Transcribe
SwiftUI app modelled on Embedding Gemma Transcribe (drop a file / hold Right ⌥ to dictate and paste) with a model
switcher: Whisper-small (SinhaSpeech), v6x_top8 greedy, v6x_top8 + KenLM. Inference runs in a Python worker
(app/backend/server.py, JSON lines over stdin/stdout) that the app starts and stops.
- Models: `~/Library/Application Support/SinhalaTranscribe/models/{ctc,lm,whisper}` (pulled from the Modal volumes;
  Whisper's HF-cache entry on the volume is a symlink, so it came from huggingface.co by plain curl).
  `app/prepare_models.py` adds `hf/`: EmbeddingGemma 2 tokenizer/config + audio tower only (0.61 GB vs 1.5 GB;
  every param + buffer as loaded, so outputs are bit-identical).
- `app/build.sh` = dev build (repo .venv); `app/build.sh --standalone` copies uv's relocatable CPython, the exact
  package closure (app/bundle_runtime.py) and the models into the app (3.5 GB, APFS clones). Runs on Apple silicon,
  macOS 14+. Ad-hoc signed: on another Mac, right-click → Open the first time.
- M2 Pro (MPS), 58 s clip: load ~15 s; ctc ~1–2 s, ctc_lm ~0.6–1.4 s, whisper ~5 s.
- Whisper fine-tune emits one utterance per window and loops on near-silence → its input is cut at pauses
  (speech_pieces); 30 s windows dropped most sentences.
- Dictation (Right ⌥, paste via ⌘V) needs Microphone + Accessibility; ad-hoc re-signing on each rebuild resets the
  Accessibility grant.

## Punctuation restoration (2026-10-18; sinhala_asr/punct*.py, scripts/09-11, `modal_app.py::punct_*`)
ASR text stays punctuation-free; a separate word tagger adds , . ? afterwards (labels: mark after each word; ; : → ,
and ! ෴ → .). Training text: Wikipedia si 7.1 M + FineWeb-2 sin_Sinh 30.3 M + v6 train raw transcripts 0.6 M words
(/data/punct; windows containing a ≥5-word v6 dev/test sentence dropped). Model: XLM-R base, 8000 steps × 32 windows,
lr 5e-5, L40S → `/runs/punct/xlmr_base/best` (dev micro-F1 0.751). Results: runs/punct/{baselines_eval.json,
xlmr_base/eval.json} (with 25 samples per set). B0 = . or ? (ද rule) at the end only; B1 = most frequent label per word.
ASR sets: reference marks moved onto the v6-test hypotheses by word alignment.

| set (words) | system | , F1 | . F1 | ? F1 | micro F1 | sentence-end F1 |
|---|---|---|---|---|---|---|
| wiki held-out (118k) | B1 | 0.01 | 0.86 | 0.47 | 0.69 | 0.86 |
| | **XLM-R** | **0.51** | **0.90** | **0.83** | **0.79** | **0.90** |
| web held-out (98k) | B1 | 0.00 | 0.78 | 0.39 | 0.61 | 0.78 |
| | **XLM-R** | **0.57** | **0.90** | **0.79** | **0.80** | **0.89** |
| v6 test OpenSLR fragments (14k) | B0 | 0.00 | 0.43 | 0.33 | 0.42 | 0.44 |
| | **XLM-R** | **0.39** | **0.66** | **0.68** | **0.65** | **0.67** |
| v6 test conversational, refs (4.9k) | B1 | 0.02 | 0.27 | 0.33 | 0.24 | 0.29 |
| | **XLM-R** | **0.22** | **0.51** | **0.48** | **0.46** | **0.53** |
| ASR: Gemma CTC + KenLM, all v6 test (76k) | B0 | 0.00 | 0.41 | 0.33 | 0.40 | 0.43 |
| | **XLM-R** | **0.23** | **0.62** | **0.65** | **0.61** | **0.63** |
| ASR: Gemma CTC + KenLM, conversational (4.8k) | B1 | 0.02 | 0.24 | 0.28 | 0.22 | 0.26 |
| | **XLM-R** | **0.19** | **0.43** | **0.41** | **0.39** | **0.45** |
| ASR: Whisper-small, all v6 test (75k) | **XLM-R** | 0.22 | 0.63 | 0.65 | 0.61 | 0.64 |
Caveats: v6 test clips are ~3 s fragments, so "is this the end of the sentence" is often unknowable from the text;
the conversational set is small (360 utts) and its reference punctuation is inconsistent (many complete utterances
have no final mark). The model often leaves the last word of a short utterance unmarked; in the app, where the end of
a recording is a sentence end, force . or ? on the last word. Model is 1.1 GB fp32 (~0.55 GB fp16); not in the app yet.

## Grapheme-level units (2026-10-18; planned, not run yet) → docs/GRAPHEME_PLAN.md
Replace Gemma-token CTC units with units that never split a grapheme (grapheme-kit 0.1.1 splitter + a repaya ZWJ
repair): **C** = grapheme-BPE, SentencePiece BPE vocab 3000 trained on the full KenLM text (main run), **B** = single
graphemes (optional). Same recipe as A without probes: v6_top6 → v6x_top8 → KenLM (unchanged word 4-gram).
Code: sinhala_asr/{graphemes,units}.py, scripts/12_grapheme_inventory.py, 13_build_units.py, rescore.py,
`modal_app.py::{lm_corpus,grapheme_units}`, `train --units`. New metric GER (grapheme error rate) in every score().
GER of existing systems on v6 test: v6x_top8 greedy 6.22, + KenLM 4.61 (unseen sentences 6.94), Whisper-small 5.52.
Stand-in check (v6 transcripts as corpus): gbpe3000 1.75 units/word vs Gemma 2.77, 0 round-trip failures;
encoder = 25 frames/s vs ≤ 10.6 gbpe targets/s (p99 5.0). Est. cost: ~$11 for C, ~$21–23 for B + C.
New Modal account for the grapheme runs: Phase 0 in docs/GRAPHEME_PLAN.md rebuilds the data there (pack_v6,
slr52_extra), downloads the exact KenLM + v6x_top8 from HF (`fetch_reference`) and checks the manifests against
reference/SHA256SUMS (`verify_data`). reference/ = manifests, OpenSLR speaker table, baseline hyps, A's metrics.
