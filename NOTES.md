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
| 2026-10-18 | **New Modal account `chatgptgroup321`** (grapheme runs): `modal billing summary` = $0.00 | | | | **$0.00** (new account) |
| 2026-10-18 | Phase 0: pack_v6 + slr52_extra + fetch_reference + verify_data | 8 CPU (+2 CPU) | ~25 min actual | ~$1.5 | ~$1.5 |
| 2026-10-18 | **Gate 0 passed**: all 4 manifests match reference/SHA256SUMS (v6_plus_slr52 171,163 utts, 174.3 h); **`modal billing summary`** | | | | **$0.16 metered** |
| 2026-10-18 | Decided: run C (gbpe3000) **and B** (graphemes). Phase 1: lm_corpus (rebuild KenLM text) | 8 CPU, 32 GB | ~30–40 min est. | ~$0.4 | ~$0.6 |
| 2026-10-18 | **Gate 1 passed**: si_corpus stats == si_4gram lm_config stats (160.3 M words; corpus.txt 2.60 GB) | | | | |
| 2026-10-18 | Phase 2: grapheme_units --vocabs 3000 (grapheme + gbpe3000) | 16 CPU, 64 GB | ~20–30 min est. | ~$0.4 | ~$1.0 |
| 2026-10-18 | **Gate 2 passed** (Phase 2 took ~6 min): alphabet 1,266 (+▁, unk = 1,269 grapheme units), coverage 99.96%, round-trip failures 0/0, unk 0.017% (gbpe, train) / 0.008% (grapheme), gbpe merged pieces seen <20×: 2.6%; **`modal billing summary`** | | | | **$0.41 metered** |
| 2026-10-18 | Phase 3: gb3k_v6_top6 (C) + gr_v6_top6 (B), 7,500 steps each | 2× H100 | ~55 min each est. | ~$5.5 | ~$6 |
| 2026-10-18 | gb3k_v6_top6 done (best dev2k CER 7.33 @ 7000; 7500: WER 23.17 / WER_seg 17.32 / CER 7.35 / GER 8.51); gr_v6_top6 still running; **`modal billing summary`** — Phase 3 costs ~2× the plan's estimate | | ~65 min | | **$10.23 metered** |
| 2026-10-18 | gr_v6_top6 done (best dev2k CER 6.02 @ 7500: WER 23.34 / WER_seg 15.35 / GER 7.28); **`modal billing summary`** (meter revised down) | | ~65 min | | **$10.18 metered** |
| 2026-10-18 | User approved Phases 4 + 5 for B and C (revised total est. ~$32–37). Phase 4: gb3k_v6x_top8 + gr_v6x_top8, 6,000 steps each | 2× H100 | ~60–70 min each est. | ~$9–10 | ~$20 |
| 2026-10-18 | gb3k_v6x_top8 done (best dev2k CER 6.94 @ 6000: WER 22.73 / WER_seg 16.26 / GER 8.05); gr_v6x_top8 finishing; **`modal billing summary`** | | | | **$19.32 metered** |
| 2026-10-18 | Phase 5 C: evaluate test + lm_dump (parallel), then lm_decode | 2× H100 short + 32 CPU | ~15 min + ~30–60 min est. | ~$4 | ~$24 |
| 2026-10-18 | gr_v6x_top8 done (best dev2k CER 5.78 @ 6000: WER 22.73 / WER_seg 14.53 / GER 6.96) | | | | |
| 2026-10-18 | Phase 5 B: evaluate test + lm_dump (parallel), then lm_decode | 2× H100 short + 32 CPU | ~15 min + ~30–60 min est. | ~$4 | ~$28 |
| 2026-10-18 | Phase 5 done for C (+KenLM test WER 14.73 / WER_seg 8.99 / CER 3.74 / GER 4.58, α 0.35 β 1.0) and B (best β=2.0 on grid edge); **`modal billing summary`** (meter lags) | | | | **$19.25 metered** |
| 2026-10-18 | lm_decode B extended grid: α {0.2, 0.35, 0.5} × β {2, 3, 4} (best β on edge; α stays low, so not the plan's 0.5–1.1 α range) | 32 CPU | ~30–60 min est. | ~$1.5 | ~$29.5 |
| 2026-10-18 | B extended grid confirmed α 0.35 β 2.0 (β 3/4 worse on WER). **`modal billing summary`** at end (meter lags, GPU jobs not all counted yet; check the dashboard) | | | | **$19.37 metered** |
| 2026-10-18 | B decoder fix: lm_decode gr_v6x_top8 with --space-offsets (penalty on the bare ▁ unit), α {0.35, 0.5} × β {1, 2} × space {0, −1, −2} on all 2,000 dev2k, tag _space | 32 CPU | ~1–1.5 h est. | ~$2 | |
| 2026-10-18 | B decoder fix done: dev picks α 0.35 β 1.0 space 0 (no penalty); test WER 15.18 → 14.78; **`modal billing summary`** (meter lags) | | ~1 h | | **$19.44 metered** |
| 2026-10-20 | upload_hf_bucket.py: best/ of gr_v6_top6, gr_v6x_top8, gb3k_v6_top6, gb3k_v6x_top8 (weights, config, units, eval files, lm/ logits) + /data/units (minus words.tsv) → private HF bucket `ovindu-a/sinhala-grapheme-asr`, 33 files, 3.78 GB (dry run + upload + listing) | 4 CPU, 8 GB | ~1 min upload | < $0.05 | |
| 2026-10-20 | Grapheme LM (GRAPHEME_PLAN §13) step 1: `grapheme_lm --order 8` over full si_corpus → /data/lm/si_graph8 (lm_image rebuild, KenLM max order 10); `modal billing summary` before = $19.53 metered | 16 CPU, 128 GB, 512 GiB disk (Modal min; 400 GB rejected at launch) | 16 min actual (11.1 M sents, 158.3 M words, 677 M grapheme tokens, 7,929 types; lmplz peak RSS 94 GB; lm.bin 0.75 GB) | ~$0.5 | ~$20 |
| 2026-10-20 | Grapheme LM step 2: lm_decode gr_v6x_top8/best --lm si_graph8 --lm-kind grapheme, α {0.3, 0.5, 0.7, 1.0} × β {0, 0.5, 1, 2} (β per grapheme) on all 2,000 dev2k, then test; tag _glm8 | 32 CPU, 64 GB | ~25 min actual | ~$0.7 | ~$21 |
| 2026-10-20 | `modal run upload_hf_bucket.py --what glm8` (dry run, then --no-dry-run): /data/lm/si_graph8, *_glm8* eval files, si_corpus/corpus.txt (2.6 GB, one-off) → private bucket; 6 files 3.36 GB, re-dry-run = all skipped | 4 CPU, 8 GB | ~2 min | < $0.05 | |
| 2026-10-20 | **`modal billing summary`** after grapheme LM build + decode + bucket upload | | | | **$20.20 metered** |

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

## Grapheme-level units (2026-10-18; run on Modal account chatgptgroup321) → docs/GRAPHEME_PLAN.md
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
Backup (2026-10-20, `modal run upload_hf_bucket.py --no-dry-run`): private HF Storage Bucket
**hf://buckets/ovindu-a/sinhala-grapheme-asr**: runs/{gr,gb3k}_{v6_top6,v6x_top8}/best/ (model.safetensors 0.78–0.99 GB,
config, units, eval files; v6x_top8 also lm/{dev2k,test}.logits.pt + eval_lm_*) and units/ (graphemes.json, grapheme/,
gbpe3000/; words.tsv left out). last/ (optimizer state) not uploaded; Modal volumes untouched.
Unit stats on the real corpus (v6_plus_slr52 train | v6 test), from /data/units/*/report.json:

| units | size | units/word | graphemes/unit | single-unit words | unk | round-trip fails | targets/s mean / p99 / max | units seen <20× in speech text |
|---|---|---|---|---|---|---|---|---|
| A Gemma (same text) | 5,051 used | 2.80 \| 2.79 | 1.17 \| 1.14 | 20.4% \| 20.3% | 0.000% \| 0.085% | 251 \| 27 | 3.6 / 6.9 / 11.9 | 85.9% (merged: 90.6%) |
| B grapheme | 1,269 | 4.04 \| 3.97 | 1.00 | 6.6% \| 7.6% | 0.008% \| 0.009% | 0 \| 0 | 5.2 / 10.0 / 21.7 | 58.5% (455 unused) |
| C gbpe3000 | 3,001 | 1.99 \| 1.93 | 1.65 \| 1.66 | 43.6% \| 45.7% | 0.017% \| 0.019% | 0 \| 0 | 2.6 / 5.3 / 11.8 | 27.0% (merged: 2.6% of 1,508) |

Alphabet 1,266 graphemes (corpus 7,925 distinct, speech 875; min count 50), corpus coverage 99.96%. gbpe3000 piece
lengths (graphemes): 1: 1,490, 2: 920, 3: 412, 4: 136, 5: 34, 6+: 6. Top invalid clusters are typo conjuncts like කි්‍ර.

### Results (v6 test 15,860 utts; committed copies of all result files: reference/grapheme/, see its README)

Data on the new account rebuilt by Phase 0; all 4 manifests byte-identical to the original runs (gate 0), corpus stats
identical to si_4gram (gate 1), so A / B / C saw the same audio, transcripts and LM.

| units | dev2k CER v6_top6 (best) | dev2k v6x_top8 best @6000: WER / WER_seg / CER / GER | test greedy WER / WER_seg / CER / GER | test + KenLM (α, β) WER / WER_seg / CER / GER | unseen + KenLM WER / WER_seg / CER / GER |
|---|---|---|---|---|---|
| A Gemma | 6.76 | – / – / 6.53 (@5000) / – | 19.27 / 13.47 / 4.91 / 6.22 | **14.72** / 8.95 / 3.68 / 4.61 (0.35, 1.0) | **20.57** / 13.10 / 5.59 / 6.94 |
| C gbpe3000 | 7.33 (@7000) | 22.73 / 16.26 / 6.94 / 8.05 | 19.57 / 13.28 / 5.23 / 6.21 | 14.73 / 8.99 / 3.74 / 4.58 (0.35, 1.0) | 20.79 / 13.31 / 5.77 / 6.93 |
| B graphemes | **6.02** | 22.73 / **14.53** / **5.78** / **6.96** | 19.58 / **11.71** / **4.32** / **5.40** | 15.18 / **8.31** / **3.43** / **4.33** (0.35, 2.0) | 20.79 / **12.20** / **5.17** / **6.42** |
| B + grapheme 8-gram (2026-10-20) | | | | **14.26** / **8.05** / **3.20** / **4.06** (0.5, 1.0) | **19.23** / **11.70** / **4.81** / **5.98** |

Seen-sentence subset + KenLM (9,352): A 10.01 / 5.61 / 2.13 / 2.70, C 9.85 / 5.51 / 2.09 / 2.65, B 10.66 / 5.18 / 2.01 / 2.61.
B's KenLM grid best was β = 2.0 (edge); extended once (α 0.2/0.35/0.5 × β 2/3/4): β 3 and 4 raise dev WER, so 2.0 stays.

Findings:
- **C (grapheme-BPE) = A.** Every test metric within 0.06 with KenLM; same α/β. Merging graphemes into 3k pieces
  neither helps nor hurts. Fewer English errors (106 vs 192 words), more missed words (461 vs 306).
- **B (single graphemes) wins on everything except WER.** Best CER / WER_seg / GER greedy and with KenLM, on seen
  and unseen sentences; dev CER below A from step 1000 on. Fewer one_letter (2,458 vs 2,645), near_miss, wrong_word
  and missed_word errors than A with KenLM: 9,012 vs 9,289 word errors in total.
- **B's WER loss is all insertions.** WER = (word errors + inserted words) / 75,338 ref words: A (9,289 + 1,803) =
  14.72%, B (9,012 + 2,425) = 15.18%, C (9,243 + 1,857) = 14.73%. With β = 2 (word bonus) the decoder adds ~620
  more extra words than A. Greedy B has the fewest insertions (1,184) but the most spacing errors (5,011 vs 3,220):
  with a separate space unit, word boundaries are B's weak point; WER_seg (ignores boundaries) shows the real gain.
- Next if continuing: tune B's decoder against insertions (choose α/β on WER with an insertion penalty, or a
  space-aware LM / char-level LM for B), then rerun only lm_decode (CPU, ~$1.5). No retraining needed.

### Error categories (compare_errors.py; reference/grapheme/analysis/; 75,338 test words)

| category | greedy A / C / B | + KenLM A / C / B (β 2) | KenLM change A / C / B |
|---|---|---|---|
| one letter wrong | 4,648 / 3,629 / 3,836 | 2,645 / 2,514 / 2,458 | −2,003 / −1,115 / −1,378 |
| a few letters wrong | 1,446 / 1,625 / 1,176 | 725 / 769 / 690 | −721 / −856 / −486 |
| different / garbled word | 2,234 / 2,658 / 1,716 | 1,521 / 1,543 / 1,317 | −713 / −1,115 / −399 |
| spacing only | 3,220 / 3,497 / 5,011 | 2,797 / 2,733 / 2,989 | −423 / −764 / −2,022 |
| same-sounding spelling | 1,043 / 1,127 / 1,333 | 1,043 / 1,057 / 1,090 | 0 / −70 / −243 |
| English | 188 / 173 / 243 | 192 / 106 / 152 | +4 / −67 / −91 |
| missed word | 218 / 315 / 191 | 306 / 461 / 256 | +88 / +146 / +65 |
| **all word errors** | 13,057 / 13,084 / 13,566 | 9,289 / 9,243 / 9,012 | −3,768 / −3,841 / −4,554 |
| extra (inserted) words | 1,462 / 1,657 / 1,184 | 1,803 / 1,857 / 2,425 | +341 / +200 / +1,241 |

- Grapheme units cut greedy one-letter errors by ~1,000 (Gemma tokens split a letter from its vowel sign: උන -> ුන), but
  KenLM already repairs most of these for A.
- KenLM's gain: A mostly one-letter fixes (53%); C spread evenly; B mostly spacing (44%) at the cost of +1,241 extra words.
- C errs a whole piece (1.65 graphemes) at a time: dropped endings (විපක්ෂයේ -> විපක්ෂ), dropped first pieces
  (වුණාට -> ණාට), missed short words (461 vs A 306). Its pieces were fitted to web text (1.99 units/word on speech vs 1.75
  when fitted to speech transcripts), 27% seen < 20× in the speech data, and it was still improving at step 6000.
- Training curves: B and C dev CER still falling at the end (LR at its 1e-5 floor), A had plateaued; a longer fresh
  schedule could help C most (untested).

### B decoder retune (scripts/08_lm_decode.py --space-offsets, --tag; runs/gr_v6x_top8/eval_lm_test_space.json)

B + KenLM's extra words were split compounds (බලනකොට -> බලන කොට, යුතුය -> යුතු ය; 1,094 test clips differ from the ref
only by extra spaces). Tried a penalty on the bare ▁ unit's log-prob, grid α {0.35, 0.5} × β {1, 2} × space {0, −1, −2}
on all 2,000 dev2k (the first grid used 1,000). The penalty controls the word count (hyp/ref words 1.017 -> 0.98) but
never lowers dev WER: it removes right splits with the wrong ones. Dev picks **α 0.35, β 1.0, no penalty** (the
1,000-utt grid had picked β 2.0 by 0.15 WER; on 2,000 utts β 1 and 2 are within 0.02).

| v6 test + KenLM | WER | WER_seg | CER | GER | seen WER / CER | unseen WER / WER_seg / CER / GER | word errs + extra words |
|---|---|---|---|---|---|---|---|
| A Gemma (0.35, 1.0) | **14.72** | 8.95 | 3.68 | 4.61 | 10.01 / 2.13 | 20.57 / 13.10 / 5.59 / 6.94 | 9,289 + 1,803 |
| C gbpe3000 (0.35, 1.0) | 14.73 | 8.99 | 3.74 | 4.58 | **9.85** / 2.09 | 20.79 / 13.31 / 5.77 / 6.93 | 9,243 + 1,857 |
| B grapheme (0.35, 2.0), old | 15.18 | **8.31** | 3.43 | 4.33 | 10.66 / 2.01 | 20.79 / **12.20** / **5.17** / 6.42 | 9,012 + 2,425 |
| B grapheme (0.35, 1.0) | 14.78 | 8.33 | 3.41 | 4.29 | 10.37 / 1.97 | 20.26 / 12.29 / 5.19 / 6.40 | 9,346 + 1,789 |
| **B + grapheme 8-gram (0.5, 1.0)** | **14.26** | **8.05** | **3.20** | **4.06** | 10.25 / **1.90** | **19.23** / **11.70** / **4.81** / **5.98** | 9,268 + 1,474 |

With β 1, B's extra words fall to A's level (1,789 vs 1,803) but merges replace splits (spacing errors 2,989 -> 3,295 vs
A's 2,797). Net: B ties A on WER (+0.06), keeps CER −0.27 / WER_seg −0.62 / GER −0.32, and has the best unseen-sentence
WER (20.26 vs 20.57). Spacing is what remains between B and a clear WER win; a global decoder knob does not fix it
(word-specific boundary conventions: ref transcripts vs web-text LM).

### B + grapheme 8-gram LM (2026-10-20; GRAPHEME_PLAN §13; reference/grapheme/training/gr_v6x_top8/eval_lm_test_glm8.json)

`grapheme_lm --order 8` over the same si_corpus text (11.1 M sentences, 158.3 M words → 677 M grapheme tokens, 7,929
types, word break "|" as a token; prune 0 0 1): lm.bin 0.75 GB, 16 min on 16 CPU (lmplz peak 94 GB RAM). The function
had asked for 400 GiB ephemeral disk, which Modal rejects (minimum 512 GiB); now 512 GiB. Decoded B
(`--lm-kind grapheme`, β = per-grapheme bonus) with the grid on all 2,000 dev2k:

| dev2k WER (hyp/ref words) | β 0 | β 0.5 | β 1.0 | β 2.0 |
|---|---|---|---|---|
| α 0.3 | 17.76 (0.985) | 17.40 | 17.14 | 17.80 (1.028) |
| α 0.5 | 17.57 | 17.10 | **16.71** (0.997) | 16.75 (1.018) |
| α 0.7 | 17.50 | 17.29 | 16.78 | 16.84 |
| α 1.0 | 18.44 (0.967) | 17.94 | 17.48 | 17.04 |

Best **α 0.5, β 1.0** (inside the grid on both axes, so no extension): dev2k 16.71 / 10.31 / 4.45 / 5.42 vs B + word LM
17.16 / 10.70 / 4.68 / 5.64 (WER / WER_seg / CER / GER). Test: **14.26 / 8.05 / 3.20 / 4.06** (A 14.72 / 8.95 / 3.68 /
4.61, B + word LM 14.78 / 8.33 / 3.41 / 4.29); unseen sentences 19.23 / 11.70 / 4.81 / 5.98 (A 20.57, B + word LM 20.26
WER); seen 10.25 WER (A 10.01, B + word LM 10.37). Better than both on every source (openslr 13.09 vs 13.51 / 13.53).

Findings (compare_errors.py; reference/grapheme/analysis/compare_{graph_lm_b1_vs_glm8,gemma_lm_vs_graph_glm8}_test.txt):
- **First system to beat A on WER** (−0.46 test, −1.34 on unseen sentences); the gain is largest on unseen text, where the
  word 4-gram only had a flat unk penalty.
- WER = (word errors + inserted words) / 75,338: A 9,289 + 1,803, B + word LM 9,346 + 1,789, **B + glm8 9,268 + 1,474**.
  Most of the gain is fewer inserted words (−315 vs B + word LM, −329 vs A); fewer one_letter / near_miss / wrong_word /
  missed_word / English errors too.
- **Spacing is still not fixed**: spacing-only errors 3,441 (B + word LM 3,295, A 2,797). The grapheme LM stops the split
  compounds (fewer extra words) but merges/splits differently rather than less; still the main gap to close for B.
- Next if continuing: combine both LMs (word LM at "|", grapheme LM per unit; decoder change), or a higher order/less
  pruning. Files: bucket `lm/si_graph8/`, `lm/si_corpus/corpus.txt`, `runs/gr_v6x_top8/best/lm/*_glm8*`.
