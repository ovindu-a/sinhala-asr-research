# Grapheme-level CTC units: run plan

**Status (2026-10-18): run for B and C.** Results and budget: NOTES.md ("Grapheme-level units"); result files:
reference/grapheme/. Deviation: B's KenLM α/β was re-picked on all 2,000 dev2k utts (β 1.0 instead of 2.0, see NOTES).

Goal: replace the CTC reader's output units (Gemma tokenizer ids) with units that never split a Sinhala grapheme
cluster, and rerun the v6_top6 → v6x_top8 → KenLM recipe to see whether that helps. Everything else (EmbeddingGemma 2
audio encoder, data, packing, hyperparameters, word 4-gram KenLM, scoring) stays the same.

This document is written to be executed step by step on another machine (by you or by Claude Code). Each step lists
the exact command, what it writes, how to check it, and what to do if it fails. **Do the steps in order; each phase
has a gate.**

---

## 1. The systems

| id | units | how they are made | units in the vocabulary |
|---|---|---|---|
| **A** (done) | Gemma tokenizer ids | `Units.build` over the training text | 2,774 (+ blank, unk) |
| **C** (main run) | **grapheme-BPE, 3k** (`gbpe3000`) | SentencePiece BPE over the KenLM corpus where every grapheme is one symbol; merges never split a grapheme | 3,000 SentencePiece pieces incl. `<unk>` (+ blank) = 3,001 units |
| **B** (optional) | single graphemes (`grapheme`) | the grapheme alphabet + a space unit `▁` | ~800–1,200 |

A is the existing `v6x_top8` (+ KenLM). C is the experiment you asked for. B tells you whether the gain (if any)
comes from respecting grapheme boundaries (A vs B) or from merging graphemes into bigger pieces (B vs C). B costs
another ~$10; decide before Phase 3 whether to run it.

Recipe for both B and C (identical to A, minus the probes):

1. `*_v6_top6`: SinhaSpeech v6 train (113 h), top 6 encoder layers unfrozen, 7,500 steps × 1,000 s batches,
   lr 2e-4, warmup 5%, H100 (~55 min).
2. `*_v6x_top8`: init from that unit type's own `*_v6_top6/best`, v6 + SLR-52 extra (174 h), top 8 unfrozen,
   6,000 steps, lr 1e-4, warmup 3% (~60 min).
3. Greedy test eval, logit dump, KenLM α/β grid on 1,000 dev utts, test decode with the best α/β.

Run names on the `sinhala-asr-runs` volume:

| | v6_top6 | v6x_top8 |
|---|---|---|
| C | `gb3k_v6_top6` | `gb3k_v6x_top8` |
| B | `gr_v6_top6` | `gr_v6x_top8` |

---

## 2. What the code does (already implemented and tested locally)

| file | what |
|---|---|
| `sinhala_asr/graphemes.py` | grapheme splitting with **grapheme-kit 0.1.1** (`GraphemeSplitter`). Uses the splitter, not `Graphemizer`: Graphemizer rewrites text (Tamil/Sinhala fixups), so its output does not always join back to the input. Repairs the splitter's repaya rule (it strips the ZWJ in ර්‍ය; our `normalize()` keeps it), so ර්‍ය is one grapheme like every other ්‍ය conjunct. `is_valid()` rejects typo clusters (දිි, ලේ්, stray marks). `WordCache` splits each distinct word once. |
| `sinhala_asr/units.py` | three unit kinds behind one interface (`encode`, `decode`, `pieces`, `init_ids`, `save`); `load_units(path)` picks the kind from `units.json`. Old Gemma `units.json` files load unchanged. `PieceUnits` (gbpe) stores the SentencePiece model inside `units.json` (base64), so a checkpoint is self-contained. Grapheme/gbpe `decode` runs `normalize()`, which restores ZWJ in ්‍ය/්‍ර if the model emits e.g. ද් + ය as two units. |
| `sinhala_asr/model.py` | head init accepts a list of Gemma ids per unit: each grapheme/piece row starts as the mean Gemma text embedding of the tokens of its text (`"▁x"` → ids of `" x"`, `"▁"` → the `▁` token). Gemma units: unchanged, bit-identical init. |
| `sinhala_asr/metrics.py` | adds `ger` (grapheme error rate; spaces count, like `cer`) to every `score()` result. |
| `scripts/07_build_lm.py` | `--corpus-only` (write the KenLM text and stop) and `--keep-corpus`. |
| `scripts/12_grapheme_inventory.py` | corpus → `/data/units/graphemes.json` (alphabet + counts + stats) and `/data/units/words.tsv` (word, count). |
| `scripts/13_build_units.py` | builds `grapheme` or `gbpe<V>` units + `report.json` (the go/no-go numbers). SentencePiece is trained on `word<TAB>count` lines = the whole corpus, exactly, at a fraction of the memory. |
| `scripts/train.py`, `evaluate.py`, `08_lm_decode.py`, `benchmark_speed.py` | use `load_units` / `units.init_ids` / `units.pieces`; work for every kind. |
| `reference/`, `scripts/unpack_reference.sh` | small reference files for a fresh Modal account (§3) |
| `modal_app.py::fetch_reference`, `verify_data` | KenLM + A's checkpoint from Hugging Face; manifest checksums vs `reference/SHA256SUMS` |
| `scripts/rescore.py` | re-scores any `*.hyps.jsonl` (all metrics + GER), overall / per source / seen-vs-unseen sentences. CPU, local. |
| `modal_app.py` | new `lm_corpus` and `grapheme_units` functions; `train` takes `--units <dir under /data/units>`. The base image adds `grapheme-kit==0.1.1`, `sentencepiece`, `rapidfuzz`. |

### Local checks already done (Mac, 2026-10-18)

- Grapheme round trip `"".join(split(t)) == t` on all 341,407 manifest transcripts: **0 failures**.
- On a stand-in corpus (v6 + OpenSLR training transcripts only, 0.64 M words), `13_build_units.py`:

  | units | units/word | graphemes/unit | single-unit words | targets/s mean / p99 / max | round-trip failures |
  |---|---|---|---|---|---|
  | Gemma (same text) | 2.77 | 1.15 | 21% | 3.9 / 7.0 / 11.9 | 173 train rows (decode quirks) |
  | graphemes (806) | 3.98 (incl. space unit) | 1.00 | 7.5% | 5.6 / 10.6 / 21.7 | 0 |
  | gbpe3000 | 1.75 | 1.84 | 56% | 2.5 / 5.0 / 10.6 | 0 |

  The encoder emits **25 frames/s**, so even graphemes fit CTC comfortably. These numbers will move on the real
  corpus (web text → pieces fitted to web vocabulary, fewer to speech text); Phase 2 re-measures them.
- 20-step MPS smoke training for grapheme and gbpe units (real audio tower, stand-in text embeddings): loss falls,
  eval + checkpoint save + `units.json` reload work.
- KenLM beam search (`08_lm_decode._decode_one` with the real `si_4gram`) on synthetic peaked log-probs for 30 test
  sentences: 30/30 recovered for both grapheme and gbpe units, i.e. the `▁` word-start handling works with QUAN's
  decoder unchanged.

### Baselines with GER (v6 test, 15,860 utts; `runs/rescore_test.json`)

| system | subset | WER | WER_seg | CER | GER |
|---|---|---|---|---|---|
| A v6x_top8 greedy | all | 19.27 | 13.47 | 4.91 | 6.22 |
| A v6x_top8 + KenLM | all | **14.72** | **8.95** | **3.68** | **4.61** |
| | seen sentences (9,352) | 10.01 | 5.61 | 2.13 | 2.70 |
| | unseen sentences (6,508) | 20.57 | 13.10 | 5.59 | 6.94 |
| Whisper-small run12-best | all | 15.50 | 10.70 | 4.43 | 5.52 |
| | unseen | 21.80 | 15.06 | 6.68 | 8.23 |

Reference dev2k curves (dev CER by step; full logs in `reference/runs/*/metrics.jsonl`) to compare against during
training:

| step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 7500 |
|---|---|---|---|---|---|---|---|---|
| A v6_top6 | 12.12 | 9.38 | 8.40 | 7.60 | 7.19 | 6.95 | 6.77 | **6.76** |
| A v6x_top8 | 7.33 | 7.36 | 6.89 | 6.74 | **6.53** | 6.58 | | |

---

## 3. Setup on the other machine (new Modal account)

This plan runs on a **different Modal account** from the original runs, whose volumes held the packed audio,
manifests, KenLM and checkpoints. Nothing from the old account is needed. What the new account needs comes from
three places:

| what | from | how |
|---|---|---|
| v6 audio + manifests (113 h train, dev2k, test) | Hugging Face `SinhaSpeech/sinhala-asr-data` (public) | `pack_v6` (§3a) |
| SLR-52 extra audio (+58 h) | Hugging Face `SPEAK-ASR/openslr-sinhala-asr` + `reference/openslr52/utt_spk_text.tsv.gz` (OpenSLR-52 speaker table, CC BY-SA 4.0) | `modal volume put` + `slr52_extra` (§3a) |
| word 4-gram KenLM `si_4gram` (exact file) + system A's `v6x_top8/best` | Hugging Face `ovindu-a/sinhala-gemma-asr` (public) | `fetch_reference` (§3a) |
| baseline hypotheses, reference manifests + checksums, A's metrics | **this repo**, `reference/` (compressed, 20 MB) | `scripts/unpack_reference.sh` |

`reference/` contents:

```
reference/SHA256SUMS                         sha256 of the 4 manifests below (as on the original volume)
reference/manifests/ss_v6/{dev2k,test}.jsonl  + train.jsonl.gz       SinhaSpeech v6 manifests (normalized text)
reference/manifests/v6_plus_slr52/train.jsonl.gz                   v6 train + SLR-52 extra (A's v6x_top8 data)
reference/openslr52/utt_spk_text.tsv.gz      OpenSLR-52 utt/speaker/text table (needed by slr52_extra)
reference/baseline/                          A greedy / A + KenLM / Whisper-small test hyps (.gz), their eval json,
                                             rescore_test.json (GER baselines), si_4gram lm_config.json
reference/runs/{v6_top6,v6x_top8}/           A's args.json, config.json, units.json, metrics.jsonl (dev curves)
```

```bash
git clone https://github.com/ovindu-a/sinhala-asr-research.git && cd sinhala-asr-research
# (public copy without the Mac app; the private ovindu-a/sinhala-asr has the same branch plus app/)
git checkout grapheme-units
uv sync                          # local venv: rescore / compare_errors / unpack checks
sh scripts/unpack_reference.sh   # -> data/manifests/..., data/openslr52/..., runs/*.hyps.jsonl; prints 4x OK
uv run scripts/rescore.py runs/v6x_top8_lm_test.hyps.jsonl --train data/manifests/ss_v6/train.jsonl
# must print WER 14.72% / CER 3.68% / GER 4.61% on "all"

pip install modal                # or: uv tool install modal
modal setup                      # log in to the NEW account (creates a token)
modal profile current            # make sure this is the new account before launching anything
```

The volumes (`sinhala-asr-data`, `sinhala-asr-runs`, `sinhala-asr-hf-cache`) are created automatically on first
use. Run every `modal` command from the repo root (the app mounts `scripts/`, `sinhala_asr/` and, for the
punctuation functions, `runs/*.hyps.jsonl`).

### 3a. Phase 0: rebuild the data on the new account (CPU, ~$1.5, ~2 h)

```bash
# 1. OpenSLR-52 speaker table (slr52_extra uses it to keep v6 dev/test speakers out of training)
modal volume put sinhala-asr-data data/openslr52/utt_spk_text.tsv openslr52/utt_spk_text.tsv

# 2. SinhaSpeech v6 -> packed PCM + manifests: /data/{packed,manifests}/ss_v6/{train,dev2k,test}   (~10 min)
modal run modal_app.py::pack_v6

# 3. SLR-52 utterances missing from v6 (minus v6 val/test speakers), appended after v6 train:
#    /data/{packed,manifests}/v6_plus_slr52/{train,dev2k,test}   (~45 min analysis + ~40 min packing)
modal run --detach modal_app.py::slr52_extra

# 4. exact KenLM + A's checkpoint from Hugging Face (~5 min)
modal run modal_app.py::fetch_reference

# 5. gate 0: every manifest byte-identical to the original runs
modal run modal_app.py::verify_data
```

**Gate 0.** `verify_data` must print `OK` for all 4 manifests, plus sizes for `lm.bin` (~0.39 GB), the packed
`train.pcm` files and `model.safetensors`.

- `ss_v6/*` DIFF: the SinhaSpeech dataset changed on Hugging Face since 2026-10-15 (the dataset was at revision
  `371180dcf62b1beb58c99e125d613d5f1960e077` on 2026-10-18). Pin it: add `revision="371180dc…"` to the
  `hf_hub_download` call in `scripts/05_pack_parquet.py`, rerun step 2.
- `v6_plus_slr52/train.jsonl` DIFF while `ss_v6/*` is OK: the SLR-52 mirror changed. Compare row counts with
  `reference/manifests/v6_plus_slr52/train.jsonl.gz` (171,163 rows); if within ~1%, continue and note it in
  NOTES.md (A and the new runs then saw slightly different extra data, so compare B/C's v6x_top8 with A's with that
  caveat; v6_top6 is unaffected).

Optional sanity check of the new account end to end (~$1, H100 ~15 min): score A's downloaded checkpoint on the
rebuilt test set; it must reproduce `reference/baseline/v6x_top8_eval_ss_v6_test.json` (greedy WER 19.27 / CER 4.91):

```bash
modal run modal_app.py::evaluate --ckpt v6x_top8/best --split test
```

Rules:

- **Budget.** The new account has its own credit. Before every job, run `modal billing summary`, add a row to the
  budget table in `NOTES.md`, and only then launch.
- Keep only `last/` and `best/` per run (the training script already does this).
- zsh: pass Modal arguments explicitly (no `$var` word-splitting tricks).
- Never push models to Hugging Face from this plan.

---

## 4. Cost

From the actual Modal bills of the original runs (not list prices):

| step | resources | time | est. cost |
|---|---|---|---|
| Phase 0: `pack_v6` + `slr52_extra` + `fetch_reference` | 8 CPU | ~2 h | ~$1.5 |
| optional: `evaluate` A's checkpoint on the new account | H100 | ~15 min | ~$1 |
| `lm_corpus` (rebuild KenLM text) | 8 CPU | ~30–40 min | ~$0.3–0.5 |
| `grapheme_units` (inventory + units + reports) | 16 CPU | ~20–30 min | ~$0.3–0.5 |
| `*_v6_top6` | H100 | ~55 min | ~$2.75 |
| `*_v6x_top8` | H100 | ~60 min | ~$5 |
| `evaluate` + `lm_dump` | H100, short | ~15 min | ~$1 |
| `lm_decode` (grid + test) | 32 CPU | ~30–60 min | ~$1.5 |
| **C only** (incl. Phase 0) | | | **~$12.5** |
| **C + B** (incl. Phase 0) | | | **~$23–25** |
| optional speed benchmark (per system) | H100 | ~10 min | ~$0.5 |

---

## 5. Phase 1: rebuild the KenLM text corpus (CPU)

`07_build_lm.py` deleted `corpus.txt` after building `si_4gram`, so regenerate it: same sources (Wikipedia si
20231101 + FineWeb-2 sin_Sinh, capped at 150 M web words + v6/SLR-52 training transcripts), same dev/test leak
filter, kept this time.

```bash
modal run --detach modal_app.py::lm_corpus
modal app logs sinhala-asr          # follow progress (or the URL modal prints)
```

Writes `/data/lm/si_corpus/corpus.txt` (~2–3 GB) and `/data/lm/si_corpus/corpus_stats.json`.

**Check** (gate 1): the counts must match the KenLM's.

```bash
mkdir -p runs/grapheme
modal volume get sinhala-asr-data lm/si_corpus/corpus_stats.json runs/grapheme/
python -c "import json;a=json.load(open('runs/grapheme/corpus_stats.json'))['stats'];b=json.load(open('reference/baseline/lm_config.json'))['stats'];print(a==b);print(a);print(b)"
```

`True` → same text as the KenLM (≈ 160 M words). If they differ slightly (e.g. a Hugging Face parquet revision
changed), note the difference in NOTES.md and continue; it does not affect comparability of the ASR runs, which all
use the same `si_4gram` LM.

---

## 6. Phase 2: grapheme alphabet, units, reports (CPU)

```bash
modal run --detach modal_app.py::grapheme_units --vocabs 3000
# C only, skip the plain-grapheme units:  add  --no-grapheme
```

Steps inside the job:

1. `12_grapheme_inventory.py`: counts every word of the corpus, splits each distinct word into graphemes, keeps
   valid graphemes seen in the speech training text or ≥ 50 times in the corpus (+ a–z, 0–9) as the **alphabet**;
   writes `/data/units/graphemes.json` and `/data/units/words.tsv` (corpus words seen ≥ 2 times whose graphemes are
   all in the alphabet).
2. `13_build_units.py --kind grapheme` → `/data/units/grapheme/{units.json,report.json}` (unless `--no-grapheme`).
3. `13_build_units.py --kind gbpe --vocab 3000 --compare-gemma` → `/data/units/gbpe3000/{units.json,sp.model,
   sp.vocab,report.json}`.

Pull the reports:

```bash
mkdir -p runs/grapheme
modal volume get sinhala-asr-data units/graphemes.json runs/grapheme/
modal volume get sinhala-asr-data units/gbpe3000/report.json runs/grapheme/gbpe3000_report.json
modal volume get sinhala-asr-data units/gbpe3000/sp.vocab runs/grapheme/gbpe3000_sp.vocab
modal volume get sinhala-asr-data units/grapheme/report.json runs/grapheme/grapheme_report.json   # if built
```

**Gate 2**: read the job log (or the report files) and check:

| check | where | pass | if it fails |
|---|---|---|---|
| alphabet size | `graphemes.json` → `stats.alphabet` | ~800–1,500 | > 2,000: the validity filter is letting junk through; raise `--min-count` (e.g. 200) and rerun Phase 2 |
| corpus coverage | `stats.corpus_tokens_covered` | ≥ 99.9% | look at `stats.invalid_top` |
| round-trip failures | report → each manifest → `roundtrip_failures` | **0** | do not train; debug `sinhala_asr/graphemes.py` / `units.py` on the failing rows |
| `<unk>` rate | `unk_rate` | < 0.1% of units (typo clusters only) | look at `speech_invalid` in `graphemes.json` |
| merged pieces seen < 20 times in the speech text | log line "merged pieces (2+ graphemes) seen < 20 times" (`merged_rare_lt20_share`) | ≤ 20% | rerun with `--vocabs 2000 --no-grapheme --no-inventory` and use `gbpe2000` everywhere below |
| targets per second | `targets_per_s.p99` | well under 25 (encoder frames/s) | — (expected ~5) |
| units per word vs Gemma | report vs `gemma` block | gbpe < Gemma's ~2.8 | — (informative) |

Also look at `unit_len_hist` in the report (how many pieces are 1, 2, 3, … graphemes long; on the stand-in corpus:
1,029 / 1,103 / 600 / 214 / 45, and 7 longer) and the real-text pieces (`python -c "from sinhala_asr.units import load_units;
u=load_units('units.json'); print(u.pieces[:200])"` after pulling `units/gbpe3000/units.json`). Write the key
numbers into NOTES.md.

---

## 7. Phase 3: `*_v6_top6` (GPU)

C (and B in parallel if you run it; two separate containers):

```bash
modal run --detach modal_app.py::train --name gb3k_v6_top6 --manifests ss_v6 --dev dev2k --unfreeze-top-k 6 \
    --steps 7500 --units gbpe3000 --extra "--batch-seconds 1000 --eval-every 1000"
modal run --detach modal_app.py::train --name gr_v6_top6 --manifests ss_v6 --dev dev2k --unfreeze-top-k 6 \
    --steps 7500 --units grapheme --extra "--batch-seconds 1000 --eval-every 1000"      # B only
```

At start the log must show `units 3,001 (gbpe)` (or `(grapheme)`) and `dev target tokens mapped to <unk>: 0.0x%`.

Monitor: `modal app logs sinhala-asr`, or pull metrics:
`modal volume get sinhala-asr-runs gb3k_v6_top6/metrics.jsonl runs/gb3k_v6_top6/`.
Eval lines now include `GER`.

**Gate 3** (dev2k CER vs A's curve in §2):

- Loss must fall within the first 500 steps (A: ~0.84 at the end). Stuck at > 5 after 1,000 steps → stop
  (`modal app stop`), check the units (`--units` path, unk rate).
- At step 2000, CER within ~1.5 points of A's 9.38 → let it finish. Clearly worse at 3000 (> 1.5 points above
  8.40) → still finish (it's cheap relative to restarting), but expect a negative result.
- Larger units usually converge more slowly at the start; judge at 7500, not 1000.

Preemption/retry: the function always passes `--resume` and Modal retries once; a restarted container continues
from `/runs/<name>/last`. If it hits the 165-minute cap, rerun the same command: it resumes.

---

## 8. Phase 4: `*_v6x_top8` (GPU)

Only after the matching `*_v6_top6` finished (it needs `best/`):

```bash
modal run --detach modal_app.py::train --name gb3k_v6x_top8 --manifests v6_plus_slr52 --dev dev2k \
    --unfreeze-top-k 8 --steps 6000 \
    --extra "--init-from /runs/gb3k_v6_top6/best --lr 1e-4 --warmup 0.03 --batch-seconds 1000 --eval-every 1000"
modal run --detach modal_app.py::train --name gr_v6x_top8 --manifests v6_plus_slr52 --dev dev2k \
    --unfreeze-top-k 8 --steps 6000 \
    --extra "--init-from /runs/gr_v6_top6/best --lr 1e-4 --warmup 0.03 --batch-seconds 1000 --eval-every 1000"  # B
```

Do **not** pass `--units` here: with `--init-from` the units come from the checkpoint (must log the same kind and
size as Phase 3). Compare the dev curve with A's v6x_top8 (best 6.53 at step 5000).

---

## 9. Phase 5: test evaluation and KenLM decoding

For each finished `*_v6x_top8` (shown for C):

```bash
# greedy, all metrics incl. GER, per source; writes /runs/gb3k_v6x_top8/best/eval_ss_v6_test.{json,hyps.jsonl}
modal run modal_app.py::evaluate --ckpt gb3k_v6x_top8/best --split test

# top-16 log-probs per frame for dev2k + test (GPU, short)
modal run modal_app.py::lm_dump --ckpt gb3k_v6x_top8/best

# alpha/beta grid on 1,000 dev utts, then test with the best pair (CPU)
modal run --detach modal_app.py::lm_decode --ckpt gb3k_v6x_top8/best
```

`lm_decode` uses the existing word 4-gram (`/data/lm/si_4gram`) unchanged. Default grid: α ∈ {0.2, 0.35, 0.5, 0.7},
β ∈ {0, 1, 2} (A's best: α 0.35, β 1.0). **If the best pair is on the edge of the grid** (α = 0.7 or β = 2 or
β = 0), extend it once:

```bash
modal run --detach modal_app.py::lm_decode --ckpt gb3k_v6x_top8/best --extra "--alphas 0.5 0.7 0.9 1.1 --betas 1 2 3 4"
```

(Units of different length change how strongly the word LM should be weighted, so the optimum may move.)

Outputs: `/runs/gb3k_v6x_top8/best/lm/eval_lm_test.{json,hyps.jsonl}` (grid included in the json).

Pull everything:

```bash
R=gb3k_v6x_top8   # then the same for gr_v6x_top8
mkdir -p runs/$R
modal volume get sinhala-asr-runs $R/metrics.jsonl runs/$R/
modal volume get sinhala-asr-runs $R/best/eval_ss_v6_test.json runs/$R/
modal volume get sinhala-asr-runs $R/best/eval_ss_v6_test.hyps.jsonl runs/${R}_test.hyps.jsonl
modal volume get sinhala-asr-runs $R/best/lm/eval_lm_test.json runs/$R/
modal volume get sinhala-asr-runs $R/best/lm/eval_lm_test.hyps.jsonl runs/${R}_lm_test.hyps.jsonl
modal volume get sinhala-asr-runs $R/best/units.json runs/$R/
```

(`modal volume get` takes literal arguments; in zsh write the names out if the variable form misbehaves.)

---

## 10. Phase 6: analysis (local, free)

```bash
# all systems, all metrics incl. GER, per source + seen/unseen sentences
uv run scripts/rescore.py runs/v6x_top8_test.hyps.jsonl runs/v6x_top8_lm_test.hyps.jsonl \
    runs/gb3k_v6x_top8_test.hyps.jsonl runs/gb3k_v6x_top8_lm_test.hyps.jsonl runs/whisper_test.hyps.jsonl \
    --train data/manifests/ss_v6/train.jsonl --out runs/rescore_grapheme_test.json

# error categories: grapheme-BPE vs Gemma units, both with KenLM, then both greedy
uv run scripts/compare_errors.py runs/v6x_top8_lm_test.hyps.jsonl runs/gb3k_v6x_top8_lm_test.hyps.jsonl \
    --names gemma_lm gbpe_lm > runs/compare_gemma_vs_gbpe_lm_test.txt
uv run scripts/compare_errors.py runs/v6x_top8_test.hyps.jsonl runs/gb3k_v6x_top8_test.hyps.jsonl \
    --names gemma gbpe > runs/compare_gemma_vs_gbpe_greedy_test.txt
uv run scripts/error_analysis.py runs/gb3k_v6x_top8_test.hyps.jsonl runs/gb3k_v6x_top8_lm_test.hyps.jsonl
```

What to look at:

- Headline: WER / WER_seg / CER / **GER** on all v6 test, greedy and + KenLM, vs A (14.72 / 8.95 / 3.68 / 4.61).
- **Unseen sentences** (6,508): where better units should matter most (A + KenLM: 20.57 / 13.10 / 5.59 / 6.94).
- Error categories: `one_letter` and `near_miss` (should drop if grapheme units help), `spacing` (gbpe pieces
  carry `▁`, so word boundaries may improve), insertions and runaway utterances.
- Greedy gap: A greedy 19.27 → +LM 14.72. If gbpe greedy is much closer to its +LM number, the units themselves
  carry more of the language knowledge.

Optional speed check (H100, ~$0.5; the head is 3,001 wide vs 2,776, and the beam has fewer, longer pieces):

```bash
modal run modal_app.py::benchmark --ckpt gb3k_v6x_top8/best
modal volume get sinhala-asr-runs gb3k_v6x_top8/best/speed.json runs/gb3k_v6x_top8/
```

---

## 11. Phase 7: write-up

Add a "Grapheme units" section to `NOTES.md`:

1. Budget rows for every job + a `modal billing summary` line at the end.
2. Unit stats (alphabet size, units/word, graphemes/unit, single-unit words, rare share) for Gemma / B / C.
3. Results table:

| units | dev2k CER v6_top6 | dev2k CER v6x_top8 (best step) | test greedy WER / WER_seg / CER / GER | test + KenLM (α, β) WER / WER_seg / CER / GER | unseen + KenLM WER / CER / GER |
|---|---|---|---|---|---|
| A Gemma | 6.76 | 6.53 (5000) | 19.27 / 13.47 / 4.91 / 6.22 | 14.72 / 8.95 / 3.68 / 4.61 (0.35, 1.0) | 20.57 / 5.59 / 6.94 |
| C gbpe3000 | | | | | |
| B graphemes | | | | | |

4. The error-category deltas and 3–5 example utterances where the systems differ.

Commit the `runs/` summary files that are small (`*.json`, reports, compare txt) only if you want them in git;
`runs/` is git-ignored by default.

---

## 12. Troubleshooting

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: grapheme_kit` / `sentencepiece` in a Modal job | the image did not rebuild: make sure you are on branch `grapheme-units` (`base_image` lists them) |
| `lm_corpus` very slow on downloads | FineWeb-2 parquet download from HF; the job has a 2 h timeout. Rerun; `hf_hub_download` resumes |
| `grapheme_units` OOM | raise `memory=` in `modal_app.py::grapheme_units` (64 GB default; the word Counter of ~160 M tokens needs a few GB) |
| `13_build_units.py`: SentencePiece "Vocabulary size is smaller than required_chars" | alphabet bigger than `--vocab`: raise `--min-count` in Phase 2 |
| training log `units N (gemma)` | `--units` was not passed / misspelled: stop, delete `/runs/<name>` (`modal volume rm -r sinhala-asr-runs <name>`), relaunch |
| `lm_decode` IndexError | handled: utterances whose frames only have empty pieces fall back to greedy (as for A) |
| Whisper-style runaway repetitions in hyps | count them in `compare_errors.py` output ("runaway"); not expected from CTC |

---

## 13. Next: grapheme n-gram LM for B (decoder only, CPU)

Why: B (single graphemes) beats A on CER / WER_seg / GER but only ties on WER (14.78 vs 14.72), and the gap is word
spacing. The word 4-gram scores a word only once it is complete and gives unseen words a flat penalty, so it cannot
judge බලනකොට vs බලන කොට. A grapheme n-gram over the same text, with "|" as the word break, scores every grapheme
including the spaces.

Code (tested locally with a toy bigram LM and B's real units: 40/40 clean test sentences recovered; on an
acoustically ambiguous space the LM decides it):
- `scripts/14_grapheme_lm.py`: si_corpus → grapheme tokens (`අ ම් මා | ගෙ ද ර`) → `lmplz` → `lm.bin` + `vocab.txt`.
- `scripts/08_lm_decode.py --lm-kind grapheme`: every unit becomes its own "word" for QUAN's decoder (piece
  `"▁"+grapheme`, space unit `"▁|"`), so no new beam search. β becomes a per-grapheme bonus. New `--beam`, `--procs`.
- `modal_app.py::grapheme_lm`: 16 CPU / 128 GB / 512 GiB disk (Modal minimum). `lm_image` now builds KenLM (lmplz + Python module)
  with max order 10; the word 4-gram still loads.

Steps (Modal account `chatgptgroup321`; log each job in the NOTES.md budget table first):

```bash
# 1. grapheme 8-gram over the full si_corpus (~700 M grapheme tokens; est. 1–2 h, ~$2–4)
modal run --detach modal_app.py::grapheme_lm --order 8
#    quicker first version if wanted: --max-words 50000000 --out si_graph8_50m

# 2. decode B: grid on ALL 2,000 dev2k utts (the 1,000-utt grid misled β last time), then test
modal run --detach modal_app.py::lm_decode --ckpt gr_v6x_top8/best --lm si_graph8 \
    --extra "--lm-kind grapheme --tag _glm8 --tune-limit 2000 --alphas 0.3 0.5 0.7 1.0 --betas 0 0.5 1.0 2.0"
#    if the best α or β is on the grid edge, extend once in that direction (same --tag with a suffix)
```

Outputs: `/runs/gr_v6x_top8/best/lm/eval_lm_test_glm8.{json,hyps.jsonl}` (grid inside the json). Compare with
B + word LM (14.78 / 8.33 / 3.41 / 4.29, unseen WER 20.26) and A + word LM (14.72 / 8.95 / 3.68 / 4.61) using
`scripts/rescore.py` and `scripts/compare_errors.py` (watch "spacing" and inserted words). Then sync `lm.bin`,
`vocab.txt`, `lm_config.json` and the new eval files to the HF bucket (`hf://buckets/ovindu-a/sinhala-grapheme-asr/
lm/si_graph8/`, `runs/gr_v6x_top8/best/lm/`) so molab can use them.

If it helps but WER still trails: combine both LMs (needs a decoder change: word LM at "▁|", grapheme LM per unit).

---

## 14. Next: word 4-gram + grapheme 8-gram together for B (decoder only, CPU)

Why: B + grapheme 8-gram (14.26 / 8.05 / 3.20 / 4.06) beats A on everything, but its spacing-only errors rose
(3,441 vs B + word LM 3,295, A 2,797). The grapheme LM knows spelling and typical space positions; the word 4-gram
judges whole words at each boundary. Using both should cut the remaining spacing errors.

Code: `sinhala_asr/dual_lm.py` (DualLMDecoder: QUAN's prefix beam search with two KenLM states per beam; grapheme LM
on every unit, "|" for spaces; word LM when a space closes a word and at the end) and `08_lm_decode.py --lm-kind dual
--glm <dir>` with `--g-alphas/--g-betas` (grapheme LM weights) next to `--alphas/--betas` (word LM weights).
Tested locally: with the grapheme weights at 0 it returns exactly the word-LM decoder's output (40/40), same speed.

Steps (Modal account `chatgptgroup321`; budget row first):

```bash
modal run --detach modal_app.py::lm_decode --ckpt gr_v6x_top8/best --lm si_4gram \
    --extra "--lm-kind dual --glm /data/lm/si_graph8 --tag _dual --tune-limit 2000 \
             --alphas 0.1 0.2 0.35 --betas 0 1.0 --g-alphas 0.3 0.5 --g-betas 0.5 1.0"
```

24 combinations on all 2,000 dev2k utts + test: est. ~45–60 min on 32 CPU, ~$1.5–2. If a best value is on a grid
edge, extend once in that direction with `--tag _dual2`. If the container runs out of memory (two LMs + the
word-prefix set per process), add `--procs 16`.

Compare (`rescore.py`, `compare_errors.py`) with B + grapheme 8-gram (14.26 / 8.05 / 3.20 / 4.06, unseen WER 19.23,
spacing-only 3,441) and A (14.72, spacing-only 2,797). Outputs: `/runs/gr_v6x_top8/best/lm/eval_lm_test_dual.*`.
