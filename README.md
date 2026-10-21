# Sinhala ASR: EmbeddingGemma 2 audio encoder + CTC reader

Research code and results for Sinhala speech recognition built on Google's **EmbeddingGemma 2** audio encoder, with a
small CTC "reader" on top (after [QUAN-Transcribe](https://github.com/qnguyen3/QUAN-Transcribe), which does the same
for Vietnamese). The main question explored here: **what should the model output?** Gemma tokenizer pieces (the
QUAN default), single grapheme clusters, or grapheme-BPE pieces, and which language model goes with each.

**Short answer:** single graphemes with a grapheme n-gram LM. On sentences no model or LM has seen
(held-out OpenSLR-30, 8.85 h), that system makes **3.3 points fewer word errors and 1.4 fewer character errors**
than the Gemma-token baseline, and both are far ahead of the Whisper-small fine-tune trained on the same data.

| held-out OpenSLR-30 (4,070 utts, new sentences + speakers) | WER | CER |
|---|---|---|
| Whisper-small fine-tune (SinhaSpeech run12-best) | 37.43 | 11.92 |
| A: Gemma-token units + word 4-gram LM | 28.74 | 5.80 |
| **B: grapheme units + grapheme 8-gram LM** | **25.41** | **4.38** |

All numbers on this page use one normalization and one scorer (`sinhala_asr/text.py`, `sinhala_asr/metrics.py`).
The full working log, with the cost of every cloud job, is in [NOTES.md](NOTES.md).

---

## 1. Setup

**Model.** EmbeddingGemma 2's audio tower (frozen, except the top *k* layers) → a learned mix of all its layers → a
6-layer Transformer reader (768-d, 12 heads, ~44 M parameters) → CTC over the output units. Each unit's output row is
initialised from Gemma text embeddings of its text. SpecAugment and dynamic-chunk attention masks during training.
The encoder emits 25 frames per second. Code: `sinhala_asr/model.py`, `scripts/train.py`.

**Data.**
- **SinhaSpeech v6** ([SinhaSpeech/sinhala-asr-data](https://huggingface.co/datasets/SinhaSpeech/sinhala-asr-data),
  `stratified_v6`): 113 h train, speaker-disjoint dev / test (test: 15,860 utterances, 14.7 h). Sources: OpenSLR-52
  (~97%), plus small "linga", "youtube" and "bizbrains" sets. Utterances with digits are dropped from training only.
- **SLR-52 extra**: 48,547 OpenSLR-52 utterances (~58 h, 368 speakers) that v6 left out, after removing the speakers of
  v6 dev/test (identified through sentences only one speaker read): `scripts/06_slr52_extra.py`. v6 + extra = 174 h.
- **Language-model text**: Sinhala Wikipedia + FineWeb-2 (`sin_Sinh`, capped at 150 M words) + the training
  transcripts, 160 M words, with every sentence that contains a v6 dev / test sentence removed (`scripts/07_build_lm.py`).

**Metrics.** WER; **WER_seg** (spacing-agnostic WER: the hypothesis is re-split at the reference's word boundaries, so
only letter errors count); CER (spaces count); **GER** (grapheme error rate: CER over grapheme clusters).

**Important property of v6 test:** 59% of its sentences also occur verbatim in the training transcripts (OpenSLR
prompts repeat across speakers; the split is by speaker, not by sentence). Results are therefore also given for
**seen** (9,352) and **unseen** (6,508) sentences, and on a separately built held-out set (section 5).

## 2. Feasibility and the Gemma-token baseline (A)

| step | result |
|---|---|
| Gemma tokenizer on Sinhala | 3.26 tokens / word, only 6.5% of words are one token: the units are close to characters |
| zero-shot audio → text retrieval with EmbeddingGemma 2 | Sinhala recall@1 82% of 300 (chance 0.3%): the encoder already "knows" Sinhala |
| probe: 20 h OpenSLR, frozen encoder vs top-2 layers unfrozen | dev WER / CER 47.8 / 14.1 vs **44.8 / 12.8** (2,000 steps) |
| v6, top-2 vs top-6 unfrozen (7,500 steps) | dev2k CER 7.67 vs **6.76** |
| + SLR-52 extra, top-8 (`v6x_top8`, 6,000 steps) | dev2k CER **6.53** |
| + word 4-gram KenLM (α 0.35, β 1.0, beam 32) | v6 test **WER 14.72 / CER 3.68** (greedy 19.27 / 4.91); Whisper-small fine-tune 15.50 / 4.43 |

The LM removes 29% of word errors, mostly one-letter errors (Gemma tokens often split a consonant from its vowel
sign). **Speed** (H100, 200 test clips): greedy CTC is **18.5× faster than Whisper-small** batched (RTFx 3,402 vs
184) and 11× one clip at a time; the KenLM beam adds ~2.6 ms per clip on one CPU core. Off-the-shelf
`openai/whisper-small` and `whisper-large-v3-turbo` are unusable on Sinhala (CER > 100%).

## 3. Grapheme units

Three kinds of output units, trained with the same recipe (v6 top-6 → + SLR-52 top-8) and scored the same way:

| units | how | size |
|---|---|---|
| **A** Gemma tokens | QUAN default: every Gemma token id seen in training | 2,774 |
| **B** single graphemes | grapheme clusters ([grapheme-kit](https://pypi.org/project/grapheme-kit/) 0.1.1 splitter + a repair for repaya ZWJ, `sinhala_asr/graphemes.py`), space as its own unit | 1,269 |
| **C** grapheme-BPE | SentencePiece BPE over grapheme symbols (merges never split a grapheme), 3,000 pieces, fitted to the LM text | 3,001 |

Graphemes cannot produce impossible letter sequences (Gemma tokens can, e.g. a vowel sign with no consonant), and every
unit is a pronounceable piece of script. Sinhala graphemes form a small closed set: 99.87% of test graphemes occur
20+ times in the training transcripts.

**Language models for grapheme units.** The word 4-gram works with all units. For B there is also a **grapheme 8-gram**
over the same 160 M-word text (one token per grapheme, `|` for spaces; `scripts/14_grapheme_lm.py`): it scores every
grapheme as it is emitted, including where spaces go, and handles unseen words. A two-LM decoder
(`sinhala_asr/dual_lm.py`) combines both.

**Further experiments.** Training every system 6,000 steps longer from its final checkpoint (fresh LR schedule; A
too, as a fair baseline); a **parts-based output head** for B (each grapheme's row = its own row + shared rows for its
consonant / conjunct, vowel sign and ං/ඃ, so rare graphemes borrow from common ones); and **C redo**, a grapheme-BPE
fitted to speech (alphabet from the speech transcripts only, transcripts weighted 50×).

## 4. Results on v6 test (15,860 utterances)

WER / WER_seg / CER / GER. Bold: best in the column among systems that do not memorize training sentences.

**Whole test set**

| system | LM | WER | WER_seg | CER | GER |
|---|---|---|---|---|---|
| Whisper-small fine-tune | – | 15.50 | 10.70 | 4.43 | 5.52 |
| A | – | 19.27 | 13.47 | 4.91 | 6.22 |
| A longer | – | 18.38 | 12.78 | 4.67 | 5.92 |
| B | – | 19.58 | 11.71 | 4.32 | 5.40 |
| B longer | – | 18.70 | 11.36 | 4.15 | 5.18 |
| B longer + head parts | – | 18.65 | 11.32 | 4.15 | 5.17 |
| C (web-fitted) | – | 19.57 | 13.28 | 5.23 | 6.21 |
| C redo (speech-fitted) | – | 20.06 | 13.38 | 5.45 | 6.47 |
| A | word 4-gram | 14.72 | 8.95 | 3.68 | 4.61 |
| A longer | word 4-gram | 14.26 | 8.67 | 3.61 | 4.52 |
| C (web-fitted) | word 4-gram | 14.73 | 8.99 | 3.74 | 4.58 |
| B | word 4-gram | 14.78 | 8.33 | 3.41 | 4.29 |
| B | grapheme 8-gram | 14.26 | 8.05 | 3.20 | 4.06 |
| B | word + grapheme 8-gram | 14.59 | 7.90 | 3.23 | 4.10 |
| B longer | grapheme 8-gram | 14.11 | **7.86** | **3.13** | **3.97** |
| B longer + head parts | grapheme 8-gram | **13.89** | 7.87 | 3.15 | 3.98 |
| *B + in-domain grapheme LM (memorizes)* | *grapheme 8-gram + training-transcript LM* | *11.77* | *7.18* | *2.88* | *3.59* |

**Seen vs unseen sentences**

| system | LM | seen (9,352): WER / WER_seg / CER / GER | unseen (6,508): WER / WER_seg / CER / GER |
|---|---|---|---|
| Whisper-small fine-tune | – | 10.43 / 7.19 / 2.59 / 3.31 | 21.80 / 15.06 / 6.68 / 8.23 |
| A | – | 15.13 / 10.40 / 3.44 / 4.40 | 24.41 / 17.29 / 6.71 / 8.44 |
| A longer | – | 13.72 / 9.51 / 3.12 / 3.98 | 24.16 / 16.85 / 6.57 / 8.28 |
| B | – | 15.86 / 8.79 / 3.04 / 3.88 | 24.19 / 15.34 / 5.89 / 7.25 |
| B longer | – | 14.68 / 8.29 / 2.81 / 3.57 | 23.70 / 15.17 / 5.81 / 7.15 |
| A | word 4-gram | 10.01 / 5.61 / 2.13 / 2.70 | 20.57 / 13.10 / 5.59 / 6.94 |
| A longer | word 4-gram | 9.13 / 5.03 / 1.91 / 2.43 | 20.62 / 13.20 / 5.69 / 7.07 |
| B | grapheme 8-gram | 10.25 / 5.11 / 1.90 / 2.48 | **19.23** / 11.70 / 4.81 / **5.98** |
| B longer | grapheme 8-gram | 9.71 / 4.76 / 1.78 / 2.32 | 19.57 / 11.70 / **4.79** / **5.98** |
| B longer + head parts | grapheme 8-gram | **9.46 / 4.65 / 1.74 / 2.27** | 19.38 / 11.86 / 4.89 / 6.05 |
| *B + in-domain LM* | | *5.08 / 2.71 / 1.03 / 1.32* | *20.09 / 12.74 / 5.16 / 6.36* |

The in-domain LM (a grapheme 8-gram over the training transcripts only) halves seen-sentence errors and makes unseen
ones worse: it memorizes the repeated prompts. It is shown for completeness, not as a result.

## 5. Results on a held-out set (OpenSLR-30)

[OpenSLR-30](https://openslr.org/30) is Google's studio-recorded Sinhala TTS corpus (CC BY-SA 4.0), here from the HF
mirror [keshan/multispeaker-tts-sinhala](https://huggingface.co/datasets/keshan/multispeaker-tts-sinhala) (5,364
utterances: 2,064 OpenSLR-30 proper, 12 speakers, plus 3,300 from one more speaker, `pn_sin_01`). Removed: every
sentence in the training transcripts or v6 dev / test (863) and every sentence found in the LM text (431). Left:
**4,070 utterances, 8.85 h, 13 speakers**: new sentences, new speakers, studio instead of phone audio
(`scripts/15_slr30_heldout.py`). LM weights are each system's v6-dev choice; nothing is tuned on this set
(`scripts/16_eval_heldout.py`, run on an M2 Mac in ~70 min).

| system | LM | all: WER / WER_seg / CER / GER | OpenSLR-30 (12 spk) | pn_sin_01 (1 spk) |
|---|---|---|---|---|
| Whisper-small fine-tune | – | 37.43 / 22.12 / 11.92 / 13.94 | 33.04 / 16.38 / 5.73 / 7.83 | 38.24 / 23.19 / 13.14 / 15.11 |
| A | – | 36.47 / 20.57 / 7.60 / 9.86 | 31.49 / 19.06 / 5.82 / 7.87 | 37.40 / 20.85 / 7.95 / 10.24 |
| A longer | – | 36.19 / 20.73 / 7.57 / 9.80 | 31.36 / 19.27 / 5.88 / 7.85 | 37.09 / 20.99 / 7.90 / 10.18 |
| B | – | 32.89 / 15.72 / 5.85 / 7.54 | 28.35 / 15.51 / 4.69 / 6.31 | 33.73 / 15.76 / 6.07 / 7.78 |
| B longer | – | 32.73 / 15.84 / 5.84 / 7.55 | 27.92 / 15.28 / 4.63 / 6.24 | 33.62 / 15.94 / 6.08 / 7.80 |
| B longer + head parts | – | 32.62 / 15.88 / 5.83 / 7.53 | 27.51 / 15.33 / 4.56 / 6.18 | 33.57 / 15.99 / 6.07 / 7.79 |
| C (web-fitted) | – | 37.28 / 21.44 / 8.56 / 10.24 | 32.22 / 18.92 / 6.46 / 8.16 | 38.21 / 21.91 / 8.97 / 10.64 |
| C redo (speech-fitted) | – | 38.01 / 22.25 / 9.10 / 10.86 | 34.13 / 20.29 / 7.16 / 8.97 | 38.73 / 22.61 / 9.48 / 11.22 |
| A | word 4-gram | 28.74 / 14.50 / 5.80 / 7.36 | 25.74 / 13.31 / 4.27 / 5.79 | 29.30 / 14.72 / 6.10 / 7.66 |
| A longer | word 4-gram | 28.46 / 14.57 / 5.80 / 7.39 | 25.05 / 13.12 / 4.17 / 5.65 | 29.08 / 14.83 / 6.11 / 7.73 |
| C (web-fitted) | word 4-gram | 29.49 / 15.52 / 6.56 / 7.98 | 26.57 / 13.44 / 4.61 / 6.04 | 30.03 / 15.91 / 6.95 / 8.35 |
| C redo | word 4-gram | 29.48 / 15.98 / 6.88 / 8.30 | 27.58 / 14.26 / 4.92 / 6.45 | 29.83 / 16.30 / 7.26 / 8.65 |
| B | word 4-gram | 26.64 / 12.20 / 4.75 / 6.17 | 24.64 / 12.26 / 3.83 / 5.28 | 27.01 / 12.19 / 4.93 / 6.34 |
| B | word + grapheme 8-gram | 26.14 / **11.37** / 4.50 / 5.87 | 25.31 / 11.51 / 3.77 / 5.21 | 26.29 / **11.35** / 4.64 / 6.00 |
| B | grapheme 8-gram | 25.41 / 11.41 / 4.38 / 5.75 | 22.69 / 11.48 / 3.49 / 4.82 | 25.91 / 11.40 / 4.56 / 5.93 |
| B longer | grapheme 8-gram | 25.29 / **11.35** / **4.34** / **5.71** | 22.93 / **11.38** / 3.47 / 4.81 | 25.73 / **11.35** / **4.52** / **5.88** |
| B longer + head parts | grapheme 8-gram | **25.07** / 11.56 / 4.41 / 5.75 | **21.61** / 11.55 / **3.41** / **4.71** | **25.71** / 11.56 / 4.61 / 5.94 |

Paired bootstrap (95% CI) on this set: longer training changes WER by −0.12 to −0.29 points, significant only for
A + word LM; head parts vs the flat head is within chance without an LM (−0.11 WER, −0.01 CER).

## 6. Conclusions

1. **Grapheme units are the better output for Sinhala.** On v6 test they tie or slightly beat Gemma tokens; on new
   sentences and speakers (held-out) B beats A by 3.6 WER / 1.8 CER without an LM and by 3.3 WER / 1.4 CER with
   each one's best LM. The gain is in spelling (WER_seg, CER, GER); plain WER also depends on word spacing, B's
   weak point, which a grapheme LM mostly fixes.
2. **The LM should match the units.** A grapheme 8-gram suits grapheme units better than the word 4-gram (held-out
   25.41 vs 26.64 WER for B); adding the word LM on top did not help WER.
3. **Merging graphemes into larger pieces (C) is worse at this data size**, whether the pieces are fitted to web text
   or to speech: fewer examples per unit, and one wrong piece costs ~1.7 graphemes.
4. **Training longer mostly helps memorized prompts.** +6,000 steps gave −0.9 greedy WER on v6 test (mostly seen
   sentences) but −0.15 to −0.3 on held-out data, for A and B alike.
5. **The parts-based head is neutral overall**: a few points better on rare graphemes (47% → 52% correct for
   graphemes seen 1–19 times), too few to move WER / CER on these test sets.
6. **Whisper-small (fine-tuned on the same v6 data) does well on v6 test but generalizes poorly**: held-out CER 11.92,
   worst of all, with dropped and looping output on long utterances. The CTC models are also ~18× faster.

## 7. Caveats

- v6 test shares 59% of its sentences with training; compare unseen-sentence or held-out numbers for new speech.
- The held-out set is read speech (studio TTS recordings), 75% from a single speaker; spontaneous speech is untested.
- Every system's LM weights were tuned on v6 dev2k, which has the same seen-sentence overlap as v6 test.
- Training drops utterances with digits, while v6 test keeps them (numbers as digits vs as spoken words).
- Graphemes never seen in training audio cannot be produced (0.06% of v6 test, mostly digits and reference typos).

## 8. Punctuation restoration (separate model)

ASR output has no punctuation; an XLM-R base word tagger adds `, . ?` afterwards (`sinhala_asr/punct*.py`,
`scripts/09-11`), trained on 38 M words of Wikipedia + FineWeb-2 + transcripts. Micro-F1: 0.79–0.80 on held-out
wiki / web text, 0.61 on Gemma-CTC + KenLM transcripts of v6 test (sentence-end F1 0.63; v6 clips are ~3 s
fragments, so many sentence ends are not knowable from the text). See
NOTES.md, "Punctuation restoration".

## 9. Reproducing

| file | purpose |
|---|---|
| `docs/GRAPHEME_PLAN.md` | step-by-step plan for the grapheme experiments on Modal, incl. building all data on a fresh account |
| `docs/MOLAB.md`, `notebooks/molab_*.py` | the longer-training, head-parts, C-redo and decode runs as marimo notebooks for [molab](https://molab.marimo.io) |
| `modal_app.py` | data packing, training, evaluation, LM building and decoding on [Modal](https://modal.com) |
| `scripts/03`–`06` | data preparation (OpenSLR-52, SinhaSpeech v6, SLR-52 extra) |
| `scripts/07`, `14` | word 4-gram / grapheme n-gram KenLM |
| `scripts/train.py`, `evaluate.py`, `08_lm_decode.py` | training, greedy evaluation, KenLM beam search (word, grapheme, dual) |
| `scripts/12`, `13` | grapheme alphabet and grapheme / grapheme-BPE units |
| `scripts/15`, `16` | held-out OpenSLR-30 set and its evaluation (CUDA, Apple MPS or CPU) |
| `scripts/rescore.py`, `compare_errors.py`, `error_analysis.py` | metrics incl. seen / unseen split, error categories |
| `reference/` | manifests with checksums, every system's test hypotheses, training curves, unit reports, held-out results |

Checkpoints: A (`v6x_top8`) and the word 4-gram are public at
[ovindu-a/sinhala-gemma-asr](https://huggingface.co/ovindu-a/sinhala-gemma-asr); the grapheme / grapheme-BPE
checkpoints and the grapheme 8-gram are in a private Hugging Face bucket. A macOS app built on system A is released
separately at [ovindu-a/sinhala-transcribe](https://github.com/ovindu-a/sinhala-transcribe).

Data licences: OpenSLR-52 and OpenSLR-30 CC BY-SA 4.0; Wikipedia CC BY-SA; FineWeb-2 ODC-By; SinhaSpeech v6 as
published by its authors (licence not stated on Hugging Face).
