# reference/grapheme/

Results of the grapheme-unit runs (docs/GRAPHEME_PLAN.md), Modal account chatgptgroup321, 2026-10-18. Write-up and
budget: NOTES.md, "Grapheme-level units". Checkpoints stay on the `sinhala-asr-runs` volume (not pushed anywhere).

| dir | what |
|---|---|
| `training/<name>/metrics.jsonl` | training log + dev2k evals for gb3k_v6_top6, gr_v6_top6, gb3k_v6x_top8, gr_v6x_top8 |
| `training/*_v6x_top8/eval_ss_v6_test.json` | greedy test scores (overall + per source) |
| `training/*_v6x_top8/eval_lm_test.json` | KenLM α/β grid on 1,000 dev2k utts + test scores (B: α 0.35, β 2.0) |
| `training/gr_v6x_top8/eval_lm_test_space.json` | B retune: α × β × space penalty on all 2,000 dev2k; picked α 0.35, β 1.0, no penalty |
| `training/gr_v6x_top8/eval_lm_test_glm8.json` | B + grapheme 8-gram (`--lm-kind grapheme`), α × β grid on all 2,000 dev2k + test; picked α 0.5, β 1.0 (2026-10-20) |
| `training/gr_v6x_top8/eval_lm_test_dual{,2}.json` | B + word 4-gram + grapheme 8-gram (`--lm-kind dual`), grid on all 2,000 dev2k + test; `_dual2` = edge extension; picked α 0.2, β 1.0, gα 0.3, gβ 1.0 (2026-10-20) |
| `training/gr_v6x_top8/eval_lm_test_tr{,2}.json` | B + si_graph8 + in-domain tr_graph8 (training transcripts), word LM off; `_tr2` = edge extension, dev pick gα 0.2, gβ 1.5, g2α 0.7 (2026-10-20). Read seen/unseen: analysis/rescore_tr_test.json |
| `units/` | grapheme alphabet (`graphemes.json`), units.json for B and C, Phase 2 reports, SentencePiece vocab, corpus stats |
| `hyps/*.hyps.jsonl.gz` | test hypotheses: C greedy / + KenLM, B greedy / + KenLM β 2 (`lm`) / β 1 (`lm_b1`) / + grapheme 8-gram (`glm8`) / + both LMs (`dual`) / + in-domain grapheme LM (`tr`, `tr2`) |
| `analysis/` | `rescore.py` (all metrics, per source, seen/unseen) and `compare_errors.py` outputs |

Unpack the hypotheses next to the baselines: `for f in reference/grapheme/hyps/*.gz; do gunzip -c $f > runs/$(basename $f .gz); done`

v6 test + KenLM, WER / WER_seg / CER / GER: A Gemma 14.72 / 8.95 / 3.68 / 4.61, C gbpe3000 14.73 / 8.99 / 3.74 / 4.58,
B graphemes (β 1) 14.78 / 8.33 / 3.41 / 4.29.
B + grapheme 8-gram (α 0.5, β 1.0; 2026-10-20): 14.26 / 8.05 / 3.20 / 4.06.
B + word 4-gram + grapheme 8-gram (α 0.2, β 1.0, gα 0.3, gβ 1.0): 14.59 / 7.90 / 3.23 / 4.10.
B + si_graph8 + tr_graph8 (dev pick, `_tr2`): 11.77 / 7.18 / 2.88 / 3.59 overall, but seen 5.08 / unseen 20.09 WER (si_graph8 alone: 10.25 / 19.23).
