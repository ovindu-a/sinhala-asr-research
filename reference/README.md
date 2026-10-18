# reference/

Small files that let the grapheme-unit plan (docs/GRAPHEME_PLAN.md) run on a fresh Modal account and be compared
with the original Gemma-unit runs. Unpack with `sh scripts/unpack_reference.sh`.

Sources: transcripts from SinhaSpeech/sinhala-asr-data (Hugging Face; licence listed as unknown) and OpenSLR-52
(Kjartansson et al., CC BY-SA 4.0). Hypotheses are outputs of the original v6x_top8 model and of
SinhaSpeech/whisper-small-sinhala-run12-best on the v6 test set.
