#!/bin/sh
# Unpack reference/ (committed, compressed) into the git-ignored data/ and runs/ dirs the scripts expect.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/manifests/ss_v6 data/manifests/v6_plus_slr52 data/openslr52 runs
cp reference/manifests/ss_v6/dev2k.jsonl reference/manifests/ss_v6/test.jsonl data/manifests/ss_v6/
gunzip -c reference/manifests/ss_v6/train.jsonl.gz > data/manifests/ss_v6/train.jsonl
gunzip -c reference/manifests/v6_plus_slr52/train.jsonl.gz > data/manifests/v6_plus_slr52/train.jsonl
gunzip -c reference/openslr52/utt_spk_text.tsv.gz > data/openslr52/utt_spk_text.tsv
for f in v6x_top8_test v6x_top8_lm_test whisper_test; do
  gunzip -c reference/baseline/$f.hyps.jsonl.gz > runs/$f.hyps.jsonl
done
(cd data/manifests && shasum -a 256 -c ../../reference/SHA256SUMS)
echo "unpacked into data/ and runs/"
