"""Re-score *.hyps.jsonl files (ref/hyp per utterance) with the current metrics, incl. GER, overall / per source and
split into test sentences that do / do not occur verbatim in the training manifest. CPU only, runs locally.

  uv run scripts/rescore.py runs/v6x_top8_lm_test.hyps.jsonl runs/whisper_test.hyps.jsonl \
      --train data/manifests/ss_v6/train.jsonl --out runs/rescore_test.json
"""

import argparse
import collections
import json

from sinhala_asr.metrics import score


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps", nargs="+")
    ap.add_argument("--train", default=None, help="train manifest for the seen/unseen-sentence split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seen = set()
    if args.train:
        seen = {r["text"] for r in load(args.train)}

    res = {}
    print(f"{'file':45s} {'subset':14s} {'utts':>6s} {'WER':>7s} {'WER_seg':>8s} {'CER':>7s} {'GER':>7s}")
    for path in args.hyps:
        rows = load(path)
        groups = collections.defaultdict(list)
        for r in rows:
            groups["all"].append(r)
            groups[f"src:{r.get('src') or 'all'}"].append(r)
            if args.train:
                groups["seen" if r["ref"] in seen else "unseen"].append(r)
        res[path] = {}
        for g, rs in sorted(groups.items(), key=lambda x: (x[0] != "all", x[0])):
            m = score([r["ref"] for r in rs], [r["hyp"] for r in rs])
            res[path][g] = m
            print(f"{path[-45:]:45s} {g[:14]:14s} {m['utts']:6d} {m['wer']:7.2%} {m['wer_seg']:8.2%} {m['cer']:7.2%} "
                  f"{m['ger']:7.2%}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
