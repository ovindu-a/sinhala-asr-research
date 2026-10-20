"""CTC prefix beam search over grapheme units with two KenLMs at once (shallow fusion):

  score = log P_ctc(prefix)
        + alpha_g * ln P_graph(graphemes, "|" for each space) + beta_g * n_units      (every unit, as it is emitted)
        + alpha_w * ln P_word(completed words)               + beta_w * n_words      (at each space and at the end)

The grapheme n-gram (14_grapheme_lm.py) knows spelling and where spaces usually go; the word 4-gram (07_build_lm.py)
judges whole words at each boundary. Same beam mechanics as quan_transcribe.lm.KenLMDecoder (blank / repeat
handling, pruning, unknown-word and non-prefix penalties on the word side), so with alpha_g = beta_g = 0 it gives the
word-LM decoder's result. With alpha_w = beta_w = 0 it is close to `08_lm_decode.py --lm-kind grapheme` (which scores
each grapheme one unit later, when the next unit starts) but not identical.

extra_glms: more grapheme n-grams [(path, alpha), ...] added log-linearly on every unit, e.g. one trained on the
speech training transcripts only (14_grapheme_lm.py --manifest), which carries the references' own spelling and
spacing conventions that the 160 M-word web-text LM drowns out. beta_g is shared (one bonus per unit).

pieces: text of every unit (index = unit id): "" for blank/unk, "▁" for the space unit, else the grapheme.
"""

import math

import numpy as np

LN10 = math.log(10.0)
NEG_INF = float("-inf")
SPACE, SEP = "▁", "|"


def _lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


class _Beam:
    __slots__ = ("pb", "pnb", "w_state", "g_state", "lm_score", "partial")

    def __init__(self, w_state, g_state, lm_score: float, partial: str):
        self.pb = NEG_INF
        self.pnb = NEG_INF
        self.w_state = w_state  # word LM state after the completed words
        self.g_state = g_state  # grapheme LM states (one per grapheme LM) after every emitted unit
        self.lm_score = lm_score  # weighted LM scores so far (both LMs, incl. betas)
        self.partial = partial  # graphemes of the word in progress


class DualLMDecoder:
    def __init__(self, word_lm: str, graph_lm: str, pieces: list[str], unigrams: list[str] | None = None,
                 alpha_w: float = 0.35, beta_w: float = 1.0, alpha_g: float = 0.5, beta_g: float = 1.0,
                 unk_offset: float = -10.0, blank: int = 0, extra_glms=()):
        import kenlm

        self.wlm = kenlm.Model(word_lm)
        self.glms = [(kenlm.Model(graph_lm), alpha_g)] + [(kenlm.Model(p), a) for p, a in extra_glms]
        self.pieces = pieces
        self.alpha_w, self.beta_w, self.alpha_g, self.beta_g = alpha_w, beta_w, alpha_g, beta_g
        self.unk_offset, self.blank = unk_offset, blank
        self.unigrams = set(unigrams) if unigrams else None
        self.prefixes = None
        if self.unigrams and (alpha_w or beta_w):
            self.prefixes = {w[:i] for w in self.unigrams for i in range(1, len(w) + 1)}

    def _word(self, state, word: str):
        out = type(state)()
        s = self.wlm.BaseScore(state, word, out) * LN10 * self.alpha_w + self.beta_w
        if self.unigrams is not None and word not in self.unigrams and (self.alpha_w or self.beta_w):
            s += self.unk_offset
        return out, s

    def _graph(self, states: tuple, token: str):
        outs, s = [], self.beta_g
        for (lm, alpha), st in zip(self.glms, states):
            out = type(st)()
            s += lm.BaseScore(st, token, out) * LN10 * alpha
            outs.append(out)
        return tuple(outs), s

    def _partial_penalty(self, partial: str) -> float:
        if not partial or self.prefixes is None or partial in self.prefixes:
            return 0.0
        return self.unk_offset

    def decode(self, log_probs: np.ndarray, beam_width: int = 32, token_min_logp: float = -5.0,
               beam_prune_logp: float = -10.0) -> list[int]:
        """log_probs: (T, V) CTC log-probabilities of one utterance. Returns the best unit-id sequence."""
        import kenlm

        ws = kenlm.State()
        self.wlm.BeginSentenceWrite(ws)
        gs = []
        for lm, _ in self.glms:
            st = kenlm.State()
            lm.BeginSentenceWrite(st)
            gs.append(st)
        root = _Beam(ws, tuple(gs), 0.0, "")
        root.pb = 0.0
        beams: dict[tuple, _Beam] = {(): root}

        for lp in log_probs:
            cands = np.flatnonzero(lp >= token_min_logp)
            if len(cands) == 0:
                cands = np.array([int(lp.argmax())])
            new: dict[tuple, _Beam] = {}

            def child(prefix: tuple, parent: _Beam, unit: int) -> _Beam:
                key = prefix + (unit,)
                if key not in new:
                    old = beams.get(key)  # same prefix last frame: LM part is unchanged
                    if old is not None:
                        new[key] = _Beam(old.w_state, old.g_state, old.lm_score, old.partial)
                    elif self.pieces[unit] == SPACE:
                        w_state, score = parent.w_state, parent.lm_score
                        if parent.partial:
                            w_state, s = self._word(w_state, parent.partial)
                            score += s
                        g_state, s = self._graph(parent.g_state, SEP)
                        new[key] = _Beam(w_state, g_state, score + s, "")
                    else:
                        g = self.pieces[unit]
                        g_state, s = self._graph(parent.g_state, g)
                        new[key] = _Beam(parent.w_state, g_state, parent.lm_score + s, parent.partial + g)
                return new[key]

            def same(prefix: tuple, beam: _Beam) -> _Beam:
                if prefix not in new:
                    new[prefix] = _Beam(beam.w_state, beam.g_state, beam.lm_score, beam.partial)
                return new[prefix]

            for prefix, beam in beams.items():
                total = _lse(beam.pb, beam.pnb)
                for c in cands:
                    c = int(c)
                    p = float(lp[c])
                    if c == self.blank:
                        nb = same(prefix, beam)
                        nb.pb = _lse(nb.pb, total + p)
                    elif prefix and c == prefix[-1]:
                        nb = same(prefix, beam)  # repeated unit without a blank in between: collapses
                        nb.pnb = _lse(nb.pnb, beam.pnb + p)
                        ch = child(prefix, beam, c)  # repeated unit after a blank: a new unit
                        ch.pnb = _lse(ch.pnb, beam.pb + p)
                    elif self.pieces[c]:
                        ch = child(prefix, beam, c)
                        ch.pnb = _lse(ch.pnb, total + p)

            scored = sorted(
                ((_lse(b.pb, b.pnb) + b.lm_score + self._partial_penalty(b.partial), k) for k, b in new.items()),
                reverse=True,
            )
            best = scored[0][0]
            beams = {k: new[k] for s, k in scored[:beam_width] if s >= best + beam_prune_logp}

        def final(item):  # finish the last word and every LM's sentence
            prefix, b = item
            w_state, score = b.w_state, b.lm_score
            if b.partial:
                w_state, s = self._word(w_state, b.partial)
                score += s
            end = type(w_state)()
            score += self.wlm.BaseScore(w_state, "</s>", end) * LN10 * self.alpha_w
            for (lm, alpha), st in zip(self.glms, b.g_state):
                score += lm.BaseScore(st, "</s>", type(st)()) * LN10 * alpha
            return _lse(b.pb, b.pnb) + score

        return list(max(beams.items(), key=final)[0])
