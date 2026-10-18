"""Grapheme clusters for Sinhala CTC units, on top of grapheme-kit (pinned to 0.1.1).

Uses grapheme-kit's GraphemeSplitter directly, not its Graphemizer: Graphemizer first rewrites the text (Tamil fixups,
Sinhala vowel-sign reorderings), so its output does not always join back to the input. Our own `normalize()` stays
the one text normalizer for targets and scoring.

One repair on top of the splitter: it strips the ZWJ from repaya (ර්‍), but `normalize()` keeps ZWJ in every ්‍ය/්‍ර,
so ර්‍ය would come back as ර් + ය without the ZWJ. The dropped ZWJ is glued back and the next cluster joined to it,
which makes ර්‍ය one grapheme like every other ්‍ය conjunct (ත්‍ය, ද්‍යා). After this, "".join(split(t)) == t for
every transcript (checked on all 341k manifest rows).
"""

import re

import grapheme  # grapheme-kit's own dependency
from grapheme_kit.utils.graphemes import GraphemeSplitter

ZWJ = "‍"
_splitter = GraphemeSplitter()

_C = "[ක-ෆ]"  # consonants
_V = "[අ-ඖ]"  # independent vowels
_SIGN = "[ා-ෑි-ුූෘ-ෟෲෳ]"  # dependent vowel signs (not al-lakuna)
_MOD = "[ංඃ]"  # anusvara, visarga
_conj = f"{_C}(?:්{ZWJ}{_C})*"
VALID = re.compile(f"^(?:{_V}{_MOD}?|{_conj}්|{_conj}{_SIGN}?{_MOD}?|[a-z]|[0-9])$")


def _fallback(text: str) -> list[str]:
    """Unicode extended grapheme clusters with every ZWJ-ended cluster joined to the next one (the splitter's rule)."""
    out, glue = [], False
    for g in grapheme.graphemes(text):
        if glue:
            out[-1] += g
        else:
            out.append(g)
        glue = g.endswith(ZWJ)
    return out


def split(text: str) -> list[str]:
    """Grapheme clusters of normalized text; spaces are their own clusters. Joins back to exactly `text`.
    grapheme-kit's splitter raises IndexError on a ZWJ at the very end (model output can have one); those strings
    use the fallback above."""
    try:
        gs = _splitter.split(text)
    except IndexError:
        return _fallback(text)
    if "".join(gs) == text:
        return gs
    out, pos, glue = [], 0, False
    for g in gs:
        if not text.startswith(g, pos):
            return _fallback(text)
        if glue:
            out[-1] += g
            glue = False
        else:
            out.append(g)
        pos += len(g)
        if pos < len(text) and text[pos] == ZWJ:
            out[-1] += ZWJ
            pos += 1
            glue = True
    return out if "".join(out) == text else _fallback(text)


def is_valid(g: str) -> bool:
    """A well-formed Sinhala grapheme (or a Latin letter / digit). Rejects typos like දිි, ලේ් and stray marks."""
    return bool(VALID.match(g))


class WordCache:
    """split() per distinct word; corpora repeat words, so this is ~50x faster than splitting every sentence."""

    def __init__(self):
        self.cache: dict[str, list[str]] = {}

    def word(self, w: str) -> list[str]:
        g = self.cache.get(w)
        if g is None:
            g = self.cache[w] = split(w)
        return g

    def sentence(self, text: str) -> list[str]:
        out = []
        for k, w in enumerate(text.split(" ")):
            if k:
                out.append(" ")
            out.extend(self.word(w))
        return out
