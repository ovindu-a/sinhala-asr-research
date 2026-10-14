"""Transcript normalization for Sinhala CTC targets and scoring.

Same function for training targets and for WER/CER references, so spelling variants that sound identical do not
count as errors:
  - Unicode NFC
  - ZWJ kept only in yansaya/rakaransaya (්‍ය, ්‍ර), and added there when it is missing; every other ZWJ/ZWNJ is
    dropped (touching letters and repaya are rendering choices, not different speech)
  - punctuation removed, Latin lowercased (code-mixed English stays as English)
  - whitespace collapsed
Digits are left in place; callers decide whether to drop those utterances (has_digits) until numbers are verbalized.
"""

import re
import unicodedata

ZWJ, ZWNJ = "‍", "‌"
AL = "්"  # ් al-lakuna (virama)
_YR = "යර"

_conjunct = re.compile(f"{AL}{ZWJ}?([{_YR}])")
_zw = re.compile(f"[{ZWJ}{ZWNJ}]")
_keep = re.compile(r"[^඀-෿a-z0-9‍ ]")
_ws = re.compile(r"\s+")
_digits = re.compile(r"[0-9]")
_sinhala = re.compile(r"[඀-෿]")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = _zw.sub("", t)
    t = _conjunct.sub(lambda m: AL + ZWJ + m.group(1), t)
    t = _keep.sub(" ", t)
    return _ws.sub(" ", t).strip()


def has_digits(text: str) -> bool:
    return bool(_digits.search(text))


def is_sinhala(text: str) -> bool:
    return bool(_sinhala.search(text))
