"""
Text normalization for Hindi / Hinglish voice queries.

transliterate_indic() is reused from our Amazon ML Challenge 2026 entity-resolution pipeline
(rule-based Devanagari -> Latin). phonetic_key() folds common Hinglish spelling / ASR variants
("chahoonga" ~ "chahunga", "kya" ~ "kia") so Romanized and Devanagari queries meet in one space.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Indic -> Latin transliteration
# ---------------------------------------------------------------------------
INDIC_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

_CONS = {  # offset -> latin consonant (inherent vowel added separately)
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL = {  # independent vowels
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "l",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "ri", 0x61: "l", 0x50: "om",
}
_SIGN = {  # dependent vowel signs (replace inherent 'a')
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "l", 0x63: "l",
}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}  # candrabindu, anusvara, visarga, gurmukhi tippi
_VIRAMA = 0x4D
_CHILLU = ["n", "n", "r", "l", "l", "k"]
_INDIC_RE = re.compile("[ऀ-ൿ]")


def _indic_offset(ch):
    cp = ord(ch)
    if 0x0900 <= cp <= 0x0D7F:
        return cp & 0x7F  # every block is 0x80 wide
    return None


def transliterate_indic(text: str) -> str:
    if not _INDIC_RE.search(text):
        return text
    out = []
    pending = False  # a consonant waiting for its vowel
    word_len = 0  # Indic code points seen in the current word
    for ch in text:
        off = _indic_offset(ch)
        if off is None:
            if pending and word_len <= 1:
                out.append("a")  # word-final schwa is dropped except in one-letter words
            pending = False
            word_len = 0
            out.append(ch)
            continue
        word_len += 1
        if off in _CONS:
            if pending:
                out.append("a")
            out.append(_CONS[off])
            pending = True
        elif off in _SIGN:
            out.append(_SIGN[off])
            pending = False
        elif off == _VIRAMA:
            pending = False
        elif 0x0D7A <= ord(ch) <= 0x0D7F:  # Malayalam chillu letters: consonant without vowel
            if pending:
                out.append("a")
            out.append(_CHILLU[ord(ch) - 0x0D7A])
            pending = False
        elif off in _VOWEL:
            if pending:
                out.append("a")
                pending = False
            out.append(_VOWEL[off])
        elif off in _NASAL:
            if pending:
                out.append("a")
                pending = False
            out.append(_NASAL[off])
        elif 0x66 <= off <= 0x6F:  # native digits
            if pending:
                out.append("a")
                pending = False
            out.append(str(off - 0x66))
        else:  # nukta, danda, length marks, etc.
            if off in (0x64, 0x65):
                if pending:
                    out.append("a")
                    pending = False
                out.append(" ")
    if pending and word_len <= 1:
        out.append("a")
    return "".join(out)


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PHON = [
    (re.compile(r"(.)\1+"), r"\1"),        # collapse doubled letters: baje/bajje, karna/karrna
    (re.compile(r"ee|ii|ie"), "i"),
    (re.compile(r"oo|uu|ou"), "u"),
    (re.compile(r"aa"), "a"),
    (re.compile(r"w"), "v"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"z"), "j"),
    (re.compile(r"q"), "k"),
    (re.compile(r"c(?!h)"), "k"),
    (re.compile(r"(?<=[a-z])h(?=[^aeiou]|$)"), ""),  # drop non-initial aspiration h: bhai keeps, kah->ka
    (re.compile(r"y(?=[aeiou])"), "i"),     # kya ~ kia
]


def clean(text: str) -> str:
    """Lowercase, NFKC, Devanagari -> Latin, strip punctuation."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = transliterate_indic(t)
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", t).strip()


def phonetic_key(text: str) -> str:
    """clean() plus Hinglish phonetic folding, token by token."""
    out = []
    for tok in clean(text).split():
        if not tok.isdigit():
            for pat, rep in _PHON:
                tok = pat.sub(rep, tok)
        out.append(tok)
    return " ".join(out)


if __name__ == "__main__":
    for s in ["मैं चाहता हूं कि इस घटना को मेरे कैलेंडर से हटा दिया जाए",
              "main chahta hoon ki is ghatna ko mere calendar se hata diya jaye",
              "kia ap imel kholenge", "क्या आप ईमेल खोलेंगे"]:
        print(phonetic_key(s))
