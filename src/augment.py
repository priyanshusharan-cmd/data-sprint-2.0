"""
Train-time augmentations that move the training distribution towards the test distribution.

Measured shift (analysis/eda.py):
  * script flip: train 61% Devanagari / 33% Roman, test 33% Devanagari / 64% Roman
  * colloquial register in test only (bhai / please / na / jaldi / hey ~3% of rows each, ~0 in train)
  * digits in test ("convert 5 pm ist to pst"), none in train
  * ASR-style spellings ("mjhe", "volyoom")

All functions are deterministic given an `rng`, rule-based, and use no external data or LLMs.
The official task description itself uses this register: "bhai kal subah 7 baje ka alarm laga de".
"""
import random
import re

from normalize import _CONS  # Devanagari consonant table reused from the Amazon ML Challenge pipeline

DEV = re.compile("[ऀ-ॿ]")
_NUKTA = {"ph": "f", "j": "z", "d": "r", "dh": "rh", "k": "k", "kh": "kh", "g": "g"}
_SIGN = {0x3E: ("aa", "a"), 0x40: ("ee", "i"), 0x42: ("oo", "u"), 0x3F: ("i", "i"), 0x41: ("u", "u"),
         0x47: ("e", "e"), 0x48: ("ai", "ai"), 0x4B: ("o", "o"), 0x4C: ("au", "au"), 0x45: ("e", "e"),
         0x49: ("o", "o"), 0x43: ("ri", "ri"), 0x44: ("ri", "ri")}
_VOW = {0x05: ("a", "a"), 0x06: ("aa", "a"), 0x07: ("i", "i"), 0x08: ("ee", "i"), 0x09: ("u", "u"),
        0x0A: ("oo", "u"), 0x0B: ("ri", "ri"), 0x0F: ("e", "e"), 0x10: ("ai", "ai"), 0x13: ("o", "o"),
        0x14: ("au", "au"), 0x11: ("o", "o"), 0x0D: ("e", "e")}
_DIGIT_DEV = {chr(0x0966 + i): str(i) for i in range(10)}


def _romanize_word(word, long_double):
    """Devanagari word -> Roman in the dataset's style (aa/ee/oo for long vowels, Hindi schwa deletion)."""
    syl = []  # [consonant, vowel, inherent?]
    for ch in word:
        cp = ord(ch)
        if not 0x0900 <= cp <= 0x097F:
            syl.append([ch, "", False])
            continue
        off = cp - 0x0900
        if off in _CONS:
            syl.append([_CONS[off], "a", True])
        elif off == 0x3C and syl:  # nukta
            syl[-1][0] = _NUKTA.get(syl[-1][0], syl[-1][0])
        elif off in _SIGN and syl:
            syl[-1][1], syl[-1][2] = _SIGN[off][0 if long_double else 1], False
        elif off == 0x4D and syl:  # virama
            syl[-1][1], syl[-1][2] = "", False
        elif off in _VOW:
            syl.append(["", _VOW[off][0 if long_double else 1], False])
        elif off in (0x01, 0x02):  # candrabindu / anusvara
            if syl:
                syl[-1][1] += "n"
                syl[-1][2] = False
        elif off == 0x03 and syl:
            syl[-1][1] += "h"
        elif chr(cp) in _DIGIT_DEV:
            syl.append([_DIGIT_DEV[chr(cp)], "", False])
    # schwa deletion, right to left: drop final schwa; drop medial schwa in V C(a) C V
    n = len(syl)
    for i in range(n - 1, -1, -1):
        if not syl[i][2]:
            continue
        if i == n - 1 and n > 1:
            syl[i][1] = ""
        elif 0 < i < n - 1 and syl[i - 1][1] and syl[i + 1][0] and syl[i + 1][1]:
            syl[i][1] = ""
    return "".join(c + v for c, v, _ in syl)


def romanize(text, rng=None, p_long=0.75):
    """Romanize every Devanagari word; long vowels written doubled with prob p_long (per word)."""
    rng = rng or random
    return " ".join(_romanize_word(w, rng.random() < p_long) if DEV.search(w) else w for w in text.split())


# --- colloquial register --------------------------------------------------------------------
PREFIX = ["bhai", "yaar", "hey", "please", "plz", "zara", "ok", "suno", "achha"]
SUFFIX = ["na", "yaar", "bhai", "please", "jaldi", "plz", "jaldi se", "abhi"]


def colloquial(text, rng):
    if rng.random() < 0.5:
        text = f"{rng.choice(PREFIX)} {text}"
    if rng.random() < 0.6:
        text = f"{text} {rng.choice(SUFFIX)}"
    return text


# --- number words -> digits -----------------------------------------------------------------
NUM = {"ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhah": 6,
       "chhe": 6, "chah": 6, "saat": 7, "sat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10, "gyaarah": 11,
       "gyarah": 11, "baarah": 12, "barah": 12, "pandrah": 15, "bees": 20, "tees": 30, "pachaas": 50, "sau": 100,
       "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9,
       "दस": 10, "ग्यारह": 11, "बारह": 12, "पंद्रह": 15, "बीस": 20, "तीस": 30, "पचास": 50, "सौ": 100}
_TIME_CTX = {"baje", "bje", "बजे", "minute", "minat", "मिनट", "ghante", "घंटे", "din", "दिन", "pm", "am"}


def digits(text, rng):
    """Replace number words by digits when followed by a time/quantity word ('saat baje' -> '7 baje')."""
    toks = text.split()
    for i, t in enumerate(toks[:-1]):
        if t in NUM and (toks[i + 1] in _TIME_CTX or rng.random() < 0.3) and t not in ("do", "दो") or (
                t in ("do", "दो") and toks[i + 1] in _TIME_CTX):
            toks[i] = str(NUM[t])
    return " ".join(toks)


# --- ASR / typing noise ---------------------------------------------------------------------
def asr_noise(text, rng, p=0.12):
    out = []
    for w in text.split():
        r = rng.random()
        if len(w) > 3 and r < p and not DEV.search(w):  # drop an inner vowel: mujhe -> mjhe
            idx = [i for i, c in enumerate(w[1:-1], 1) if c in "aeiou"]
            if idx:
                i = rng.choice(idx)
                w = w[:i] + w[i + 1:]
        elif len(w) > 3 and r < 2 * p and not DEV.search(w):  # vowel length swap: ee<->i, oo<->u, aa<->a
            w = rng.choice([w.replace("ee", "i"), w.replace("i", "ee", 1), w.replace("aa", "a"),
                            w.replace("oo", "u"), w.replace("u", "oo", 1)])
        elif len(w) > 3 and r < 2 * p and DEV.search(w):  # drop a matra in Devanagari
            idx = [i for i, c in enumerate(w) if 0x093E <= ord(c) <= 0x094C]
            if idx:
                i = rng.choice(idx)
                w = w[:i] + w[i + 1:]
        elif len(w) > 5 and r < 2.5 * p:  # split word (ASR segmentation)
            i = rng.randint(2, len(w) - 2)
            w = w[:i] + " " + w[i:]
        out.append(w)
    return " ".join(out)


def augment(text, rng, p_roman=0.7, p_coll=0.35, p_digit=0.5, p_noise=0.5):
    """One random view of `text` drawn to look like the test distribution."""
    if DEV.search(text) and rng.random() < p_roman:
        text = romanize(text, rng)
    if rng.random() < p_digit:
        text = digits(text, rng)
    if rng.random() < p_coll:
        text = colloquial(text, rng)
    if rng.random() < p_noise:
        text = asr_noise(text, rng)
    return text


if __name__ == "__main__":
    rng = random.Random(0)
    for s in ["अगली सुबह सात बजे के लिए अलार्म", "मुझे गर्म कॉफी चाहिए", "दिल्ली के आसपास सर्वोत्तम रेस्टोरेंट कौन सा है",
              "मुझे इस कलाकार की आवाज़ से नफ़रत है", "kripaya rasoeeghar ki battiyaan band kar den"]:
        print(romanize(s, random.Random(1)), " | ", augment(s, rng))
