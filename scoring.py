"""
Kol Emes / Posuk Practice - word-by-word Hebrew reading scorer.

Pure functions only (no Flask, no OpenAI), so they can be tested offline
with `python -m unittest test_scoring.py` without paying for API calls.

How a word is compared
----------------------
Whisper writes Hebrew WITHOUT nikud, in modern spelling, so the expected
posuk and the transcription are both reduced to a "sound skeleton" first:

  * nikud, trop, punctuation removed; final letters -> regular letters
  * any form of the Name (ה׳, הויה, יהוה, אדני, אדוני, אדושם, השם) -> one token
  * sound-alike letters merged: ב/פ/ו -> P,  ח/כ/ק -> K,  ת/ס/ט -> S
    (Ashkenazi "sav", and Whisper's habit of writing ק/ט for כ/ת sounds)
  * vowel letters are handled by POSITION, not dropped everywhere:
      - first letter א/ה/ע -> "H"  (keeps prefixes like ה- "the", א- "I will")
      - last letter  א/ה/ע -> "A"  (an "a"/"e" ending: נא = נה)
      - first/last י stays (yud prefix = "he will", yud suffix = "my"/plural)
      - first/last ו stays (ו- "and", -ו "his"/"they")
      - in the MIDDLE of a word, א ה ע י ו are dropped, because Whisper
        adds or leaves out these vowel letters freely (אהל / אוהל,
        אברהם / אברם / אברוהם)

Then a word passes only if:
  * its skeleton equals what was heard, or
  * the word is long (5+ skeleton letters) and differs by at most one
    letter (two for 8+), with the FIRST and LAST letters still matching -
    so a dropped/added prefix or suffix (ויאמר vs ויאמרו, אליו vs אלי)
    is always a mistake.

Limitation (by design, not fixable here): because Whisper outputs no
vowels, a reading with wrong nikud but the right consonants (וַיַּרְא for
וַיֵּרָא) cannot be detected by this scorer.
"""

import re
from difflib import SequenceMatcher

MAQAF = "־"

# Sefaria ends some pesukim with a section-break marker, {פ} or {ס}. It is
# layout, not a word to read, but it was being scored as a word the student
# skipped (a perfect reading of Bereishis 19:38 scored 94%).
SECTION_MARKER = re.compile(r'\{\s*[פס]\s*\}')


def strip_section_markers(text):
    return re.sub(r'\s+', ' ', SECTION_MARKER.sub(' ', text or '')).strip()


SHEM = "#"  # single-character token standing for any form of the Name
# Forms of the Name as written in the posuk or as Whisper may write what
# the child said ("Ado-noy", "Ado-shem", "Hashem"), with an optional
# one-letter prefix (ו "and", ל "to", מ "from", ב "in", כ "like", ה).
_SHEM_FORMS = ("יהוה", "הויה", "אדני", "אדוני", "אדושם", "אדשם", "השם")
_SHEM_RE = re.compile(r'^([ובלכמה]?)(?:' + "|".join(_SHEM_FORMS) + r')$')
_SHEM_ABBREV_RE = re.compile(r'^([ובלכמ]?)ה[׳\'׳]$')

_FINALS = str.maketrans("םןץףך", "מנצפכ")
_MERGE = str.maketrans({
    "ב": "P", "פ": "P", "ו": "P",
    "ח": "K", "כ": "K", "ק": "K",
    "ת": "S", "ס": "S", "ט": "S",
})
_GUTTURALS = "אהע"
_MEDIAL_VOWEL_LETTERS = set("אהעיו")


def _letters_only(w):
    return re.sub(r'[^א-ת]', '', w)


def normalize_word(w):
    """Sound skeleton of one word (see module docstring)."""
    w = (w or "").strip()
    m = _SHEM_ABBREV_RE.match(re.sub(r'[֑-ׇ]', '', w))
    if m:
        return (m.group(1).translate(_FINALS).translate(_MERGE)) + SHEM
    letters = _letters_only(w)
    if not letters:
        return ""
    m = _SHEM_RE.match(letters)
    if m:
        return m.group(1).translate(_MERGE) + SHEM
    letters = letters.translate(_FINALS)

    out = []
    last = len(letters) - 1
    for i, ch in enumerate(letters):
        if i == 0:
            # ה at the start is usually "the" - keep it apart from א/ע
            out.append("h" if ch == "ה" else ("H" if ch in _GUTTURALS else ch))
        elif i == last:
            out.append("A" if ch in _GUTTURALS else ch)
        elif ch == "י" and i == last - 1 and letters[last] == "ו":
            out.append(ch)  # -יו ending ("his"): בניו is not בנו
        elif ch in _MEDIAL_VOWEL_LETTERS:
            continue
        else:
            out.append(ch)
    return "".join(out).translate(_MERGE)


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _allowed_edits(n):
    if n <= 4:
        return 0
    if n <= 7:
        return 1
    return 2


def skeletons_match(expected_norm, heard_norm, require_first=True):
    """True if a heard skeleton counts as the expected one."""
    if not expected_norm or not heard_norm:
        return False
    if expected_norm == heard_norm:
        return True
    allowed = _allowed_edits(len(expected_norm))
    if not allowed:
        return False
    if expected_norm[-1] != heard_norm[-1]:
        return False
    if require_first and expected_norm[0] != heard_norm[0]:
        return False
    return _edit_distance(expected_norm, heard_norm) <= allowed


def words_match(expected_word, heard_word):
    return skeletons_match(normalize_word(expected_word), normalize_word(heard_word))


# Words that are so short / phonetically weak that they routinely blend
# into the next word in fast connected reading, so Whisper often drops
# them. Flagging them punishes fluent reading, so they get a free pass.
ELIDABLE_WORDS = {"את"}
ELIDABLE_NORMS = {normalize_word(w) for w in ELIDABLE_WORDS}


def _split(text):
    return (text or "").replace(MAQAF, " ").split()


def _fused_match(first_raw, second_raw, heard_raw):
    """Whisper sometimes writes two adjacent words as one token
    (יקח נא -> יקחנא, פתח האהל -> פתחהאל). Credit both words only if the
    heard token sounds like the two words TOGETHER - never because one
    short word happens to appear somewhere inside a longer token."""
    joined = normalize_word(_letters_only(first_raw) + _letters_only(second_raw))
    return skeletons_match(joined, normalize_word(heard_raw), require_first=False)


def score_words_detailed(expected_text, heard_text):
    """Returns (score_percent, [(word, is_match), ...])."""
    exp_pairs = [(r, normalize_word(r)) for r in _split(expected_text)]
    exp_pairs = [(r, n) for r, n in exp_pairs if n]
    heard_pairs = [(r, normalize_word(r)) for r in _split(heard_text)]
    heard_pairs = [(r, n) for r, n in heard_pairs if n]

    total = len(exp_pairs)
    results = [False] * total
    matcher = SequenceMatcher(None, [n for _, n in exp_pairs],
                              [n for _, n in heard_pairs], autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i1, i2):
                results[k] = True
            continue
        if tag == 'insert':
            # extra words that weren't in the posuk - not counted either way
            continue
        # 'replace' or 'delete': try to pair each expected word in the
        # span with a nearby heard word in the same span.
        used = set()
        span_len = i2 - i1
        for off in range(span_len):
            k = i1 + off
            e_raw, e_norm = exp_pairs[k]
            if results[k]:
                continue
            if e_norm in ELIDABLE_NORMS:
                results[k] = True
                continue
            # positional candidate first, then its neighbours (±1)
            pos = min(j1 + off, j2 - 1)
            candidates = [p for p in (pos, pos - 1, pos + 1) if j1 <= p < j2]
            for p in candidates:
                if p in used:
                    continue
                if skeletons_match(e_norm, heard_pairs[p][1]):
                    results[k] = True
                    used.add(p)
                    break
                # fused with the next / previous expected word?
                h_raw = heard_pairs[p][0]
                if k + 1 < total and _fused_match(e_raw, exp_pairs[k + 1][0], h_raw):
                    results[k] = results[k + 1] = True
                    used.add(p)
                    break
                if k - 1 >= 0 and _fused_match(exp_pairs[k - 1][0], e_raw, h_raw):
                    results[k] = results[k - 1] = True
                    used.add(p)
                    break

    matched = sum(results)
    score = round((matched / total) * 100) if total else 0
    return score, [(exp_pairs[k][0], results[k]) for k in range(total)]


def score_text(expected_text, actual_text):
    expected_text = strip_section_markers(expected_text)
    if not expected_text:
        return None, []
    return score_words_detailed(expected_text, actual_text)


def score_single_word(expected_word, heard_text, context=""):
    """For the "redo just this word" button. True only if a heard token is
    the target word, or the target fused with its neighbour from the
    context window that student.html sends."""
    target_norm = normalize_word(expected_word)
    if target_norm in ELIDABLE_NORMS:
        return True
    heard = [h for h in _split(heard_text) if normalize_word(h)]
    if any(skeletons_match(target_norm, normalize_word(h)) for h in heard):
        return True
    ctx = _split(context)
    idx = next((i for i, w in enumerate(ctx) if normalize_word(w) == target_norm), None)
    if idx is None:
        return False
    for h in heard:
        if idx + 1 < len(ctx) and _fused_match(ctx[idx], ctx[idx + 1], h):
            return True
        if idx > 0 and _fused_match(ctx[idx - 1], ctx[idx], h):
            return True
    return False
