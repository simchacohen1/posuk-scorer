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
      - first letter א/ה/ע -> "H"
      - last letter  א/ה/ע/י -> "A"  (a vowel ending: נא = נה, and the
        Ashkenazi "hinei" written הני = הנה)
      - first י and first/last ו stay (ו- "and", -ו "his"/"they",
        י- "he will")
      - in the MIDDLE of a word, א ה ע י ו are dropped, because Whisper
        adds or leaves out these vowel letters freely (אהל / אוהל,
        אברהם / אברם / אברוהם)
  * "Hashem" misheard as חשם / השן still counts as the Name
  * "Elokim" (אלוקים, אלקינו...) is the same word as אלהים, אלהינו...
  * a שׂ (sin, marked by its dot in the posuk) may be heard as ש or ס;
    a שׁ (shin) heard as ס is still a mistake

Then a word passes only if:
  * its skeleton equals what was heard, or
  * the word has 5+ skeleton letters and differs by one added or missing
    letter (5-6 letters), or by one changed letter / two added or missing
    letters (7+), with the FIRST letter and LAST TWO letters still
    matching - so a changed ending (ויאמר vs ויאמרו, אבותיכם vs
    אבותיהם) or a dropped ו "and" is always a mistake.
  * Short words (4 skeleton letters or fewer) must match exactly: one
    changed letter there is usually a different word (וישלח "sent" vs
    וישכח "forgot", וילך vs וימלך, ידע vs ירא).

את is not required (fast readers blend it into the next word) - but only
when the student read the words on both sides of it, so silence, an
unrelated sentence, or stopping just before it gets no credit for it.

Limitations (by design, not fixable from Whisper's text):
  * wrong nikud with the right letters (וַיַּרְא for וַיֵּרָא) passes,
    because Whisper outputs no reliable vowels;
  * a dropped ה "the" (אהל for האהל) and בנו for בניו pass, because
    Whisper itself writes these inconsistently for correct readings.
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
SIN_DOT = chr(0x05C2)  # the dot that makes ש a sin (שׂ)
_NIKUD_AND_TROP = chr(0x0591) + "-" + chr(0x05C7)
# "Elokim": the ה of אלהים / אלהינו read (and written) as ק.
_ELOKIM_RE = re.compile(r'אלו?ק')


def _letters_only(w):
    return re.sub(r'[^א-ת]', '', w)


def _sin_as_samech(w):
    """The word with every dotted sin (שׂ) written as ס."""
    return re.sub('ש([' + _NIKUD_AND_TROP + ']*)', lambda m: ('ס' if SIN_DOT in m.group(1) else 'ש') + m.group(1), w)


def word_variants(w):
    """All skeletons a word may be read as: just normalize_word(w), plus
    the ס reading when the word has a dotted sin."""
    first = normalize_word(w)
    if SIN_DOT not in (w or ""):
        return (first,)
    second = normalize_word(_sin_as_samech(w))
    return (first,) if second == first else (first, second)


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
    letters = _ELOKIM_RE.sub('אלה', letters.translate(_FINALS))
    # Whisper often mishears "Hashem" as חשם or השן (seen in saved attempts)
    if len(letters) == 3 and letters[0] in "הח" and letters[1] == "ש" and letters[2] in "מנ":
        return SHEM

    out = []
    last = len(letters) - 1
    for i, ch in enumerate(letters):
        if i == 0:
            out.append("H" if ch in _GUTTURALS else ch)
        elif i == last:
            # Ashkenazi tzere at the end ("hinei") is written with a yud,
            # so a final yud is the same vowel ending as final ה/א/ע.
            out.append("A" if ch in _GUTTURALS or ch == "י" else ch)
        elif ch in _MEDIAL_VOWEL_LETTERS:
            continue
        else:
            out.append(ch)
    return "".join(out).translate(_MERGE)


def _indel_distance(a, b):
    """Letters added + letters missing (a changed letter counts as 2)."""
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return len(a) + len(b) - 2 * prev[-1]


def _allowed_edits(n):
    if n <= 4:
        return 0
    if n <= 6:
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
    if expected_norm[-2:] != heard_norm[-2:]:
        return False
    if require_first and expected_norm[0] != heard_norm[0]:
        return False
    return _indel_distance(expected_norm, heard_norm) <= allowed


def variants_match(expected_variants, heard_variants, require_first=True):
    return any(skeletons_match(e, h, require_first)
               for e in expected_variants for h in heard_variants)


def words_match(expected_word, heard_word):
    return variants_match(word_variants(expected_word), word_variants(heard_word))


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
    if SHEM in normalize_word(first_raw) or SHEM in normalize_word(second_raw):
        return False  # ה׳ is an abbreviation; its letters can't be joined
    joined = word_variants(_sin_dots_kept(first_raw) + _sin_dots_kept(second_raw))
    return variants_match(joined, word_variants(heard_raw), require_first=False)


def _sin_dots_kept(w):
    """Letters of a word, keeping the sin dot so word_variants can see it."""
    return re.sub('[^א-ת' + SIN_DOT + ']', '', w)


def score_words_detailed(expected_text, heard_text):
    """Returns (score_percent, [(word, is_match), ...])."""
    exp_pairs = [(r, normalize_word(r)) for r in _split(expected_text)]
    exp_pairs = [(r, n) for r, n in exp_pairs if n]
    heard_pairs = [(r, normalize_word(r)) for r in _split(heard_text)]
    heard_pairs = [(r, n) for r, n in heard_pairs if n]
    exp_variants = [word_variants(r) for r, _ in exp_pairs]
    heard_variants = [word_variants(r) for r, _ in heard_pairs]

    total = len(exp_pairs)
    results = [False] * total
    elided = []  # את words not heard; free only if both neighbours were read
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
                elided.append(k)
                continue
            # positional candidate first, then nearby heard words (up to 3
            # away), since Whisper's word count rarely lines up exactly
            pos = j1 + off
            candidates = sorted((p for p in range(j1, j2) if abs(p - pos) <= 3),
                                key=lambda p: abs(p - pos))
            for p in candidates:
                if p in used:
                    continue
                # fused with the next / previous expected word? Checked
                # first, so a long word doesn't claim a two-word token alone
                h_raw = heard_pairs[p][0]
                if k + 1 < total and _fused_match(e_raw, exp_pairs[k + 1][0], h_raw):
                    results[k] = results[k + 1] = True
                    used.add(p)
                    break
                if k - 1 >= 0 and _fused_match(exp_pairs[k - 1][0], e_raw, h_raw):
                    results[k] = results[k - 1] = True
                    used.add(p)
                    break
                if variants_match(exp_variants[k], heard_variants[p]):
                    results[k] = True
                    used.add(p)
                    break

    # A blended את sits between two words the student read; one stopping
    # right before it, or reading nothing, gets no credit for it.
    for k in elided:
        neighbours = [n for n in (k - 1, k + 1) if 0 <= n < total]
        if not results[k] and neighbours and all(results[n] for n in neighbours):
            results[k] = True

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
    heard = [h for h in _split(heard_text) if normalize_word(h)]
    # No free pass for את here: the student is saying just this word, so
    # silence or a different word must not count.
    if any(words_match(expected_word, h) for h in heard):
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
