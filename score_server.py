"""
Kol Emes - scoring server (cloud transcription version).

Run this once (python score_server.py) and leave the window open in the
background. It waits for recordings sent from assignment_recorder.html,
sends each one to OpenAI's hosted Whisper API for transcription, scores
it, and logs the result to scores_log.csv in this same folder.

Requires:  pip install flask openai

Before starting, set your OpenAI API key as an environment variable
named OPENAI_API_KEY (do NOT paste the key directly into this file).

  Windows (PowerShell):   setx OPENAI_API_KEY "sk-...."
                          (then close and reopen the terminal)
  Mac/Linux:              export OPENAI_API_KEY="sk-...."

Start it, then just open assignment_recorder.html in your browser as usual.
Leave this terminal window running the whole time students are recording.
Close it (Ctrl+C) when you're done for the day.
"""

import io
import os
import re
import csv
import datetime
import tempfile
from difflib import SequenceMatcher

from flask import Flask, request, jsonify

from openai import OpenAI

# ---- same expected text + normalization used in score_students.py ----
expected = {
    1:  "וירא אליו הויה באלוני ממרא והוא ישב פתח האהל כחום היום",
    2:  "וישא עיניו וירא והנה שלשה אנשים נצבים עליו וירא וירץ לקראתם מפתח האהל וישתחו ארצה",
    3:  "ויאמר אדני אם נא מצאתי חן בעיניך אל נא תעבר מעל עבדך",
    4:  "יקח נא מעט מים ורחצו רגליכם והשענו תחת העץ",
    5:  "ואקחה פת לחם וסעדו לבכם אחר תעברו כי על כן עברתם על עבדכם ויאמרו כן תעשה כאשר דברת",
    6:  "וימהר אברהם האהלה אל שרה ויאמר מהרי שלש סאים קמח סלת לושי ועשי עגות",
    7:  "ואל הבקר רץ אברהם ויקח בן בקר רך וטוב ויתן אל הנער וימהר לעשות אתו",
    8:  "ויקח חמאה וחלב ובן הבקר אשר עשה ויתן לפניהם והוא עמד עליהם תחת העץ ויאכלו",
    9:  "ויאמרו אליו איה שרה אשתך ויאמר הנה באהל",
    10: "ויאמר שוב אשוב אליך כעת חיה והנה בן לשרה אשתך ושרה שמעת פתח האהל והוא אחריו",
    11: "ואברהם ושרה זקנים באים בימים חדל להיות לשרה ארח כנשים",
    12: "ותצחק שרה בקרבה לאמר אחרי בלתי היתה לי עדנה ואדני זקן",
    13: "ויאמר הויה אל אברהם למה זה צחקה שרה לאמר האף אמנם אלד ואני זקנתי",
    14: "היפלא מהויה דבר למועד אשוב אליך כעת חיה ולשרה בן",
    15: "ותכחש שרה לאמר לא צחקתי כי יראה ויאמר לא כי צחקת",
    16: "ויקמו משם האנשים וישקפו על פני סדם ואברהם הלך עמם לשלחם",
    17: "והויה אמר המכסה אני מאברהם אשר אני עשה",
    18: "ואברהם היו יהיה לגוי גדול ועצום ונברכו בו כל גויי הארץ",
    19: "כי ידעתיו למען אשר יצוה את בניו ואת ביתו אחריו ושמרו דרך הויה לעשות צדקה ומשפט למען הביא הויה על אברהם את אשר דבר עליו",
    20: "ויאמר הויה זעקת סדם ועמרה כי רבה וחטאתם כי כבדה מאד",
    21: "ארדה נא ואראה הכצעקתה הבאה אלי עשו כלה ואם לא אדעה",
    22: "ויפנו משם האנשים וילכו סדמה ואברהם עודנו עמד לפני הויה",
    23: "ויגש אברהם ויאמר האף תספה צדיק עם רשע",
    24: "אולי יש חמשים צדיקם בתוך העיר האף תספה ולא תשא למקום למען חמשים הצדיקם אשר בקרבה",
    25: "חללה לך מעשת כדבר הזה להמית צדיק עם רשע והיה כצדיק כרשע חללה לך השפט כל הארץ לא יעשה משפט",
    26: "ויאמר הויה אם אמצא בסדם חמשים צדיקם בתוך העיר ונשאתי לכל המקום בעבורם",
    27: "ויען אברהם ויאמר הנה נא הואלתי לדבר אל אדני ואנכי עפר ואפר",
    28: "אולי יחסרון חמשים הצדיקם חמשה התשחית בחמשה את כל העיר ויאמר לא אשחית אם אמצא שם ארבעים וחמשה",
    29: "ויסף עוד לדבר אליו ויאמר אולי ימצאון שם ארבעים ויאמר לא אעשה בעבור הארבעים",
    30: "ויאמר אל נא יחר לאדני ואדברה אולי ימצאון שם שלשים ויאמר לא אעשה אם אמצא שם שלשים",
    31: "ויאמר הנה נא הואלתי לדבר אל אדני אולי ימצאון שם עשרים ויאמר לא אשחית בעבור העשרים",
    32: "ויאמר אל נא יחר לאדני ואדברה אך הפעם אולי ימצאון שם עשרה ויאמר לא אשחית בעבור העשרה",
    33: "וילך הויה כאשר כלה לדבר אל אברהם ואברהם שב למקמו",
}

def _consonant_skeleton(w):
    """Shared first stage of word normalization: merges sound-alike
    consonants (vav/bet/pey, chet/chaf, tav/samech) and strips niqqud and
    punctuation, but - unlike normalize_word below - always keeps the weak
    letters (alef/hey/ayin/yud) in place. Used directly by the fusion
    fallback in score_words_detailed/score_word: when the transcriber
    blends two adjacent words into one longer token (e.g. "יקח נא" heard
    back as one word like "וייקחנא"), the short word's letters often
    still appear intact as a contiguous chunk inside that longer token,
    but only if weak letters haven't already been stripped out of either
    side - so containment checks need this less-aggressive form rather
    than normalize_word's fully-stripped one."""
    w = w.replace("הויה", "השם")
    w = w.replace("אדני", "השם")
    w = w.replace("ה\u05F3", "השם")
    w = w.replace("ה'", "השם")
    w = re.sub(r'[^א-ת]', '', w)
    finals = "םןץףך"
    regulars = "מנצפכ"
    w = w.translate(str.maketrans(finals, regulars))
    w = re.sub(r'[תס]', 'S', w)
    w = re.sub(r'[בפו]', 'P', w)
    w = re.sub(r'[חכ]', 'K', w)
    return w

def skeleton_fuzzy_contains(target_skel, heard_skel, threshold=0.75):
    """Checks whether target_skel's letters are present, in order, inside
    heard_skel - close enough to count even if a letter got dropped right
    at the seam where two words fused together.

    A strict substring check (target_skel in heard_skel) catches most
    fusions, e.g. "יקח נא" heard as one word "וייקחנא" still contains "נא"
    intact at the end. But sometimes the fusion also swallows a letter at
    the boundary - e.g. "פתח האהל" (two words, each starting/ending near an
    ה/ח sound) heard as one blended word "פסחהאל", which drops one of the
    repeated ה's where the words run together, so "האהל" no longer appears
    as an exact substring even though it was clearly said. This slides a
    window of about the same length as target_skel across heard_skel and
    accepts it if any window is a close enough fuzzy match, the same way
    single-word comparisons already tolerate a minor transcription slip."""
    if len(target_skel) < 2 or not heard_skel:
        return False
    if target_skel in heard_skel:
        return True
    win_len = len(target_skel)
    best_ratio = 0.0
    for wlen in range(max(1, win_len - 1), win_len + 2):
        for start in range(0, max(1, len(heard_skel) - wlen + 1)):
            window = heard_skel[start:start + wlen]
            if window:
                best_ratio = max(best_ratio, SequenceMatcher(None, target_skel, window).ratio())
    return best_ratio >= threshold

def normalize_word(w):
    """Normalize a single word for comparison (handles common pronunciation/spelling variation).

    Words made mostly/entirely of the "weak" letters (alef/hey/ayin/vav/yud)
    - e.g. vayehi, vehaya, haya, Y-H-V-H itself - used to collapse to an empty
    string once those letters were stripped out below, since they can blend
    or go silent in speech. An empty normalized word was then dropped from
    scoring entirely instead of counting as a miss (or a match), so words
    like "vayehi" silently vanished from the results instead of being graded.
    Fix: only strip the weak letters if something non-empty survives; if
    stripping would erase the whole word, keep the untouched consonant
    skeleton instead so the word still participates in scoring."""
    skeleton = _consonant_skeleton(w)  # consonants only (with sound-alike letters merged), before weak-letter stripping below
    stripped = re.sub(r'[אהע]', '', skeleton)
    stripped = re.sub(r'[י]', '', stripped)
    # A word that's almost entirely weak letters - like "וירא" (ו,י,ר,א),
    # where only the ר survives - used to be returned as-is once non-empty.
    # But a single leftover consonant is too thin a signature: (1) it can
    # collide with a completely different word that also reduces to that
    # same one letter (e.g. in Perek 18 posuk 1, "אליו" and "האהל" both
    # collapse to "ל"), which can throw off the word-by-word alignment for
    # neighboring words too; and (2) it forces an all-or-nothing exact-match
    # at the token level instead of the fuzzy ratio comparison used for
    # longer words, so it never gets the partial credit that absorbs a minor
    # transcription slip. Keep the fuller, unstripped skeleton instead
    # whenever stripping would leave fewer than 2 consonants, so words like
    # "וירא" stay distinctive enough to align and fuzzy-match reliably.
    return stripped if len(stripped) >= 2 else skeleton

# Words that are so short / phonetically weak that they routinely blend
# into the word right before or after them in fast connected reading - so
# much so that Whisper often doesn't (or can't) transcribe them as their
# own separate token at all. When that happens, the word simply isn't in
# the heard list at all (a 'delete' in the alignment below), or it gets
# absorbed into a neighboring token as part of a 'replace'. Either way,
# flagging it as "mispronounced" punishes normal fluent reading, not an
# actual mistake - so words in this set get a free pass whenever they show
# up as a miss. "את" (es) is the main offender; add others here later if
# the same false-flag shows up for them (e.g. "כי", "גם", "עם").
ELIDABLE_WORDS = {"את"}
ELIDABLE_NORMS = {normalize_word(w) for w in ELIDABLE_WORDS}

def score_words_detailed(expected_text, heard_text, word_threshold=0.75):
    """Word-by-word comparison. Returns (score_percent, [(word, is_match), ...])
    so wrong/missing words actually count as misses instead of being diluted
    across one long normalized string."""
    # A maqaf (the Hebrew hyphen "־", U+05BE) joins two words under one
    # accent in the printed text - e.g. "מֵעַל־עַבְדֶּךָ" - and Whisper's
    # transcription sometimes reproduces that joined form even when the
    # expected text (as typed in the `expected` dict / sent from student.html)
    # has the same two words separated by a plain space. A maqaf isn't
    # whitespace, so plain .split() left the heard side with one combined
    # token where the expected side had two, which misaligned the word-by-
    # word comparison and made both real words show up as wrong even though
    # they were read correctly. Replacing the maqaf with a space before
    # splitting (on both sides, so either could contain one) fixes the
    # token count to match.
    MAQAF = "\u05be"
    exp_raw = expected_text.replace(MAQAF, " ").split()
    heard_raw = heard_text.replace(MAQAF, " ").split()
    exp_norm = [normalize_word(w) for w in exp_raw]
    heard_norm = [normalize_word(w) for w in heard_raw]

    exp_pairs = [(r, n) for r, n in zip(exp_raw, exp_norm) if n]
    heard_pairs = [(r, n) for r, n in zip(heard_raw, heard_norm) if n]

    exp_norm_list = [n for r, n in exp_pairs]
    heard_norm_list = [n for r, n in heard_pairs]

    matcher = SequenceMatcher(None, exp_norm_list, heard_norm_list)
    matched = 0
    total = len(exp_pairs)
    word_results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i1, i2):
                word_results.append((exp_pairs[k][0], True))
            matched += (i2 - i1)
        elif tag == 'replace':
            span_e = exp_pairs[i1:i2]
            span_h = heard_pairs[j1:j2]
            # Pre-compute the less-aggressive skeleton form (weak letters
            # kept) for every heard token in this mismatched group, for the
            # fusion fallback below.
            span_h_skeletons = [_consonant_skeleton(h_raw) for h_raw, _ in span_h]
            for k in range(len(span_e)):
                e_raw, e_norm = span_e[k]
                # Elidable words (like "את") routinely vanish from or blend
                # into the transcription in fast connected reading - give
                # them a free pass here rather than requiring the normal
                # fuzzy-match/fusion checks below to happen to catch them.
                if e_norm in ELIDABLE_NORMS:
                    word_results.append((e_raw, True))
                    matched += 1
                    continue
                is_match = False
                if k < len(span_h):
                    h_raw, h_norm = span_h[k]
                    ratio = SequenceMatcher(None, e_norm, h_norm).ratio()
                    is_match = ratio >= word_threshold
                if not is_match:
                    # The transcriber sometimes blends two adjacent short
                    # words into one longer token (e.g. "יקח נא" heard back
                    # as a single word like "וייקחנא"), which strands one
                    # of the two expected words with no positionally-aligned
                    # heard counterpart even though it really was said. If
                    # this expected word's letters survive intact as a
                    # contiguous chunk inside ANY heard token in this same
                    # group (not just the one at the same position), credit
                    # it as matched rather than marking it wrong.
                    e_skel = _consonant_skeleton(e_raw)
                    if any(skeleton_fuzzy_contains(e_skel, hs) for hs in span_h_skeletons):
                        is_match = True
                if is_match:
                    matched += 1
                word_results.append((e_raw, is_match))
        elif tag == 'delete':
            # An expected word with no heard counterpart at all - most
            # often because it was genuinely skipped/misread, but also the
            # shape a truly elidable word (like "את") takes when Whisper
            # drops it from the transcript entirely instead of blending it
            # into a neighboring token. Give elidable words a free pass here
            # too, same as in the 'replace' branch above.
            for k in range(i1, i2):
                e_raw, e_norm = exp_pairs[k]
                if e_norm in ELIDABLE_NORMS:
                    word_results.append((e_raw, True))
                    matched += 1
                else:
                    word_results.append((e_raw, False))
        # 'insert' = extra word said that wasn't expected -> not counted against/for

    score = round((matched / total) * 100) if total else 0
    return score, word_results

def score_text(expected_text, actual_text):
    if not expected_text:
        return None, []
    return score_words_detailed(expected_text, actual_text)

LOG_PATH = "scores_log.csv"

def log_result(name, perek_n, posuk_n, score, transcription):
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "student", "perek", "posuk", "score", "transcription"])
        writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                          name, perek_n, posuk_n, score, transcription])

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit(
        "OPENAI_API_KEY environment variable is not set.\n"
        "Set it first, then re-run this script. See the instructions at the "
        "top of score_server.py."
    )

client = OpenAI(api_key=api_key)
print("Connected to OpenAI transcription API. Server starting on http://localhost:5005")
print("Leave this window open. Open assignment_recorder.html in your browser now.")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    # Render (and other hosts) ping this to confirm the service is alive.
    return "OK", 200

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/score", methods=["OPTIONS"])
def score_options():
    return ("", 204)

@app.route("/score", methods=["POST"])
def score():
    name = request.form.get("name", "student")
    perek_n = request.form.get("perek", "")
    posuk_n = int(request.form.get("posuk", "0"))
    # The student.html app sends the exact text of whatever posuk is on
    # screen. That's the real source of truth — posuk numbers repeat across
    # perakim (e.g. every perek has a "posuk 1"), so falling back to the old
    # hardcoded table (which only ever held Perek 18's text) by posuk number
    # alone caused wrong-perek text to be used for scoring. Only fall back
    # to that table if a client ever fails to send posukText.
    expected_text = request.form.get("posukText", "").strip() or expected.get(posuk_n, "")
    audio_file = request.files.get("audio")

    if audio_file is None:
        return jsonify({"error": "No audio received"}), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # A blank/near-silent recording (mic didn't capture anything, or the
        # browser sent a truncated file) isn't worth sending to the API at
        # all - catch it here and return a friendly error immediately.
        if os.path.getsize(tmp_path) < 1000:
            return jsonify({
                "error": "Recording appears empty or too short. Please record again."
            }), 400

        try:
            prompt_hint = expected_text
            with open(tmp_path, "rb") as audio_fp:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_fp,
                    language="he",
                    prompt=prompt_hint,
                )
            transcription = result.text.strip()
        except Exception as e:
            # Covers API errors, network errors, or any transcription
            # failure - student sees a friendly message, you see the real
            # error in this terminal window.
            print(f"Transcription error: {e}")
            return jsonify({
        	   "Could not process this recording. Please try recording again."
            }), 400
    finally:
        os.remove(tmp_path)

    posuk_score, word_results = score_text(expected_text, transcription)
    if posuk_score is None:
        return jsonify({"error": f"No expected text on file for posuk {posuk_n}"}), 400

    log_result(name, perek_n, posuk_n, posuk_score, transcription)

    return jsonify({
        "score": posuk_score,
        "transcription": transcription,
        "words": [{"word": w, "match": ok} for w, ok in word_results],
    })

@app.route("/score_word", methods=["OPTIONS"])
def score_word_options():
    return ("", 204)

@app.route("/score_word", methods=["POST"])
def score_word():
    """Checks a short recording against a single expected word, for the
    per-word "redo just this word" feature in student.html. A student who
    got one or two words flagged wrong in a full-posuk reading can
    re-record just that word instead of the whole posuk; this endpoint
    transcribes the short clip and reports whether it now matches, without
    touching the original full recording/transcript/score at all - the
    frontend folds the result back into the existing word-by-word list and
    recomputes the overall percentage itself.
    """
    expected_word = request.form.get("word", "").strip()
    # A bare 1-2 letter word gives Whisper very little to work with in
    # isolation, unlike full-posuk scoring where a whole sentence of
    # context helps it land on the right transcription. student.html sends
    # a small window of the surrounding words (still just re-grading this
    # one target word, not changing what's checked) so the prompt hint has
    # some real phrase structure to anchor on instead of one bare word.
    context = request.form.get("context", "").strip()
    prompt_hint = context or expected_word
    audio_file = request.files.get("audio")

    if not expected_word:
        return jsonify({"error": "No expected word given"}), 400
    if audio_file is None:
        return jsonify({"error": "No audio received"}), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if os.path.getsize(tmp_path) < 1000:
            return jsonify({
                "error": "Recording appears empty or too short. Please record again."
            }), 400

        try:
            with open(tmp_path, "rb") as audio_fp:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_fp,
                    language="he",
                    prompt=prompt_hint,
                )
            transcription = result.text.strip()
        except Exception as e:
            print(f"Transcription error (word fix): {e}")
            return jsonify({
                "error": "Could not process this recording. Please try recording again."
            }), 400
    finally:
        os.remove(tmp_path)

    # A one-or-two-word clip often comes back from Whisper with a bit of
    # extra filler (a repeated syllable, a stray "um", etc.), so rather than
    # requiring the whole transcription to equal the target word, check
    # whether ANY word in what was heard is a close enough match to it -
    # same fuzzy-ratio approach and threshold used for full-posuk scoring.
    #
    # An elidable word (like "את") gets a free pass here too, same as in
    # score_words_detailed above: if a student is redoing just this one
    # word and it's one that routinely doesn't survive transcription on its
    # own, don't make them fight the transcriber to clear it.
    if normalize_word(expected_word) in ELIDABLE_NORMS:
        return jsonify({
            "match": True,
            "transcription": "",
        })

    MAQAF = "\u05be"
    heard_words = transcription.replace(MAQAF, " ").split()
    target_norm = normalize_word(expected_word)
    target_skel = _consonant_skeleton(expected_word)
    best_ratio = 0.0
    fused_match = False
    for hw in heard_words:
        hw_norm = normalize_word(hw)
        if hw_norm:
            ratio = SequenceMatcher(None, target_norm, hw_norm).ratio()
            best_ratio = max(best_ratio, ratio)
        # Fusion fallback: the context we send (the target word plus a
        # couple of neighbors) can end up blended by the transcriber into
        # one longer token, especially for a short word like this one -
        # e.g. "יקח נא" heard back as a single word like "וייקחנא". If the
        # target word's letters survive intact as a contiguous chunk
        # inside a longer heard token, count that as a match too.
        hw_skel = _consonant_skeleton(hw)
        if skeleton_fuzzy_contains(target_skel, hw_skel):
            fused_match = True

    match = best_ratio >= 0.75 or fused_match

    return jsonify({
        "match": match,
        "transcription": transcription,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
