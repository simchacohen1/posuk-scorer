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

def normalize_word(w):
    """Normalize a single word for comparison (handles common pronunciation/spelling variation)."""
    w = w.replace("הויה", "השם")
    w = w.replace("אדני", "השם")
    w = re.sub(r'[^א-ת]', '', w)
    finals = "םןץףך"
    regulars = "מנצפכ"
    w = w.translate(str.maketrans(finals, regulars))
    w = re.sub(r'[תס]', 'S', w)
    w = re.sub(r'[בפ]', 'P', w)
    w = re.sub(r'[חכ]', 'K', w)
    w = re.sub(r'[אהע]', '', w)
    w = re.sub(r'[וי]', '', w)
    return w

def score_words_detailed(expected_text, heard_text, word_threshold=0.75):
    """Word-by-word comparison. Returns (score_percent, [(word, is_match), ...])
    so wrong/missing words actually count as misses instead of being diluted
    across one long normalized string."""
    exp_raw = expected_text.split()
    heard_raw = heard_text.split()
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
            for k in range(max(len(span_e), len(span_h))):
                if k < len(span_e):
                    e_raw, e_norm = span_e[k]
                    if k < len(span_h):
                        h_raw, h_norm = span_h[k]
                        ratio = SequenceMatcher(None, e_norm, h_norm).ratio()
                        is_match = ratio >= word_threshold
                        if is_match:
                            matched += 1
                        word_results.append((e_raw, is_match))
                    else:
                        word_results.append((e_raw, False))
        elif tag == 'delete':
            for k in range(i1, i2):
                word_results.append((exp_pairs[k][0], False))
        # 'insert' = extra word said that wasn't expected -> not counted against/for

    score = round((matched / total) * 100) if total else 0
    return score, word_results

def score_text(posuk_n, actual_text):
    if posuk_n not in expected:
        return None, []
    return score_words_detailed(expected[posuk_n], actual_text)

LOG_PATH = "scores_log.csv"

def log_result(name, posuk_n, score, transcription):
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "student", "posuk", "score", "transcription"])
        writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                          name, posuk_n, score, transcription])

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
    posuk_n = int(request.form.get("posuk", "0"))
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
            prompt_hint = expected.get(posuk_n, "")
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

    posuk_score, word_results = score_text(posuk_n, transcription)
    if posuk_score is None:
        return jsonify({"error": f"No expected text on file for posuk {posuk_n}"}), 400

    log_result(name, posuk_n, posuk_score, transcription)

    return jsonify({
        "score": posuk_score,
        "transcription": transcription,
        "words": [{"word": w, "match": ok} for w, ok in word_results],
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
