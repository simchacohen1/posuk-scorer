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

# All word normalization / matching lives in scoring.py so it can be
# tested offline (python -m unittest test_scoring.py).
from scoring import score_text, score_single_word

# What to give Whisper as its "prompt" hint.
#   posuk (default) - send the expected posuk text. Whisper copies its
#                     spelling, which can hide a child's misreading, BUT
#                     without it whisper-1 turns correct Ashkenazi readings
#                     into nonsense: re-transcribing 71 teacher-reviewed
#                     recordings without the hint dropped agreement with the
#                     teacher from 33/71 to 13/71. So the hint stays.
#   none            - no hint (for future experiments, e.g. a better model).
WHISPER_PROMPT_MODE = os.environ.get("WHISPER_PROMPT_MODE", "posuk").strip().lower()

def whisper_prompt(expected_text):
    return expected_text if WHISPER_PROMPT_MODE == "posuk" else None

# Which OpenAI transcription model to use (Render env TRANSCRIBE_MODEL).
# Tested 2026-11-06 on 71 teacher-reviewed recordings, with the posuk hint:
#   whisper-1 (default) - agreed with the teacher on 32/71; never copied a
#                         deliberately wrong hint (0/71).
#   gpt-transcribe      - agreed on 59/71 and caught all 3 readings the
#                         teacher marked wrong, but copied a wrong hint 7/71.
#   gpt-4o-transcribe   - agreed on 59/71 but copied a wrong hint 15/71 and
#                         missed 2 of the 3 wrong readings - not recommended.
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "whisper-1").strip()

def transcribe(audio_fp, prompt):
    kwargs = {"model": TRANSCRIBE_MODEL, "file": audio_fp, "language": "he"}
    if TRANSCRIBE_MODEL == "whisper-1":
        kwargs["temperature"] = 0
    if prompt:
        kwargs["prompt"] = prompt
    return client.audio.transcriptions.create(**kwargs).text.strip()

# OpenAI rejects a recording whose file name doesn't match its real format
# ("Invalid file format"). Every upload used to be saved as .webm, so
# Safari / iPhone / iPad recordings (audio/mp4) could never be scored.
# Keep the type the browser sent, from the upload's file name or MIME type.
_AUDIO_EXTS = {"webm", "mp4", "m4a", "aac", "ogg", "oga", "mp3", "mpeg", "mpga", "wav", "flac"}
_MIME_EXTS = {"mp4": "mp4", "aac": "m4a", "ogg": "ogg", "mpeg": "mp3", "wav": "wav", "flac": "flac", "webm": "webm"}

def audio_suffix(audio_file):
    name = (audio_file.filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext == "aac":
        ext = "m4a"  # OpenAI doesn't accept .aac; Safari's AAC is in an mp4/m4a container
    if ext in _AUDIO_EXTS:
        return "." + ext
    mime = (audio_file.mimetype or "").lower()
    for key, mapped in _MIME_EXTS.items():
        if key in mime:
            return "." + mapped
    return ".webm"

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
    # General reading passages may have no numbered posuk; accept blank as 0.
    raw_posuk = (request.form.get("posuk") or "0").strip()
    try:
        posuk_n = int(raw_posuk)
    except ValueError:
        return jsonify({"error": "Invalid posuk number"}), 400
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

    with tempfile.NamedTemporaryFile(suffix=audio_suffix(audio_file), delete=False) as tmp:
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
            with open(tmp_path, "rb") as audio_fp:
                transcription = transcribe(audio_fp, whisper_prompt(expected_text))
        except Exception as e:
            # Covers API errors, network errors, or any transcription
            # failure - student sees a friendly message, you see the real
            # error in this terminal window.
            print(f"Transcription error: {e}")
            return jsonify({
                "error": "Could not process this recording. Please try recording again."
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
    # student.html also sends a small window of the surrounding words. It is
    # the Whisper hint (see WHISPER_PROMPT_MODE) and is also used to
    # recognise the target fused with a neighbour (e.g. "יקח נא" written as
    # one token). Only the target word itself can make this pass.
    context = request.form.get("context", "").strip()
    audio_file = request.files.get("audio")

    if not expected_word:
        return jsonify({"error": "No expected word given"}), 400
    if audio_file is None:
        return jsonify({"error": "No audio received"}), 400

    with tempfile.NamedTemporaryFile(suffix=audio_suffix(audio_file), delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if os.path.getsize(tmp_path) < 1000:
            return jsonify({
                "error": "Recording appears empty or too short. Please record again."
            }), 400

        try:
            with open(tmp_path, "rb") as audio_fp:
                transcription = transcribe(audio_fp, whisper_prompt(context or expected_word))
        except Exception as e:
            print(f"Transcription error (word fix): {e}")
            return jsonify({
                "error": "Could not process this recording. Please try recording again."
            }), 400
    finally:
        os.remove(tmp_path)

    match = score_single_word(expected_word, transcription, context)

    return jsonify({
        "match": match,
        "transcription": transcription,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
