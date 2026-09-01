import os
import json
import re
import time

from flask import Flask, request, jsonify
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Copy .env.example to .env and paste your "
        "key in, or set the environment variable another way before starting "
        "the server."
    )

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash-lite"  # gemini-2.5-flash-lite was retired for new users - Google's own error message pointed us here, and it has fresh unused daily quota

app = Flask(__name__)


PROMPT_TEMPLATE = """You are helping a student turn their lecture notes into study material.

Below is the raw text extracted from one lecture. Read it carefully, then produce:

1. A concise reviewer (notes) covering the key concepts, written in clear plain
   text with short paragraphs and headings where useful.
2. A bank of quiz questions covering the material, made up of ALL FOUR of these
   types:
   - "mcq": multiple choice, exactly 4 choices, one correct
   - "identification": a short-answer question with one correct answer
   - "matching": a set of 4-6 term/definition pairs to match
   - "enumeration": "list N things" style, with all acceptable correct items

For EVERY question, assign a short "topic" label (2-5 words) naming the
specific concept it tests, drawn from the lecture itself (e.g. "Krebs cycle",
"Newton's second law"). Reuse the exact same topic label across questions that
test the same concept, so a student's performance can later be grouped by
topic.

Aim for roughly 8 mcq, 6 identification, 2 matching (each with 4-6 pairs), and
3 enumeration questions, adjusted to fit how much material is actually in the
text.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching
exactly this shape:

{{
  "notes": "string",
  "questions": [
    {{
      "type": "mcq",
      "topic": "string",
      "prompt": "string",
      "choices": ["string", "string", "string", "string"],
      "answer": "string (must exactly match one of choices)"
    }},
    {{
      "type": "identification",
      "topic": "string",
      "prompt": "string",
      "answer": "string"
    }},
    {{
      "type": "matching",
      "topic": "string",
      "prompt": "string (short instruction)",
      "pairs": [{{"left": "string", "right": "string"}}]
    }},
    {{
      "type": "enumeration",
      "topic": "string",
      "prompt": "string",
      "answers": ["string", "string"]
    }}
  ]
}}

LECTURE TEXT:
\"\"\"
{lecture_text}
\"\"\"
"""


def extract_pdf_text(file_storage) -> str:
    reader = PdfReader(file_storage)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


# How many times to retry a Gemini call that fails because the model is
# temporarily overloaded (503 UNAVAILABLE), and how long to wait between
# tries. Google's free tier is more prone to this than paid tiers, so
# retrying with a short backoff clears up most transient failures without
# the user having to manually tap Retry themselves.
GEMINI_MAX_ATTEMPTS = 4
GEMINI_RETRY_DELAY_SECONDS = [3, 6, 12]  # one entry per retry (not per attempt)


def call_gemini(lecture_text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(lecture_text=lecture_text[:60000])

    last_error: Exception | None = None
    for attempt in range(GEMINI_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            raw = (response.text or "").strip()

            # Gemini sometimes wraps JSON in ```json ... ``` even when told
            # not to - pull out the {...} block rather than trusting the
            # whole reply is clean.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = match.group(0) if match else raw

            return json.loads(json_str)
        except Exception as e:  # noqa: BLE001 - inspect and decide whether to retry
            last_error = e
            is_overloaded = "UNAVAILABLE" in str(e) or "503" in str(e)
            is_last_attempt = attempt == GEMINI_MAX_ATTEMPTS - 1
            if not is_overloaded or is_last_attempt:
                raise
            delay = GEMINI_RETRY_DELAY_SECONDS[attempt]
            print(
                f"[call_gemini] Gemini overloaded (attempt {attempt + 1}/"
                f"{GEMINI_MAX_ATTEMPTS}), retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)

    # Should be unreachable (the loop always returns or raises), but keeps
    # type checkers happy and guards against future edits to the loop above.
    raise last_error  # type: ignore[misc]


@app.get("/")
def health():
    return jsonify({"status": "StudyLoop backend is running"})


@app.post("/generate")
def generate():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send it as form field 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported right now."}), 400

    lecture_text = extract_pdf_text(uploaded)
    if not lecture_text:
        return jsonify({
            "error": "Could not extract any text from that PDF. Is it a scanned image?"
        }), 422

    try:
        result = call_gemini(lecture_text)
    except json.JSONDecodeError:
        return jsonify({"error": "The AI response wasn't valid JSON. Try again."}), 502
    except Exception as e:  # noqa: BLE001 - surface any generation failure to the client
        return jsonify({"error": f"Generation failed: {e}"}), 502

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
