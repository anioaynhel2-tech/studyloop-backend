import os
import json
import re
import time
import traceback

from flask import Flask, request, jsonify
import fitz  # PyMuPDF - much lighter on memory than pypdf for text extraction
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
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
MODEL_NAME = "gemini-3.5-flash-lite"  # current fast/cheap model, free-tier friendly

app = Flask(__name__)


PROMPT_TEMPLATE = """You are helping a student turn their lecture notes into study material.

Below is the raw text extracted from one lecture. Read it carefully, then produce:

1. A reviewer broken into short sections - one per concept/topic covered in
   the lecture, not one long block of text. Each section needs a "topic"
   label (reuse the EXACT SAME short topic label you assign to the quiz
   questions below wherever a section and a question cover the same
   concept, so the reviewer and the quiz line up) and "content" with a few
   clear paragraphs covering that concept in plain text. Put the sections
   in a sensible teaching order - the order the concepts are introduced in
   the lecture, not alphabetical.
2. A THOROUGH bank of quiz questions covering the material, made up of ALL
   FOUR of these types:
   - "mcq": multiple choice, exactly 4 choices, one correct
   - "identification": a short-answer question with one correct answer
   - "matching": a set of 4-6 term/definition pairs to match
   - "enumeration": "list N things" style, with all acceptable correct items

Before writing any questions, mentally list out every distinct concept,
definition, process, fact, or relationship covered anywhere in the lecture
text - including details mentioned only once or in passing, since those are
exactly the kind of thing that shows up on an exam and catches students off
guard. Then make sure your question bank actually tests EVERY item on that
list somewhere - don't stop early just because you've hit a "typical" number
of questions. A short, sparse lecture should get a short bank; a long or
dense one should get a correspondingly larger bank. There is no upper limit -
prioritize covering everything over keeping the bank small. As a rough floor
only (not a target to stop at): at least 8 mcq, 6 identification, 2 matching
sets (4-6 pairs each), and 3 enumeration questions for even a short lecture,
scaling up well beyond that for anything longer or denser.

For EVERY question, assign a short "topic" label (2-5 words) naming the
specific concept it tests, drawn from the lecture itself (e.g. "Krebs cycle",
"Newton's second law"). Reuse the exact same topic label across questions that
test the same concept, so a student's performance can later be grouped by
topic.

For every "mcq" question, also write a short "explanation" (1-2 sentences)
that says why the correct answer is right, phrased so it naturally makes
clear why the other choices are wrong too. Keep it brief and student-facing,
not a lecture.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching
exactly this shape:

{{
  "notes": [
    {{
      "topic": "string",
      "content": "string"
    }}
  ],
  "questions": [
    {{
      "type": "mcq",
      "topic": "string",
      "prompt": "string",
      "choices": ["string", "string", "string", "string"],
      "answer": "string (must exactly match one of choices)",
      "explanation": "string (1-2 sentences on why the answer is correct)"
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


RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["notes", "questions"],
    properties={
        "notes": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["topic", "content"],
                properties={
                    "topic": types.Schema(type=types.Type.STRING),
                    "content": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        "questions": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["type", "topic", "prompt"],
                properties={
                    "type": types.Schema(
                        type=types.Type.STRING,
                        enum=["mcq", "identification", "matching", "enumeration"],
                    ),
                    "topic": types.Schema(type=types.Type.STRING),
                    "prompt": types.Schema(type=types.Type.STRING),
                    "choices": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    ),
                    "answer": types.Schema(type=types.Type.STRING),
                    "explanation": types.Schema(type=types.Type.STRING),
                    "pairs": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "left": types.Schema(type=types.Type.STRING),
                                "right": types.Schema(type=types.Type.STRING),
                            },
                        ),
                    ),
                    "answers": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    ),
                },
            ),
        ),
    },
)
def extract_pdf_text(file_storage) -> str:
    file_bytes = file_storage.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        pages = [page.get_text() or "" for page in doc]
    finally:
        doc.close()
    return "\n".join(pages).strip()


def call_gemini(lecture_text: str) -> dict:
    # 400,000 characters is roughly 100,000 tokens - comfortably within
    # Gemini's context window even for a long, dense lecture, and a big jump
    # up from the old 60,000-character cap that could cut off longer PDFs.
    prompt = PROMPT_TEMPLATE.format(lecture_text=lecture_text[:400000])

    # Gemini's own servers occasionally return 503 "high demand" errors that
    # have nothing to do with our code - Google's SDK already retries a
    # couple of times internally, but not always enough during a busy spell.
    # Retry a few more times on our end, with a short growing pause, before
    # giving up. Keep the total wait well under gunicorn's 150s timeout.
    max_attempts = 3
    wait_seconds = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                # A thorough, longer question bank means a longer response -
                # raise the output limit so a big bank doesn't get cut off
                # mid-JSON.
                config=types.GenerateContentConfig(
                    max_output_tokens=32768,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
            break
        except genai_errors.ServerError:
            if attempt == max_attempts:
                raise
            print(
                f"Gemini is overloaded (attempt {attempt}/{max_attempts}), "
                f"retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2

    raw = (response.text or "").strip()

    # Gemini sometimes wraps JSON in ```json ... ``` even when told not to -
    # pull out the {...} block rather than trusting the whole reply is clean.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    json_str = match.group(0) if match else raw

    return json.loads(json_str)


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
        traceback.print_exc()  # print the real error to Render's logs
        return jsonify({"error": "The AI response wasn't valid JSON. Try again."}), 502
    except genai_errors.ServerError:
        traceback.print_exc()  # print the real error to Render's logs
        return jsonify({
            "error": "The AI service is experiencing high demand right now. "
                     "Please wait a minute and try uploading again."
        }), 503
    except Exception as e:  # noqa: BLE001 - surface any generation failure to the client
        traceback.print_exc()  # print the real error to Render's logs
        return jsonify({"error": f"Generation failed: {e}"}), 502

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
