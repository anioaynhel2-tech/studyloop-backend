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
MODEL_NAME = "gemini-3.7-flash"  # current fast/cheap model, free-tier friendly

app = Flask(__name__)


PROMPT_TEMPLATE = """You are helping a student turn their lecture notes into study material.

Below is the raw text extracted from one lecture. Read it carefully, then produce:

1. A concise reviewer (notes) covering the key concepts, written in clear plain
   text with short paragraphs and headings where useful.
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
list somewhere - don't
