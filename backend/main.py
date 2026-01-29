import base64
import os
import re
from typing import Optional, List

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional OCR deps
from PIL import Image
import pytesseract


# -----------------------------
# Config
# -----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava:7b")

# Example:
# setx TESSERACT_PATH "C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


app = FastAPI(title="CameraTalkAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Schemas
# -----------------------------
class QuestionPayload(BaseModel):
    question: str
    image: str  # data URL or raw base64


class Box(BaseModel):
    label: str
    x: int
    y: int
    w: int
    h: int
    score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    boxes: Optional[List[Box]] = None


class QualityResponse(BaseModel):
    brightness: float
    blur_variance: float
    too_dark: bool
    too_bright: bool
    blurry: bool


# -----------------------------
# Helpers
# -----------------------------
def strip_data_url(data_url: str) -> str:
    return data_url.split(",", 1)[1] if "," in data_url else data_url


def decode_image(data_url: str) -> np.ndarray:
    encoded = strip_data_url(data_url)
    try:
        image_bytes = base64.b64decode(encoded)
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")
    return image


def quality_metrics(image: np.ndarray) -> QualityResponse:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    too_dark = brightness < 50
    too_bright = brightness > 205
    blurry = blur_variance < 80

    return QualityResponse(
        brightness=brightness,
        blur_variance=blur_variance,
        too_dark=too_dark,
        too_bright=too_bright,
        blurry=blurry,
    )


def quality_block_message(m: QualityResponse) -> Optional[str]:
    reasons = []
    if m.too_dark:
        reasons.append("too dark")
    if m.too_bright:
        reasons.append("too bright")
    if m.blurry:
        reasons.append("too blurry")

    if not reasons:
        return None

    # Example: "Image unclear: too dark and too blurry. Adjust lighting or hold steady."
    if len(reasons) == 1:
        detail = reasons[0]
    elif len(reasons) == 2:
        detail = f"{reasons[0]} and {reasons[1]}"
    else:
        detail = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    return f"Image unclear: {detail}. Adjust lighting or hold the camera steady."


def np_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def ocr_extract_text(image_bgr: np.ndarray) -> str:
    pil_img = np_to_pil(image_bgr)
    text = pytesseract.image_to_string(pil_img)
    text = " ".join(text.split())
    return text.strip()


def one_line(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    text = " ".join(text.split())
    return text


def trim_to_complete(text: str, max_words: int = 28) -> str:
    t = one_line(text)

    dangling = {"wearing", "with", "and", "or", "but", "that", "which", "to", "of", "in", "on"}
    words = t.split()
    if words and words[-1].lower().strip(".,!?") in dangling:
        t = " ".join(words[:-1])

    # Cut at first complete sentence if present
    m = re.search(r"^(.+?[.!?])\s", t)
    if m:
        t = m.group(1)

    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(",")
        t += "."

    return t.strip() or "I can't see it in the camera frame."


def is_yesno_question(q: str) -> bool:
    q = q.strip().lower()

    starters = (
        "is ", "are ", "am ", "was ", "were ",
        "do ", "does ", "did ",
        "can ", "could ", "should ", "would ", "will ",
        "has ", "have ", "had "
    )
    if q.startswith(starters):
        return True
    if q.startswith("is there") or q.startswith("are there"):
        return True

    # Common yes/no patterns
    if q.endswith("?") and (q.startswith("am i ") or q.startswith("are we ") or q.startswith("is it ")):
        return True

    return False


def enforce_yesno(answer: str) -> str:
    t = one_line(answer).lower()

    # Strict: only allow Yes. or No.
    head = t[:16]
    if "yes" in head:
        return "Yes."
    if "no" in head:
        return "No."

    if t.startswith("y"):
        return "Yes."
    if t.startswith("n"):
        return "No."

    return "I can't tell from the frame."


def should_trigger_ocr(question: str) -> bool:
    q = question.lower()

    keywords = [
        "read", "text", "words", "written", "write",
        "what does it say", "what does this say", "what is written",
        "label", "caption", "sign", "poster", "board",
        "serial", "model number", "barcode", "qr",
        "ocr", "spell", "letter", "letters", "number", "numbers", "digit", "digits",
    ]
    return any(k in q for k in keywords)


def call_ollama_vqa(question: str, image_data_url_or_b64: str, ocr_text: Optional[str] = None) -> str:
    b64 = strip_data_url(image_data_url_or_b64)
    want_yesno = is_yesno_question(question)

    system_lines = [
        "You are CameraTalkAI.",
        "Answer using only what is visible in the image.",
        "If the user asks about something not visible, reply exactly: I can't see it in the camera frame.",
        "Keep the answer short, one line.",
        "No extra explanation.",
    ]
    if want_yesno:
        system_lines.append("This is a yes/no question. Reply with only: Yes. or No. (no extra words).")

    user_content = question.strip()
    if ocr_text:
        # OCR is extra context, not returned directly
        user_content += f"\n\nOCR text (may be noisy): {ocr_text}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "\n".join(system_lines)},
            {"role": "user", "content": user_content, "images": [b64]},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 80,
        },
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=90)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama not reachable: {exc}") from exc

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")

    data = r.json()
    content = data.get("message", {}).get("content", "")
    content = one_line(content)

    if want_yesno:
        return enforce_yesno(content)

    return trim_to_complete(content)


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "CameraTalkAI API", "endpoint": "/api/answer"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/answer", response_model=AskResponse)
async def answer(payload: QuestionPayload):
    question = payload.question.strip()
    if not question:
        return AskResponse(answer="Ask a question about what the camera sees.", confidence=0.1)

    image = decode_image(payload.image)
    metrics = quality_metrics(image)

    block_msg = quality_block_message(metrics)
    if block_msg:
        return AskResponse(answer=block_msg, confidence=0.2, boxes=None)

    ocr_text = None
    if should_trigger_ocr(question):
        try:
            # If Tesseract is not installed or misconfigured, skip silently
            ocr_text = ocr_extract_text(image)
            if not ocr_text:
                ocr_text = None
        except Exception:
            ocr_text = None

    ans = call_ollama_vqa(question, payload.image, ocr_text=ocr_text)
    return AskResponse(answer=ans, confidence=0.75, boxes=None)


@app.post("/ask", response_model=AskResponse)
async def ask(payload: QuestionPayload):
    return await answer(payload)


@app.post("/quality", response_model=QualityResponse)
async def quality_check(payload: QuestionPayload):
    image = decode_image(payload.image)
    return quality_metrics(image)
