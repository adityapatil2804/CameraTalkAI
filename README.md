# CameraTalkAI

CameraTalkAI is a product-focused demo that lets users ask spoken questions about what their camera sees. The app captures a single snapshot per question (no continuous video processing), sends it alongside the transcript to a vision-language model, and reads the response aloud.

## Features (V1)
- Live camera preview with WebRTC `getUserMedia`.
- Push-to-talk voice input with the Web Speech API.
- One snapshot captured per question (no storage).
- Visual Question Answering flow (image + question → answer).
- Text-to-speech playback of the AI response.
- Disposable session: no data stored or retained.
- Image quality checks (blur + brightness) to reject bad frames.

## Tech Stack
- **Frontend:** React (CDN), HTML, CSS, JavaScript
- **Backend:** FastAPI
- **ML:** Vision-language model API (to be wired in)

## Local Setup

### 1) Backend API
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2) Frontend
```bash
cd frontend
python -m http.server 5173
```

Open `http://localhost:5173` in your browser and allow camera + microphone permissions.

## API Endpoints

### POST `/api/answer`
Payload:
```json
{
  "question": "What is on the table?",
  "image": "data:image/jpeg;base64,..."
}
```
Response:
```json
{
  "answer": "...",
  "confidence": 0.42
}
```

### POST `/ask`
Alias of `/api/answer` to match the VQA endpoint spec.

### POST `/quality`
Payload:
```json
{
  "question": "ignored",
  "image": "data:image/jpeg;base64,..."
}
```
Response:
```json
{
  "brightness": 123.4,
  "blur_variance": 85.1,
  "too_dark": false,
  "too_bright": false,
  "blurry": false
}
```

## How It Works
1. Turn on the camera.
2. Hold the microphone button and ask a question.
3. The app captures a single image frame and sends it with the transcript to `/api/answer`.
4. The backend runs a quality check, then returns an answer and the browser speaks it aloud.

## Integrating a Vision-Language Model
Replace the placeholder response in `backend/main.py` with a call to your preferred model API. The payload includes:
- `question`: user transcript
- `image`: base64-encoded JPEG data URL

## Notes
- The Web Speech API is best supported in Chromium-based browsers.
- The backend currently returns a stub response until an ML provider is connected.

## Roadmap
- OCR mode toggle
- Detection + highlight bounding boxes
- Before/after frame comparisons
- Safety response when the object is not in view
