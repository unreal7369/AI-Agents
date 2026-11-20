from pathlib import Path
import io
import uuid
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image

from .analyzer import analyze_eeg_image


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
FRONTEND = ROOT / "frontend"
UPLOADS = BASE / "uploads"

UPLOADS.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")


app = FastAPI(
    title="EEG Clinical Analysis Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def home():
    return FileResponse(
        FRONTEND / "index.html"
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "EEG Clinical Analysis Agent",
        "mode": "local",
        "llm_configured": bool(
            os.getenv("GEMINI_API_KEY")
        )
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    allowed = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "application/pdf",
    }

    if file.content_type not in allowed:
        raise HTTPException(
            400,
            (
                "Upload a PNG, JPG, JPEG, "
                "WEBP, or PDF EEG file."
            )
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            400,
            "The uploaded file is empty."
        )

    # Increase this if your EEG PDFs are larger.
    max_size = 30 * 1024 * 1024

    if len(data) > max_size:
        raise HTTPException(
            413,
            "File is larger than 30 MB."
        )

    is_pdf = (
        file.content_type == "application/pdf"
        or data.lstrip().startswith(b"%PDF")
    )

    # ========================================================
    # PDF INPUT
    # ========================================================

    if is_pdf:

        if not data.lstrip().startswith(b"%PDF"):
            raise HTTPException(
                400,
                "The uploaded file is not a valid PDF."
            )

        path = UPLOADS / (
            uuid.uuid4().hex + ".pdf"
        )

        path.write_bytes(
            data
        )

        # analyzer.py receives the original PDF bytes.
        result = analyze_eeg_image(
            data
        )

        result["file"] = {
            "name": file.filename or "EEG PDF",
            "type": "pdf",
            "content_type": "application/pdf",
            "size_bytes": len(data),
        }

        return result

    # ========================================================
    # IMAGE INPUT
    # ========================================================

    try:
        im = Image.open(
            io.BytesIO(data)
        )

        im.verify()

        im = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            400,
            "The uploaded file is not a valid image."
        )

    path = UPLOADS / (
        uuid.uuid4().hex + ".png"
    )

    im.save(
        path,
        "PNG"
    )

    result = analyze_eeg_image(
        data
    )

    result["file"] = {
        "name": file.filename or "EEG image",
        "type": "image",
        "content_type": file.content_type,
        "width": im.width,
        "height": im.height,
        "size_bytes": len(data),
    }

    return result