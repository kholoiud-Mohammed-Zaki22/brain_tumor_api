import os
import io
import threading
import numpy as np
from PIL import Image
import requests
import gdown
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/tmp/brain_tumors_classifier.keras"
GDRIVE_ID        = "15_PDLvSaciAKj8BzXkQZShEhPk5WXTH2"
IMG_SIZE         = (224, 224)
CLASS_NAMES      = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAX_FILE_SIZE_MB = 10
MIN_MODEL_BYTES  = 50_000_000  # 50 MB — real model is ~300 MB

model        = None
model_lock   = threading.Lock()
startup_done = False

# ── Download helpers ──────────────────────────────────────────────────────────
def _download_with_requests(file_id: str, dest: str):
    session = requests.Session()
    url     = "https://drive.google.com/uc"
    params  = {"id": file_id, "export": "download"}

    response = session.get(url, params=params, stream=True, timeout=(30, 600))
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params["confirm"] = value
            response = session.get(url, params=params, stream=True, timeout=(30, 600))
            break

    try:
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
    finally:
        response.close()
        session.close()


def _file_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= MIN_MODEL_BYTES


def download_model():
    if _file_ok(MODEL_PATH):
        return

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    print("Downloading model from Google Drive...")

    # Attempt 1 — gdown
    try:
        url = f"https://drive.google.com/file/d/{GDRIVE_ID}/view"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        if _file_ok(MODEL_PATH):
            print("Download complete via gdown ✓")
            return
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print("gdown produced invalid file, trying fallback...")
    except Exception as e:
        print(f"gdown failed ({e}), trying requests fallback...")

    # Attempt 2 — requests
    _download_with_requests(GDRIVE_ID, MODEL_PATH)
    if not _file_ok(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
        raise RuntimeError(f"Model download failed — file is {size} bytes.")
    print("Download complete via requests ✓")


def get_model():
    global model
    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model
        download_model()
        print("Loading model...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e1:
            try:
                import keras
                model = keras.saving.load_model(MODEL_PATH, compile=False)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model.\n  tf: {e1}\n  keras: {e2}")
        print("Model loaded ✓")
    return model

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_done
    try:
        get_model()
    except Exception as e:
        print(f"WARNING: Could not load model at startup: {e}")
    finally:
        startup_done = True
    yield

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Classifier API",
    description=(
        "Classifies brain MRI images into one of four categories: "
        "Glioma, Meningioma, No Tumor, Pituitary.\n\n"
        "Model: EfficientNetB0 | Input size: 224×224 RGB"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    model: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Brain Tumor Classifier API is running. POST /predict to classify an MRI."}


@app.get("/health", tags=["Health"])
def health():
    if not startup_done:
        return JSONResponse(
            status_code=202,
            content={"status": "loading", "model_loaded": False, "classes": CLASS_NAMES}
        )
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "model_loaded": False, "classes": CLASS_NAMES}
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "model_loaded": True, "classes": CLASS_NAMES}
    )


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="MRI image (JPG/PNG)")):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        m = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")

    arr   = preprocess_image(image_bytes)
    preds = m.predict(arr, verbose=0)[0]

    predicted_idx   = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(preds[predicted_idx]) * 100

    probabilities = {
        name: round(float(prob) * 100, 2)
        for name, prob in zip(CLASS_NAMES, preds)
    }

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=round(confidence, 2),
        probabilities=probabilities,
        model="EfficientNetB0",
    )
