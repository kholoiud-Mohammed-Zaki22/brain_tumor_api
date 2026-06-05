import os
import io
import numpy as np
from PIL import Image
import gdown
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "brain_tumors_classifier.keras"
GDRIVE_ID   = "14buoAJd_rWSSrkadIGglk10cJ6IuFmRb"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAX_FILE_SIZE_MB = 10

# ── Download model if not exists ──────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_ID}&export=download&confirm=t",
        MODEL_PATH,
        quiet=False,
        fuzzy=True,
    )
    print("Download complete ✓")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    try:
        import keras
        model = keras.saving.load_model(MODEL_PATH, compile=False)
    except Exception as e2:
        raise RuntimeError(f"Failed to load model.\n  Attempt 1: {e}\n  Attempt 2: {e2}")
print("Model loaded ✓")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor Classifier API",
    description=(
        "Classifies brain MRI images into one of four categories: "
        "Glioma, Meningioma, No Tumor, Pituitary.\n\n"
        "Model: EfficientNetB0 | Input size: 224×224 RGB"
    ),
    version="1.0.0",
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
    img = img.resize(IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "Brain Tumor Classifier API is running. POST /predict to classify an MRI."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        classes=CLASS_NAMES,
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

    arr = preprocess_image(image_bytes)
    preds = model.predict(arr, verbose=0)[0]

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
