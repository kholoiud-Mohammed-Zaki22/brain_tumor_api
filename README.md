# Brain Tumor Classifier API

EfficientNetB0-based API for brain MRI classification.

**Classes:** Glioma · Meningioma · No Tumor · Pituitary  
**Input:** 224×224 RGB image (JPG or PNG)

---

## Run locally

```bash
pip install -r requirements.txt

# Put your model file here, then:
uvicorn main:app --reload
```

Docs at: http://localhost:8000/docs

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Status check |
| GET | `/health` | Health + loaded classes |
| POST | `/predict` | Classify an MRI image |

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@brain_mri.jpg"
```

**Response:**
```json
{
  "predicted_class": "Glioma",
  "confidence": 97.43,
  "probabilities": {
    "Glioma": 97.43,
    "Meningioma": 1.12,
    "No Tumor": 0.89,
    "Pituitary": 0.56
  },
  "model": "EfficientNetB0"
}
```

---

## Docker

```bash
# Add your model file to the folder, uncomment the COPY line in Dockerfile, then:
docker build -t brain-tumor-api .
docker run -p 8000:8000 brain-tumor-api
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `brain_tumors_classifier.keras` | Path to `.keras` or `.h5` model |
