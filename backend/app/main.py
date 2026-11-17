from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from app.core.parser import extract_text_from_pdf, extract_text_from_docx
from app.ml.analyzer import CVCustomModel
from app.schemas.response import CVAnalysisResponse

cv_model = CVCustomModel(weights_path="weights/cv_role_classifier.joblib")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server boots up
    cv_model.load()
    yield
    # Cleanup on shutdown if needed

app = FastAPI(
    title="Custom CV Analyzer API",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.responses import FileResponse

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

# Allow frontend (React / Streamlit) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
def health_check():
    is_ready = (cv_model.gatekeeper is not None) and (cv_model.role_classifier is not None)
    return {"status": "healthy", "model_loaded": is_ready}

@app.post("/api/v1/analyze", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None)
):
    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 5 * 1024 * 1024:  # 5 MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 5 MB limit.")

    # 1. Extract text
    if filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_bytes)
    else:
        extracted_text = extract_text_from_docx(file_bytes)

    if len(extracted_text) < 50:
        raise HTTPException(status_code=422, detail="Could not extract readable text. Is this a scanned image PDF?")

    # 2. Run custom model & advisor
    result = cv_model.predict(
        filename=file.filename,
        cv_text=extracted_text,
        job_description=job_description
    )
    return result