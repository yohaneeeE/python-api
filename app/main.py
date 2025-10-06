import os, io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.utils import extract_text, extract_subjects_and_grades
from app.model_training import structured_model, target_le, label_encoders, text_model, vectorizer
import pandas as pd

app = FastAPI(title="Career Prediction API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Extract text
        text = extract_text(file)

        # Extract subjects and grades
        subjects_structured, mappedSkills, finalBuckets = extract_subjects_and_grades(text)

        # Structured prediction
        structured_input = pd.DataFrame([finalBuckets])
        pred_idx = structured_model.predict(structured_input)[0]
        structured_career = target_le.inverse_transform([pred_idx])[0]

        # Text-based prediction
        X_input_text = vectorizer.transform([text])
        text_pred = text_model.predict(X_input_text)[0]

        return JSONResponse({
            "text_career_prediction": text_pred,
            "structured_career_prediction": structured_career,
            "subjects_structured": subjects_structured,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "extracted_text": text
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
