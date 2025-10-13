# filename: decisiontree_api.py
import os
import re
import io
import json
import asyncio
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
from google import genai  # ✅ Gemini API

# ---------------------------
# Gemini Setup
# ---------------------------
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    client = None
    print(f"Gemini client not initialized: {e}")

# ---------------------------
# Windows Tesseract path (adjust if needed)
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---------------------------
# OCR Preprocessing
# ---------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    """Enhance OCR accuracy via cleaning, thresholding, and upscaling."""
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    w, h = img.size
    img = img.resize((int(w * 1.8), int(h * 1.8)))
    return img

async def extract_text_from_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    img = preprocess_image(img)
    text = await asyncio.to_thread(
        pytesseract.image_to_string, img, config="--oem 3 --psm 6 -l eng"
    )
    text = re.sub(r"[^A-Za-z0-9.\n\s-]", " ", text)
    return text.strip()

# ---------------------------
# Gemini: Clean subjects and grades
# ---------------------------
async def gemini_clean_subjects_and_grades(ocr_text: str):
    if not client:
        return {"subjects": [], "skills": {}}

    prompt = f"""
    You are an OCR data corrector.
    Clean and structure this text extracted from a university grade report.
    - Fix typos, capitalization, and subject name errors.
    - Ensure grades are valid numbers (e.g., 2 or 2.00 → 2.00).
    - Return a JSON list of subjects with their grades.

    Example:
    {{
      "subjects": [
        {{"name": "Object-Oriented Programming", "grade": 2.50}},
        {{"name": "Platform Technologies", "grade": 2.50}}
      ]
    }}

    Text:
    {ocr_text}
    """

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        cleaned = json.loads(response.text.strip())
        return cleaned
    except Exception as e:
        print(f"Gemini clean error: {e}")
        return {"subjects": [], "skills": {}}

# ---------------------------
# Gemini: Generate career suggestions
# ---------------------------
async def gemini_generate_career_suggestions(finalBuckets, subjects):
    if not client:
        return {"careers": []}

    prompt = f"""
    Based on the student's average skill grades:
    {finalBuckets}

    And subjects taken:
    {subjects}

    Suggest 3–4 best IT-related career paths (e.g., Software Engineer, Data Analyst, Web Developer).
    For each, write 3–4 sentences of plain text advice.
    Use a professional tone and avoid emojis or slang.

    Output in JSON format:
    {{
      "careers": [
        {{
          "career": "Software Engineer",
          "confidence": 90,
          "suggestion": "You have strong programming skills..."
        }}
      ]
    }}
    """

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        result = json.loads(response.text.strip())
        return result
    except Exception as e:
        print(f"Gemini career error: {e}")
        return {"careers": []}

# ---------------------------
# Grammar + suggestion improver
# ---------------------------
async def improve_prediction_with_gemini(prediction_text: str) -> str:
    """Use Gemini to improve grammar, fix typos, and add extra related suggestions."""
    if not client:
        return prediction_text

    prompt = f"""
    You are an expert career counselor AI. Improve and reformat the following career prediction summary:
    - Correct grammar and typos.
    - Add 2–3 more relevant suggestions aligned with IT or the predicted career.
    - Maintain a professional, encouraging tone.
    - Keep the output concise and avoid emojis or slang.

    Prediction Summary:
    {prediction_text}
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini polish error: {e}")
        return prediction_text

# ---------------------------
# Machine Learning model setup
# ---------------------------
df = pd.read_csv("cs_students.csv")
features = ["Python", "SQL", "Java"]
target = "Future Career"

data = df.copy()
labelEncoders = {}

for col in features:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        labelEncoders[col] = le

targetEncoder = LabelEncoder()
data[target] = targetEncoder.fit_transform(data[target])

X = data[features]
y = data[target]
model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X, y)

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="Career Prediction API (Gemini OCR Enhanced)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Main Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def ocr_predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        ocr_text = await extract_text_from_image(image_bytes)

        # Step 1: Clean via Gemini
        cleaned = await gemini_clean_subjects_and_grades(ocr_text)
        subjects = cleaned.get("subjects", [])

        # Step 2: Compute averages
        def avg(vals): return round(sum(vals) / len(vals), 2) if vals else 3.0
        py, sql, java = [], [], []
        for subj in subjects:
            name = subj.get("name", "").lower()
            grade = subj.get("grade", 3.0)
            if any(k in name for k in ["python", "ai", "machine learning"]):
                py.append(grade)
            elif any(k in name for k in ["sql", "database", "data"]):
                sql.append(grade)
            elif any(k in name for k in ["java", "oop", "programming"]):
                java.append(grade)

        finalBuckets = {"Python": avg(py), "SQL": avg(sql), "Java": avg(java)}

        # Step 3: Predict using ML model
        df_input = pd.DataFrame([finalBuckets])
        proba = model.predict_proba(df_input)[0]
        top_idx = proba.argmax()
        topCareer = targetEncoder.inverse_transform([top_idx])[0]

        # Step 4: Get Gemini suggestions
        gemini_result = await gemini_generate_career_suggestions(finalBuckets, subjects)
        careers = gemini_result.get("careers", [])
        top_suggestion = careers[0]["suggestion"] if careers else f"You are suited for {topCareer}. Continue improving your technical and analytical skills."

        # Step 5: Refine text
        improved_suggestion = await improve_prediction_with_gemini(top_suggestion)

        return {
            "careerPrediction": topCareer,
            "careerOptions": careers,
            "subjects_structured": subjects,
            "finalBuckets": finalBuckets,
            "rawOCR": ocr_text,
            "summary": improved_suggestion
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API (Gemini OCR Enhanced) is running."}
