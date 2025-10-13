# filename: decisiontree_api.py (Enhanced with Gemini)
import re
import io
from collections import OrderedDict
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFilter
import pytesseract
import asyncio
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# ---------- CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")  # Replace with your Gemini API key

# ---------- MACHINE LEARNING SETUP ----------
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

# ---------- FASTAPI ----------
app = FastAPI(title="Career Prediction API (Enhanced OCR + Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- OCR ENHANCEMENT ----------
def preprocess_image(img: Image.Image) -> Image.Image:
    """Improve clarity before OCR."""
    img = img.convert("L")  # grayscale
    img = img.filter(ImageFilter.MedianFilter())
    img = img.point(lambda x: 0 if x < 140 else 255, "1")  # threshold
    w, h = img.size
    img = img.resize((int(w * 1.8), int(h * 1.8)))  # upscale for better accuracy
    return img

async def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text via Tesseract with better config."""
    img = Image.open(io.BytesIO(image_bytes))
    img = preprocess_image(img)
    text = await asyncio.to_thread(pytesseract.image_to_string, img, config="--oem 3 --psm 6 -l eng")
    text = re.sub(r'[^A-Za-z0-9.\n\s-]', ' ', text)
    return text

# ---------- GEMINI HELPERS ----------
async def gemini_clean_subjects_and_grades(ocr_text: str):
    """Use Gemini to clean OCR output, correct typos and normalize grades."""
    prompt = f"""
    Clean and structure this OCR transcript of student grades.
    - Fix typos and capitalization of subject names.
    - Correct misspellings (e.g. 'Platfom Techologies' → 'Platform Technologies').
    - Make sure all grades have decimals (e.g. 2 → 2.00).
    - Return JSON in this format:
      {{
        "subjects": [{{"name": "Subject Name", "grade": 1.75}}],
        "skills": {{}}
      }}
    Text:
    {ocr_text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = await asyncio.to_thread(model.generate_content, prompt)
    try:
        json_str = response.text.strip()
        import json
        data = json.loads(json_str)
        return data
    except Exception:
        return {"subjects": [], "skills": {}}

async def gemini_generate_career_suggestions(finalBuckets, subjects):
    """Generate top careers and plain text suggestions (3–4 sentences)."""
    prompt = f"""
    Based on these skills and average grades:
    {finalBuckets}
    and these subjects:
    {subjects}

    Suggest 3–4 best IT-related career paths.
    For each, write 3–4 sentences of plain text advice (no emojis).
    Example output:
    {{
      "careers": [
        {{"career": "Software Engineer", "confidence": 90, "suggestion": "Your strong programming skills indicate..."}},
        ...
      ]
    }}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = await asyncio.to_thread(model.generate_content, prompt)
    import json
    try:
        return json.loads(response.text.strip())
    except Exception:
        return {"careers": []}

# ---------- MAIN PREDICTION ROUTE ----------
@app.post("/predict")
async def ocr_predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        raw_text = await extract_text_from_image(image_bytes)

        # Let Gemini clean up the OCR result
        cleaned = await gemini_clean_subjects_and_grades(raw_text)
        subjects = cleaned.get("subjects", [])

        # Compute average buckets from subjects
        def average(lst): return round(sum(lst) / len(lst), 2) if lst else 3.0
        py, sql, java = [], [], []

        for subj in subjects:
            name = subj.get("name", "").lower()
            grade = subj.get("grade", 3.0)
            if any(k in name for k in ["python", "machine learning", "ai"]):
                py.append(grade)
            elif any(k in name for k in ["sql", "database", "data"]):
                sql.append(grade)
            elif any(k in name for k in ["java", "programming", "oop"]):
                java.append(grade)

        finalBuckets = {
            "Python": average(py),
            "SQL": average(sql),
            "Java": average(java)
        }

        # Get Gemini-based career suggestions
        gemini_result = await gemini_generate_career_suggestions(finalBuckets, subjects)
        careers = gemini_result.get("careers", [])
        topCareer = careers[0]["career"] if careers else "General Studies"

        return {
            "careerPrediction": topCareer,
            "careerOptions": careers,
            "subjects_structured": subjects,
            "finalBuckets": finalBuckets,
            "rawOCR": raw_text
        }
    except Exception as e:
        return {"error": str(e)}
