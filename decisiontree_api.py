import os
from dotenv import load_dotenv
import re
import io
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, UnidentifiedImageError
import pytesseract
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Gemini import (optional)
try:
    from google import genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

genai_client = None
if _HAS_GENAI and GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        genai_client = None

# ---------------------------
# Load Dataset and Train Model
# ---------------------------
df = pd.read_csv("bsit_students.csv")

features = ["Python", "SQL", "Java"]
target = "Future Career"

labelEncoders = {}
for col in features:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        labelEncoders[col] = le

targetEncoder = LabelEncoder()
df[target] = targetEncoder.fit_transform(df[target])

X = df[features]
y = df[target]

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X, y)

# ---------------------------
# FastAPI + CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (with Gemini cleanup ✨)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Subject Keywords and Mappings
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "software", "coding", "development", "elective"],
    "databases": ["database", "sql", "dbms", "systems integration", "information systems"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics", "security", "assurance"],
    "networking": ["network", "infrastructure", "cloud"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "os", "architecture", "computer systems"]
}

bucketMap = {
    "programming": "Java",
    "databases": "SQL",
    "ai_ml": "Python"
}

ignore_keywords = [
    "course", "description", "final", "remarks", "re-exam", "units",
    "fullname", "year level", "program", "college", "student no",
    "academic year", "gwa", "credits", "republic", "city", "report",
    "gender", "bachelor", "semester", "university"
]

VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

def grade_to_level(grade: float) -> str:
    if grade is None:
        return "Unknown"
    if grade <= 1.75:
        return "Strong"
    elif grade <= 2.5:
        return "Average"
    else:
        return "Weak"

def snap_to_valid_grade(val: float):
    if val is None:
        return None
    return min(VALID_GRADES, key=lambda g: abs(g - val))

def classify_subject(desc: str):
    d = desc.lower()
    if "elective" in d:
        return "Major Subject"
    if any(k in d for k in ["programming", "database", "system", "architecture", "software", "network", "security", "python", "java", "sql", "web"]):
        return "IT Subject"
    return "Minor Subject"

def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if not s:
        return None
    try:
        raw = float(s)
    except:
        return None
    if 0.0 < raw <= 5.0:
        return round(raw, 2)
    if raw >= 10 and raw / 10.0 <= 5.0:
        return round(raw / 10.0, 2)
    if raw >= 100 and raw / 100.0 <= 5.0:
        return round(raw / 100.0, 2)
    return round(raw, 2)

# ---------------------------
# Gemini Text Cleaner
# ---------------------------
async def clean_text_with_gemini(ocr_text: str):
    """Uses Gemini to fix spelling, remove redundant lines, and normalize text."""
    if not genai_client:
        return ocr_text

    try:
        prompt = f"""
You are a smart text normalizer for OCR academic transcripts.
Tasks:
- Fix spelling errors in subject names.
- Remove duplicates or repeated noisy lines.
- Keep only lines containing subjects with their grades.
- Ensure subject names are human-readable (e.g. "Purposive Communication", "System Integration and Architecture 1").
- Do NOT include commentary, introductions, or formatting.

OCR text:
{ocr_text}
"""
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        cleaned = getattr(response, "text", str(response))
        cleaned = cleaned.strip()

        # Filter valid-looking lines (with numbers)
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        filtered = [l for l in lines if re.search(r'\d', l)]
        if not filtered:
            return cleaned
        return "\n".join(filtered)
    except Exception as e:
        print(f"[Gemini Cleanup Error] {e}")
        return ocr_text

# ---------------------------
# OCR → Structured Data
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured, rawSubjects, normalizedText, mappedSkills = [], OrderedDict(), {}, {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for raw_line in lines:
        if any(kw in raw_line.lower() for kw in ignore_keywords):
            continue

        clean = re.sub(r'[^\w\.\-\s]', ' ', raw_line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        parts = clean.split()
        if len(parts) < 2:
            continue

        # Find numeric grade
        float_tokens = [(i, tok) for i, tok in enumerate(parts) if re.search(r'\d', tok)]
        if not float_tokens:
            continue
        idx, tok = float_tokens[-1]
        gradeVal = snap_to_valid_grade(_normalize_grade_str(tok))

        subjDesc = " ".join(parts[:idx]).strip().title()
        category = classify_subject(subjDesc)

        assigned_bucket = None
        for group, keywords in subjectGroups.items():
            if any(k in subjDesc.lower() for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket:
                    bucket_grades.setdefault(assigned_bucket, []).append(gradeVal)
                break

        mappedSkills[subjDesc] = {
            "level": grade_to_level(gradeVal),
            "bucket": assigned_bucket
        }

        subjects_structured.append({
            "description": subjDesc,
            "grade": gradeVal,
            "category": category
        })
        rawSubjects[subjDesc] = gradeVal
        normalizedText[subjDesc] = subjDesc

    finalBuckets = {}
    for b, grades in bucket_grades.items():
        finalBuckets[b] = round(sum(grades) / len(grades), 2) if grades else 3.0
    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Gemini Career Suggestions
# ---------------------------
async def enhance_with_gemini(careerOptions, mappedSkills, finalBuckets):
    if not genai_client:
        return None
    try:
        prompt = f"""
You are a friendly career advisor.
Given:
- Final numeric buckets: {finalBuckets}
- Skills: {mappedSkills}
- Predicted careers: {careerOptions}

Give 3 practical, short suggestions (2–3 sentences total) on what the student should do next (projects, technologies, certifications).
Be positive and realistic.
"""
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        return f"(Gemini unavailable: {e})"

# ---------------------------
# Predict Career
# ---------------------------
def predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets.get("Python", 3.0),
        "SQL": finalBuckets.get("SQL", 3.0),
        "Java": finalBuckets.get("Java", 3.0),
    }])
    proba = model.predict_proba(dfInput)[0]
    careers = []
    for i, p in enumerate(proba):
        career_label = targetEncoder.inverse_transform([i])[0]
        careers.append({"career": career_label, "confidence": round(float(p) * 100, 2)})
    return sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

# ---------------------------
# /predict Route
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        raw_text = await asyncio.to_thread(pytesseract.image_to_string, img)
        cleaned_text = await clean_text_with_gemini(raw_text) if raw_text.strip() else raw_text

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(cleaned_text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)
        gemini_enhancement = await enhance_with_gemini(careerOptions, mappedSkills, finalBuckets)

        # readable output
        mappedSkills_readable = {
            subj: f"{meta['level']} ({meta['bucket']})" if meta['bucket'] else meta['level']
            for subj, meta in mappedSkills.items()
        }

        return {
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "Unknown",
            "careerOptions": careerOptions,
            "geminiSuggestions": gemini_enhancement,
            "subjects_structured": subjects_structured,
            "mappedSkills": mappedSkills_readable,
            "finalBuckets": finalBuckets
        }
    except UnidentifiedImageError:
        return {"error": "Invalid image file."}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Health Route
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API running with Gemini cleanup."}
