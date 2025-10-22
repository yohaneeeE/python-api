# main.py
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

# Optional Gemini import (only used if GEMINI_API_KEY is present)
try:
    from google import genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# Load .env file
load_dotenv()

# Read variables
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Tesseract executable path if provided
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Initialize Gemini client if available and key present
genai_client = None
if _HAS_GENAI and GEMINI_API_KEY:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        genai_client = None

# ---------------------------
# Input Schema (for reference)
# ---------------------------
class StudentInput(BaseModel):
    python: int
    sql: int
    java: int

# ---------------------------
# Train Structured Data Model
# ---------------------------
df = pd.read_csv("bsit_students.csv")

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
# FastAPI App with CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Subject Groups & Buckets
# ---------------------------
subjectGroups = {
    "programming": [
        "programming", "java", "oop", "object oriented",
        "software", "coding", "development", "elective"
    ],
    "databases": [
        "database", "sql", "dbms", "systems integration",
        "information systems", "data management"
    ],
    "ai_ml": [
        "python", "machine learning", "ai", "data mining",
        "analytics", "security", "assurance"
    ],
    "networking": [
        "networking", "networks", "cloud", "infrastructure"
    ],
    "webdev": [
        "html", "css", "javascript", "frontend", "backend", "php", "web"
    ],
    "systems": [
        "operating systems", "os", "architecture", "computer systems"
    ]
}

bucketMap = {
    "programming": "Java",
    "databases": "SQL",
    "ai_ml": "Python"
}

ignore_keywords = [
    "course", "description", "final", "remarks", "re-exam", "units",
    "fullname", "year level", "program", "college", "student no",
    "academic year", "date printed", "gwa", "credits", "republic", "city", "report",
    "gender", "bachelor", "semester", "university"
]

# ---------------------------
# Career → Required Skills Map
# ---------------------------
careerSkillMap = {
    "Software Engineer": ["programming", "databases"],
    "Data Scientist": ["ai_ml", "programming", "databases"],
    "Cloud Solutions Architect": ["networking", "databases", "programming"],
    "Web Developer": ["webdev", "programming", "databases"],
    "Computer Vision Engineer": ["ai_ml", "programming"],
    "NLP Research Scientist": ["ai_ml", "programming"]
}

# ---------------------------
# Hardcoded Certificate Suggestions
# ---------------------------
careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp", "Meta Frontend Dev", "Responsive Web Design"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer Cert."],
    "Database Administrator": ["Oracle SQL Associate", "Microsoft SQL Server"],
    "Cloud Solutions Architect": ["AWS Solutions Architect", "Azure Fundamentals"],
    "Cybersecurity Specialist": ["CompTIA Security+", "Cisco CyberOps Associate"],
    "General Studies": ["Short IT courses to explore career interests"]
}

# ---------------------------
# OCR Helpers
# ---------------------------
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

TEXT_FIXES = {
    "lective": "Elective",
    "hective": "Elective",
    "pen aire": "PE",
    "pathfit": "PE",
    "grmmunication": "Communication",
    "cobege": "College"
}

def clean_subject_text(desc: str) -> str:
    d = (desc or "").lower()
    if "pen aire" in d or "pathfit" in d:
        return "PE"
    if "lective" in d or "hective" in d:
        match = re.search(r'(\d+)', d)
        if match:
            num = match.group(1)[-1]
            return f"Elective {num}"
        return "Elective"
    if "purposive" in d and "communication" in d:
        return "Purposive Communication"
    for wrong, right in TEXT_FIXES.items():
        if wrong in d:
            d = d.replace(wrong, right.lower())
    return d.title()

# ---------------------------
# Subject Classifier Helpers
# ---------------------------
def classify_subject(desc: str):
    d = (desc or "").lower()
    if "elective" in d:
        return "Major Subject"
    if any(k in d for k in [
        "programming", "database", "data", "system", "integration", "architecture",
        "software", "network", "computing", "information", "security", "java",
        "python", "sql", "web", "algorithm"
    ]):
        return "IT Subject"
    return "Minor Subject"

def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if s == "":
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
# Gemini Text Cleanup
# ---------------------------
async def clean_text_with_gemini(ocr_text: str):
    if not genai_client:
        return ocr_text

    try:
        prompt = f"""
You are a text cleaner and spell corrector for OCR-processed academic transcripts.

Clean and correct the following text:
- Fix spelling errors in subject names.
- Remove repeated or garbage lines.
- Preserve subject names and grades.
- Keep only meaningful subject-related text.
- Do NOT add extra commentary or formatting.

OCR TEXT:
{ocr_text}
"""
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.0-flash",
            contents=prompt
        )

        cleaned = getattr(response, "text", None)
        return cleaned.strip() if cleaned else ocr_text
    except Exception as e:
        print(f"[Gemini Cleanup Error] {e}")
        return ocr_text

# ---------------------------
# OCR Extraction
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured = []
    rawSubjects = OrderedDict()
    normalizedText = {}
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if any(kw in low for kw in ignore_keywords):
            continue
        clean = re.sub(r'[^\w\.\-\s]', ' ', line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) < 2:
            continue

        float_tokens = []
        for i, tok in enumerate(parts):
            token_clean = re.sub(r'[^0-9.]', '', tok)
            if token_clean and re.search(r'\d', token_clean):
                try:
                    float_tokens.append((i, token_clean, float(token_clean)))
                except:
                    continue

        if not float_tokens:
            continue

        idx, tok, rawf = float_tokens[-1]
        gradeVal = _normalize_grade_str(tok)
        gradeVal = snap_to_valid_grade(gradeVal)

        subjDesc = " ".join(parts[:idx]).strip().title()
        subjDesc = clean_subject_text(subjDesc)
        category = classify_subject(subjDesc)

        assigned_bucket = None
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
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

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Gemini Enhancement (Career Advice)
# ---------------------------
async def enhance_with_gemini(careerOptions, mappedSkills, finalBuckets):
    if not genai_client:
        return None
    try:
        prompt = f"""
You are a helpful career advisor for BSIT students.
Given this student's analysis:
- Final numeric buckets: {finalBuckets}
- Mapped skill levels and buckets: {mappedSkills}
- Top career predictions: {careerOptions}

Provide 3 concise, practical, and personalized suggestions (mention technologies, projects, or certs).
Keep output under 5 sentences and use a motivational friendly tone.
"""
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        try:
            return response.text.strip() if hasattr(response, "text") else str(response)
        except Exception:
            return str(response)
    except Exception as e:
        return f"(Gemini unavailable: {e})"

# ---------------------------
# Predict Career
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets.get("Python", 3.0),
        "SQL": finalBuckets.get("SQL", 3.0),
        "Java": finalBuckets.get("Java", 3.0),
    }])

    proba = model.predict_proba(dfInput)[0]
    careers = []
    for encoded_label, p in zip(model.classes_, proba):
        try:
            career_label = targetEncoder.inverse_transform([int(encoded_label)])[0]
        except Exception:
            career_label = str(encoded_label)
        careers.append({"career": career_label, "confidence": round(float(p) * 100, 2)})

    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
    return careers

# ---------------------------
# API Route
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        # --- OCR and Gemini Cleanup ---
        raw_text = await asyncio.to_thread(pytesseract.image_to_string, img)
        text = await clean_text_with_gemini(raw_text)

        # --- Extraction and Prediction ---
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        # --- Gemini Enhancement ---
        gemini_enhancement = await enhance_with_gemini(careerOptions, mappedSkills, finalBuckets)

        # --- Readable Skills Output ---
        mappedSkills_readable = {}
        for subj, meta in mappedSkills.items():
            level = meta.get("level", "Unknown")
            bucket = meta.get("bucket")
            if bucket:
                mappedSkills_readable[subj] = f"{level} ({bucket})"
            else:
                mappedSkills_readable[subj] = level

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "geminiSuggestions": gemini_enhancement,
            "subjects_structured": subjects_structured,
            "mappedSkills": mappedSkills_readable,
            "finalBuckets": finalBuckets,
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Health Route
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API running with Gemini cleanup."}
