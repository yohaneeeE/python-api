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
# NOTE: Ensure bsit_students.csv is present in working directory.
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
    "databases": ["database", "sql", "dbms", "systems integration", "information systems", "database systems"],
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
    "gender", "bachelor", "semester", "university", "date printed", "name"
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
    d = (desc or "").lower()
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

# local TEXT_FIXES fallback (for when Gemini is unavailable)
TEXT_FIXES = {
    "lective": "Elective",
    "hective": "Elective",
    "pen aire": "PE",
    "pathfit": "PE",
    "grmmunication": "Communication",
    "cobege": "College",
    "sysytem": "System",
    "integation": "Integration",
    "informaton": "Information",
    "securty": "Security"
}

def fallback_clean_text(ocr_text: str) -> str:
    """Simple, local cleaning: replacements, remove big garbage sections, keep plausible subject lines."""
    lines = []
    for raw in (ocr_text or "").splitlines():
        l = raw.strip()
        if not l:
            continue
        low = l.lower()
        if any(kw in low for kw in ignore_keywords):
            continue
        # Apply text fixes
        for wrong, right in TEXT_FIXES.items():
            if wrong in low:
                # do case-insensitive replace
                l = re.sub(re.escape(wrong), right, l, flags=re.IGNORECASE)
                low = l.lower()
        # Remove lines that are obviously garbage (too short or many symbols)
        if len(re.sub(r'[^A-Za-z0-9 ]', '', l)) < 3:
            continue
        lines.append(l)
    # keep lines that contain a digit OR contain known subject keywords
    filtered = []
    for l in lines:
        if re.search(r'\d', l):
            filtered.append(l)
            continue
        if any(k in l.lower() for klist in subjectGroups.values() for k in klist):
            filtered.append(l)
    return "\n".join(filtered)

# ---------------------------
# Gemini Text Cleaner
# ---------------------------
async def clean_text_with_gemini(ocr_text: str):
    """Uses Gemini to fix spelling, remove redundant lines, and normalize text.
       If Gemini fails or returns empty, fall back to local cleaning."""
    if not genai_client:
        # no Gemini — use fallback
        return fallback_clean_text(ocr_text)

    try:
        prompt = f"""
You are a smart text normalizer for OCR academic transcripts.
Tasks:
- Fix spelling errors in subject names.
- Remove duplicates / repeated noisy lines.
- Keep only lines containing subjects with their grades (subject name followed by grade or number).
- Ensure subject names are human-readable (e.g. "Purposive Communication", "Systems Integration and Architecture 1").
- Return plain text lines only (no commentary). 

OCR text:
{ocr_text}
"""
        # Use thread to call blocking SDK
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        cleaned = getattr(response, "text", None)
        if not cleaned:
            # Sometimes the response is not in .text — try string form
            cleaned = str(response)
        cleaned = cleaned.strip()

        # Defensive: if cleaned is empty or obviously long commentary, fallback
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        # prefer lines containing digits (grades)
        filtered = [l for l in lines if re.search(r'\d', l)]
        if filtered:
            return "\n".join(filtered)
        # If Gemini returned meaningful subject keywords lines (no digits), keep those
        filtered_keywords = [l for l in lines if any(k in l.lower() for klist in subjectGroups.values() for k in klist)]
        if filtered_keywords:
            return "\n".join(filtered_keywords)
        # final fallback: local cleaner
        return fallback_clean_text(cleaned or ocr_text)
    except Exception as e:
        # print for debugging (remove or change to proper logging in production)
        print(f"[Gemini cleanup failed] {e}")
        return fallback_clean_text(ocr_text)

# ---------------------------
# OCR → Structured Data
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured, rawSubjects, normalizedText, mappedSkills = [], OrderedDict(), {}, {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for raw_line in lines:
        low_line = raw_line.lower()
        if any(kw in low_line for kw in ignore_keywords):
            continue

        # remove extra punctuation but keep periods/dashes used in names
        clean = re.sub(r'[^\w\.\-\s]', ' ', raw_line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        parts = clean.split()
        if len(parts) < 1:
            continue

        # find numeric tokens; prefer LAST numeric token as grade
        float_tokens = []
        for i, tok in enumerate(parts):
            # allow numbers like 1.75, 125, 312 etc.
            if re.search(r'\d', tok):
                token_clean = re.sub(r'[^0-9.]', '', tok)
                try:
                    val = float(token_clean)
                except:
                    # skip malformed
                    continue
                float_tokens.append((i, tok, val))

        if not float_tokens:
            # no numeric tokens: maybe line is just a subject name (keep minimal)
            subjDesc = " ".join(parts).strip().title()
            subjDesc = subjDesc if subjDesc else None
            if not subjDesc:
                continue
            gradeVal = None
        else:
            idx, tok, rawf = float_tokens[-1]
            gradeVal = _normalize_grade_str(tok)
            gradeVal = snap_to_valid_grade(gradeVal)

        # subject description is tokens before the grade token if present, else full line
        if float_tokens:
            subj_tokens = parts[:idx]
        else:
            subj_tokens = parts[:]
        subjDesc = " ".join(subj_tokens).strip().title()
        # apply local fixes (smallfixes)
        for wrong, right in TEXT_FIXES.items():
            if wrong in subjDesc.lower():
                subjDesc = re.sub(re.escape(wrong), right, subjDesc, flags=re.IGNORECASE)
        subjDesc = subjDesc or "Unknown Subject"

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
            "level": grade_to_level(gradeVal) if gradeVal is not None else "Unknown",
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
    # ensure keys exist
    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Gemini Career Suggestions
# ---------------------------
async def enhance_with_gemini(careerOptions, mappedSkills, finalBuckets):
    if not genai_client:
        # local fallback advice (concise)
        tips = []
        # Recommend based on weak buckets
        for b, val in finalBuckets.items():
            if val >= 2.75:
                tips.append(f"Improve {b} fundamentals (projects, practice, online courses).")
        if not tips:
            tips.append("Keep building projects and portfolio work to strengthen your profile.")
        return " ".join(tips)

    try:
        prompt = f"""
You are a friendly career advisor.
Given:
- Final numeric buckets: {finalBuckets}
- Skills: {mappedSkills}
- Predicted careers: {careerOptions}

Give 3 practical, short suggestions (2–3 sentences total) on what the student should do next (projects, technologies, certifications).
Be positive and realistic and do not be redundant.
"""
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        cleaned = getattr(response, "text", None)
        if not cleaned:
            cleaned = str(response)
        cleaned = cleaned.strip()
        # if response is long, keep it (frontend will show), but ensure not empty
        return cleaned if cleaned else "(No additional suggestions)"
    except Exception as e:
        print(f"[Gemini enhancement error] {e}")
        # fallback: simple local suggestions
        tips = []
        for b, val in finalBuckets.items():
            if val >= 2.75:
                tips.append(f"Improve {b} fundamentals with small projects.")
        if not tips:
            tips.append("Work on practical projects and internships.")
        return " ".join(tips)

# ---------------------------
# Predict Career
# ---------------------------
def predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets.get("Python", 3.0),
        "SQL": finalBuckets.get("SQL", 3.0),
        "Java": finalBuckets.get("Java", 3.0),
    }])
    # ensure model can predict; if an error occurs, return a safe fallback
    try:
        proba = model.predict_proba(dfInput)[0]
    except Exception as e:
        print(f"[Model predict error] {e}")
        return [{"career": "General Studies", "confidence": 50.0}]

    careers = []
    # use enumerate so index matches encoder mapping used during training
    for i, p in enumerate(proba):
        try:
            career_label = targetEncoder.inverse_transform([i])[0]
        except Exception:
            career_label = str(i)
        careers.append({"career": career_label, "confidence": round(float(p) * 100, 2)})
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
    return careers

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

        # Run OCR (in background thread)
        raw_text = await asyncio.to_thread(pytesseract.image_to_string, img)
        print("[DEBUG] raw_text length:", len(raw_text or ""))
        # If OCR returned nothing, return a helpful message instead of silent empty
        if not raw_text or not raw_text.strip():
            # return helpful response so frontend can display message
            return {
                "error": "OCR produced no readable text. Please upload a clear transcript image (not a screenshot of this page).",
                "subjects_structured": [],
                "mappedSkills": {},
                "finalBuckets": {"Python": 3.0, "SQL": 3.0, "Java": 3.0},
                "careerOptions": [{"career": "General Studies", "confidence": 50.0}]
            }

        # Ask Gemini (if present) to clean text; fallback if Gemini fails
        cleaned_text = await clean_text_with_gemini(raw_text)

        # DEBUG prints to help you inspect what was passed downstream
        print("[DEBUG] cleaned_text preview:")
        for i, line in enumerate((cleaned_text or "").splitlines()[:20]):
            print(f"  {i+1}: {line}")

        # Parse cleaned text into structured subjects
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(cleaned_text)

        # If extraction produced no subjects, try a fallback: run local fallback_clean_text on raw_text then re-parse
        if not subjects_structured:
            fb = fallback_clean_text(raw_text)
            if fb and fb.strip() and fb != cleaned_text:
                print("[DEBUG] reattempt parsing after fallback_clean_text")
                subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(fb)

        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        # Ensure careerOptions fallback
        if not careerOptions:
            careerOptions = [{"career": "General Studies", "confidence": 50.0}]

        gemini_enhancement = await enhance_with_gemini(careerOptions, mappedSkills, finalBuckets)

        # readable mapped skills
        mappedSkills_readable = {}
        for subj, meta in mappedSkills.items():
            level = meta.get("level", "Unknown")
            bucket = meta.get("bucket")
            mappedSkills_readable[subj] = f"{level} ({bucket})" if bucket else level

        return {
            "careerPrediction": careerOptions[0].get("career", "Unknown"),
            "careerOptions": careerOptions,
            "geminiSuggestions": gemini_enhancement,
            "subjects_structured": subjects_structured,
            "mappedSkills": mappedSkills_readable,
            "finalBuckets": finalBuckets
        }
    except UnidentifiedImageError:
        return {"error": "Invalid image file."}
    except Exception as e:
        # keep errors readable, but do not leak stack traces in production
        return {"error": str(e)}

# ---------------------------
# Health Route
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API running with Gemini cleanup."}
