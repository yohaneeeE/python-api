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

# Optional Gemini import
try:
    from google import genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# Load environment variables
load_dotenv()
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Tesseract config
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Initialize Gemini client
genai_client = None
if _HAS_GENAI and GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------
# Model Training
# -----------------------------
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

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Career Prediction API (COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Subject Mappings
# -----------------------------
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
    "networking": ["networking", "networks", "cloud", "infrastructure"],
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
    "academic year", "date printed", "gwa", "credits", "republic", "city", "report",
    "gender", "bachelor", "semester", "university"
]

careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp", "Meta Frontend Dev", "Responsive Web Design"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer Cert."],
    "Database Administrator": ["Oracle SQL Associate", "Microsoft SQL Server"],
    "Cloud Solutions Architect": ["AWS Solutions Architect", "Azure Fundamentals"],
    "Cybersecurity Specialist": ["CompTIA Security+", "Cisco CyberOps Associate"],
    "General Studies": ["Short IT courses to explore career interests"]
}

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


def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if not s:
        return None
    try:
        raw = float(s)
    except:
        return None
    if 1 <= raw <= 5:
        return round(raw, 2)
    if 10 <= raw <= 50:
        return round(raw / 10.0, 2)
    return None


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


# -----------------------------
# OCR Extraction (Fixed)
# -----------------------------
def extractSubjectGrades(text: str):
    # ✅ Restrict OCR text to only the subject section
    start_marker = "CODE SUBJECT TITLE"
    end_marker = "TOTAL SUBJECTS ENROLLED"
    lower_text = text.lower()

    start_idx = lower_text.find(start_marker.lower())
    end_idx = lower_text.find(end_marker.lower())

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    elif start_idx != -1:
        text = text[start_idx:]

    subjects_structured = []
    rawSubjects = OrderedDict()
    normalizedText = {}
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]

    for raw_line in lines:
        if not re.search(r'\d\.\d{1,2}', raw_line):  # skip lines with no grade like 2.50
            continue

        clean = re.sub(r'[^\w\.\-\s]', ' ', raw_line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        float_tokens = [(i, t) for i, t in enumerate(parts) if re.match(r'^\d\.\d{1,2}$', t)]
        if not float_tokens:
            continue

        idx, tok = float_tokens[0]
        gradeVal = snap_to_valid_grade(_normalize_grade_str(tok))
        subjDesc = " ".join(parts[:idx]).strip().title()

        # Skip totals / summary lines
        if any(k in subjDesc.lower() for k in ["total", "gwa", "earned", "validation", "credit"]):
            continue

        category = classify_subject(subjDesc)
        assigned_bucket = None
        for group, keywords in subjectGroups.items():
            if any(k in subjDesc.lower() for k in keywords):
                assigned_bucket = bucketMap.get(group)
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

    # Compute final average per skill
    finalBuckets = {}
    for b, grades in bucket_grades.items():
        if grades:
            finalBuckets[b] = round(sum(grades) / len(grades), 2)
        else:
            finalBuckets[b] = 3.0

    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets


# -----------------------------
# Career Prediction
# -----------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets.get("Python", 3.0),
        "SQL": finalBuckets.get("SQL", 3.0),
        "Java": finalBuckets.get("Java", 3.0),
    }])

    proba = model.predict_proba(dfInput)[0]
    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p) * 100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        suggestions = []
        for skill, grade in finalBuckets.items():
            subjMatches = [s for s, meta in mappedSkills.items() if meta.get("bucket") == skill]
            if grade >= 2.75:
                for subj in subjMatches:
                    suggestions.append(f"Improve {subj} to strengthen {skill} for {c['career']}.")
            elif 2.0 <= grade < 2.75:
                for subj in subjMatches:
                    suggestions.append(f"Enhance fundamentals in {subj} for {c['career']}.")
        if not suggestions:
            suggestions.append(f"Great performance! You're well-prepared for {c['career']}.")
        c["suggestion"] = " ".join(suggestions)
        c["certificates"] = careerCertSuggestions.get(c["career"], ["General IT certifications recommended."])

    return careers


# -----------------------------
# Certificates
# -----------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    certMap = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns well with AI/ML and Data Scientist roles.",
        "webdev": "Web Development certificate enhances your frontend/backend developer profile.",
        "python": "Python certification supports Data Science, AI, and Software Engineering careers."
    }
    for cert in certFiles:
        name = (cert.filename or "").lower()
        matched = [msg for key, msg in certMap.items() if key in name]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds career value."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results


# -----------------------------
# Routes
# -----------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        imageBytes = await file.read()
        try:
            img = Image.open(io.BytesIO(imageBytes))
        except UnidentifiedImageError:
            return {"error": "Invalid image format."}
        if img.mode != "RGB":
            img = img.convert("RGB")

        text = await asyncio.to_thread(pytesseract.image_to_string, img)
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        if not careerOptions:
            careerOptions = [{
                "career": "General Studies",
                "confidence": 50.0,
                "suggestion": "Add more subjects or improve grades for better results.",
                "certificates": careerCertSuggestions["General Studies"]
            }]

        certResults = analyzeCertificates(certificateFiles or []) if certificateFiles else [{"info": "No certificates uploaded"}]

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API running."}
