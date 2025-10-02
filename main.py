
import os
from dotenv import load_dotenv
import re
import io
from collections import OrderedDict
from typing import List

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pytesseract
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Load .env file
load_dotenv()

# Read variables
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")


# Windows Tesseract path (adjust if needed)
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------
# Input Schema
# ---------------------------
class StudentInput(BaseModel):
    python: int
    sql: int
    java: int

# ---------------------------
# Train Structured Data Model
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
# FastAPI App with CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
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
# OCR Fixes
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
    d = desc.lower()

    # --- Fix PE (PE / PathFit) ---
    if "pen aire" in d or "pathfit" in d:
        return "PE"

    # --- Fix Elective with numbers ---
    if "lective" in d or "hective" in d:
        # try to capture number (e.g., "312 Lective" => Elective 4)
        match = re.search(r'(\d+)', d)
        if match:
            num = match.group(1)[-1]  # take last digit
            return f"Elective {num}"
        return "Elective"

    # --- Purposive Communication ---
    if "purposive" in d and "communication" in d:
        return "Purposive Communication"

    # General replacements
    for wrong, right in TEXT_FIXES.items():
        if wrong in d:
            d = d.replace(wrong, right.lower())

    return d.title()
# ---------------------------
# Helpers
# ---------------------------
def classify_subject(desc: str):
    d = desc.lower()
    if "elective" in d:
        return "Major Subject"
    if any(k in d for k in [
        "programming", "database", "data", "system", "integration", "architecture",
        "software", "network", "computing", "information", "security", "java",
        "python", "sql", "web", "algorithm"
    ]):
        return "IT Subject"
    return "Minor Subject"

def normalize_code(text: str) -> str:
    if not text:
        return None
    return re.sub(r'\s+', '', text.upper())

def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if s == "":
        return None
    try:
        raw = float(s)
    except:
        return None

    candidates = [raw, raw / 10.0, raw / 100.0]
    valid = [c for c in candidates if 1.0 <= c <= 5.0]
    if valid:
        chosen = min(valid, key=lambda x: abs(x - 2.5))
        return round(chosen, 2)

    if raw >= 10:
        if raw / 10.0 <= 5.0:
            return round(raw / 10.0, 2)
        if raw / 100.0 <= 5.0:
            return round(raw / 100.0, 2)

    if 0.0 < raw <= 5.0:
        return round(raw, 2)

    return round(raw, 2)

# ---------------------------
# OCR Extraction
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured = []
    rawSubjects = OrderedDict()
    normalizedText = {}
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        low = line.lower()
        if any(kw in low for kw in ignore_keywords):
            continue

        clean = re.sub(r'[\t\r\f\v]+', ' ', line)
        clean = re.sub(r'[^\w\.\-\s]', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 2:
            continue

        subjCode = None
        if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
            subjCode = f"{parts[0].upper()} {parts[1]}"
            parts = parts[2:]
        elif re.match(r'^[A-Z]{1,4}\d{2,3}$', parts[0].upper()):
            subjCode = parts[0].upper()
            parts = parts[1:]

        if not parts:
            continue

        remarks = None
        if parts and parts[-1].isalpha():
            remarks = parts[-1]
            parts = parts[:-1]
            if not parts:
                continue

        float_tokens = []
        for i, tok in enumerate(parts):
            token_clean = re.sub(r'[^0-9.]', '', tok)
            if token_clean and re.search(r'\d', token_clean):
                try:
                    rawf = float(token_clean)
                    float_tokens.append((i, token_clean, rawf))
                except:
                    continue

        gradeVal = None
        unitsVal = None
        grade_idx = None

        if len(float_tokens) >= 2:
            prev_idx, prev_tok, prev_raw = float_tokens[-2]
            last_idx, last_tok, last_raw = float_tokens[-1]
            grade_idx = prev_idx
            gradeVal = _normalize_grade_str(prev_tok)
            gradeVal = snap_to_valid_grade(gradeVal)
            unitsVal = float(last_raw)
        elif len(float_tokens) == 1:
            idx, tok, rawf = float_tokens[0]
            grade_idx = idx
            gradeVal = _normalize_grade_str(tok)
            gradeVal = snap_to_valid_grade(gradeVal)
            unitsVal = None
        else:
            continue

        desc_tokens = parts[:grade_idx] if grade_idx is not None else parts[:]
        if desc_tokens and re.fullmatch(r'\d+', desc_tokens[0]):
            desc_tokens = desc_tokens[1:]

        subjDesc = " ".join(desc_tokens).strip().title()
        subjDesc = clean_subject_text(subjDesc)
        if not subjDesc:
            subjDesc = subjCode or "Unknown Subject"

        subjKey = f"{subjCode} {subjDesc}" if subjCode else subjDesc
        category = classify_subject(subjDesc)

    # determine mapping to skill bucket
        assigned_bucket = None
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
                    # append grade to bucket_grades
                    bucket_grades[assigned_bucket].append(gradeVal)
                break

        # NEW: store skill level instead of bucket name
        mappedSkills[subjDesc] = grade_to_level(gradeVal) if gradeVal is not None else "Unknown"

        subjects_structured.append({
            "code": subjCode,
            "description": subjDesc,
            "grade": gradeVal,
            "units": float(unitsVal) if unitsVal is not None else None,
            "remarks": remarks,
            "category": category
        })

        rawSubjects[subjKey] = gradeVal
        normalizedText[subjKey] = subjDesc

    finalBuckets = {}
    for b, grades in bucket_grades.items():
        if grades:
            finalBuckets[b] = round(sum(grades) / len(grades), 2)
        else:
            finalBuckets[b] = 3.0

    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"],
    }])

    proba = model.predict_proba(dfInput)[0]
    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        suggestions = []
        for skill, grade in finalBuckets.items():
            if grade is None:
                continue
            subjMatches = [subj for subj, mapped in mappedSkills.items() if mapped == skill]
            if grade >= 2.75:
                for subj in subjMatches:
                    suggestions.append(
                        f"Your performance in {subj} suggests you need to strengthen {skill} skills "
                        f"to better align with {c['career']} roles."
                    )
            elif 2.0 <= grade < 2.75:
                for subj in subjMatches:
                    suggestions.append(
                        f"Improving your foundation in {subj} will increase opportunities in {c['career']}."
                    )
        if "Developer" in c["career"] or "Engineer" in c["career"]:
            suggestions.append("Focus on coding projects and internships to gain practical experience.")
        if "Data" in c["career"] or "AI" in c["career"]:
            suggestions.append("Consider hands-on Python/ML projects to solidify applied skills.")
        if "Database" in c["career"] or "Architect" in c["career"]:
            suggestions.append("Build database design and cloud deployment skills for real-world readiness.")
        if not suggestions:
            suggestions.append(f"Great work! You’re already strong for {c['career']}.")
        c["suggestion"] = " ".join(suggestions)
        c["certificates"] = careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])

    return careers

# ---------------------------
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns well with AI/ML and Data Scientist roles.",
        "webdev": "Web Development certificate enhances your frontend/backend developer profile.",
        "python": "Python certification supports Data Science, AI, and Software Engineering careers."
    }
    for cert in certFiles:
        certName = cert.filename.lower()
        matched = [msg for key, msg in certificateSuggestions.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds additional value to your career profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Routes
# ---------------------------
@app.post("/ocrPredict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        text = await asyncio.to_thread(pytesseract.image_to_string, img)

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text.strip())
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        if not careerOptions:
            careerOptions = [{
                "career": "General Studies",
                "confidence": 50.0,
                "suggestion": "Add more subjects or improve grades for a better match.",
                "certificates": careerCertSuggestions["General Studies"]
            }]

        certResults = []
        if certificateFiles:
            certResults = analyzeCertificates(certificateFiles or [])
        else:
            certResults = [{"info": "No certificates uploaded"}]

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
