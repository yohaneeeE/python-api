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

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------
# Model Schema & Training
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
# FastAPI App + CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

allowed_origins = ["*"] if FRONTEND_URL == "*" else [FRONTEND_URL]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Career Prediction API is running ✅"}

# ---------------------------
# Data Maps & Helpers
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "software", "coding", "development", "elective"],
    "databases": ["database", "sql", "dbms", "information systems", "data management"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics", "security"],
    "networking": ["networking", "cloud", "infrastructure"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "architecture", "computer systems"],
}

bucketMap = {"programming": "Java", "databases": "SQL", "ai_ml": "Python"}

careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp", "Meta Frontend Dev", "Responsive Web Design"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer Cert."],
    "Database Administrator": ["Oracle SQL Associate", "Microsoft SQL Server"],
    "Cloud Solutions Architect": ["AWS Solutions Architect", "Azure Fundamentals"],
    "Cybersecurity Specialist": ["CompTIA Security+", "Cisco CyberOps Associate"],
    "General Studies": ["Short IT courses to explore career interests"],
}

VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

ignore_keywords = [
    "course", "description", "final", "remarks", "re-exam", "units", "fullname",
    "year level", "program", "college", "student no", "academic year", "gwa",
    "credits", "report", "gender", "semester", "university"
]

# ---------------------------
# Grade Utilities
# ---------------------------
def grade_to_level(grade: Optional[float]) -> str:
    if grade is None:
        return "Unknown"
    if grade <= 1.75:
        return "Strong"
    elif grade <= 2.5:
        return "Average"
    else:
        return "Weak"

def snap_to_valid_grade(val: Optional[float]):
    if val is None:
        return None
    return min(VALID_GRADES, key=lambda g: abs(g - val))

def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if not s:
        return None
    try:
        raw = float(s)
    except ValueError:
        return None
    if 1.0 <= raw <= 5.0:
        return round(raw, 2)
    if raw >= 10 and raw / 10.0 <= 5.0:
        return round(raw / 10.0, 2)
    if raw >= 100 and raw / 100.0 <= 5.0:
        return round(raw / 100.0, 2)
    return None

# ---------------------------
# OCR Parsing
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
        low = line.lower()
        if any(kw in low for kw in ignore_keywords):
            continue

        parts = re.sub(r'[^\w\.\-\s]', ' ', line).split()
        if len(parts) < 2:
            continue

        subjDesc = " ".join(parts[:-1]).title()
        gradeVal = _normalize_grade_str(parts[-1])
        gradeVal = snap_to_valid_grade(gradeVal)

        for group, keywords in subjectGroups.items():
            if any(k in subjDesc.lower() for k in keywords):
                b = bucketMap.get(group)
                if b and gradeVal:
                    bucket_grades[b].append(gradeVal)

        mappedSkills[subjDesc] = grade_to_level(gradeVal)
        subjects_structured.append({"description": subjDesc, "grade": gradeVal})
        rawSubjects[subjDesc] = gradeVal
        normalizedText[subjDesc] = subjDesc

    finalBuckets = {b: round(sum(v) / len(v), 2) if v else 3.0 for b, v in bucket_grades.items()}
    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction Logic
# ---------------------------
def predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills):
    dfInput = pd.DataFrame([finalBuckets])
    proba = model.predict_proba(dfInput)[0]

    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        c["suggestion"] = "Consider strengthening weaker subjects and exploring internships."
        c["certificates"] = careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])
    return careers

# ---------------------------
# Certificate Analyzer
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    certTips = {
        "aws": "AWS certificate strengthens Cloud & DevOps career paths.",
        "ccna": "CCNA enhances networking and system admin opportunities.",
        "datascience": "Data Science cert aligns with AI/ML roles.",
        "webdev": "Web Development cert boosts frontend/backend dev skills.",
        "python": "Python cert supports Data Science and Software Engg.",
    }
    for cert in certFiles:
        certName = cert.filename.lower()
        matched = [msg for key, msg in certTips.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds extra career value."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# /predict Endpoint
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        img_bytes = await file.read()
        try:
            img = Image.open(io.BytesIO(img_bytes))
        except UnidentifiedImageError:
            return {"error": "Uploaded file is not a valid image."}

        text = await asyncio.to_thread(pytesseract.image_to_string, img)
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text.strip())
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        if not careerOptions:
            careerOptions = [{
                "career": "General Studies",
                "confidence": 50.0,
                "suggestion": "Add more subjects or better grades for improved prediction.",
                "certificates": careerCertSuggestions["General Studies"]
            }]

        certResults = analyzeCertificates(certificateFiles or [])

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
