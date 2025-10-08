# filename: decisiontree_api.py
"""
Career Prediction API with OCR (TOR/COG + Certificates)
--------------------------------------------------------
- OCR reads grades from images (TOR or COG screenshots)
- Extracts subjects, grades, and computes skill buckets
- Predicts possible IT career paths
- Suggests certifications based on subject strengths
- Supports additional uploaded certificate files
- Safe fallback if training CSV is missing
- Optimized for Render deployment
"""

import os
import io
import re
import asyncio
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pytesseract

# Optional MySQL (keep for future use, not required for core API)
import mysql.connector

# Tesseract Path (change only if not detected automatically)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------------------
# FastAPI Initialization
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Helper Functions
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


# ---------------------------
# Training Model
# ---------------------------
def train_model():
    """Train RandomForest model or use fallback data."""
    if os.path.exists("cs_students.csv"):
        df = pd.read_csv("cs_students.csv")
    else:
        df = pd.DataFrame({
            "Python": [1.5, 2.0, 2.5, 3.0, 1.25],
            "SQL": [2.0, 2.5, 3.0, 2.75, 1.5],
            "Java": [2.0, 2.25, 3.0, 2.5, 1.75],
            "Future Career": [
                "Software Engineer", "Database Administrator",
                "Web Developer", "Data Scientist", "Software Engineer"
            ]
        })

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

    X, y = df[features], df[target]
    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X, y)
    return model, targetEncoder

model, targetEncoder = train_model()


# ---------------------------
# Keyword Maps
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "software", "coding", "development"],
    "databases": ["database", "sql", "information management", "dbms"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "networking": ["network", "infrastructure", "systems integration"],
    "systems": ["os", "architecture", "computer systems", "hardware"]
}

bucketMap = {"programming": "Java", "databases": "SQL", "ai_ml": "Python"}


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
# OCR Text Extraction
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured = []
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        clean = re.sub(r'[^A-Za-z0-9.\s]', ' ', line)
        tokens = clean.split()
        if len(tokens) < 2:
            continue

        # Detect numeric tokens (grades)
        numeric_tokens = [t for t in tokens if re.fullmatch(r"\d+(\.\d+)?", t)]
        if not numeric_tokens:
            continue

        try:
            gradeVal = float(numeric_tokens[0])
        except:
            continue

        subjDesc = " ".join(tokens[:-1]).strip().title()
        gradeVal = snap_to_valid_grade(gradeVal)
        level = grade_to_level(gradeVal)

        # classify into buckets
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket:
                    bucket_grades[assigned_bucket].append(gradeVal)

        mappedSkills[subjDesc] = level
        subjects_structured.append({
            "description": subjDesc,
            "grade": gradeVal,
            "level": level
        })

    # Compute average per bucket
    finalBuckets = {}
    for b in ("Python", "SQL", "Java"):
        grades = bucket_grades[b]
        finalBuckets[b] = round(sum(grades) / len(grades), 2) if grades else 3.0

    return subjects_structured, mappedSkills, finalBuckets


# ---------------------------
# Prediction Logic
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"]
    }])

    proba = model.predict_proba(dfInput)[0]
    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p) * 100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        strong_subjs = [s for s, lvl in mappedSkills.items() if lvl == "Strong"]
        weak_subjs = [s for s, lvl in mappedSkills.items() if lvl == "Weak"]

        suggestion = []
        if strong_subjs:
            suggestion.append(f"Strong subjects: {', '.join(strong_subjs)}.")
        if weak_subjs:
            suggestion.append(f"Needs improvement in: {', '.join(weak_subjs)}.")
        if not suggestion:
            suggestion.append("Focus on IT-related subjects for stronger alignment.")

        c["suggestion"] = " ".join(suggestion)
        c["certificates"] = careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])
    return careers


# ---------------------------
# Certificate File Analyzer
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    keywords = {
        "aws": "Great for Cloud and DevOps careers.",
        "ccna": "Boosts Networking and Systems Admin roles.",
        "python": "Strong foundation for AI/Data Science.",
        "sql": "Excellent for Database Administration.",
        "web": "Supports Web Development paths."
    }
    for cert in certFiles:
        name = cert.filename.lower()
        matched = [msg for key, msg in keywords.items() if key in name]
        if not matched:
            matched = [f"{cert.filename} adds extra credibility to your profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
async def root():
    return {"status": "Career Prediction API running ✅"}


@app.post("/ocrPredict")
async def ocrPredict(
    file: UploadFile = File(...),
    certificateFiles: Optional[List[UploadFile]] = File(None)
):
    try:
        # OCR from uploaded image
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        text = await asyncio.to_thread(pytesseract.image_to_string, img)

        # Extract & Predict
        subjects_structured, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, mappedSkills)

        # Certificates
        certResults = []
        if certificateFiles:
            certResults = analyzeCertificates(certificateFiles)
        else:
            certResults = [{"info": "No certificates uploaded"}]

        return {
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "General Studies",
            "careerOptions": careerOptions,
            "subjects": subjects_structured,
            "skills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }

    except Exception as e:
        return {"error": str(e)}
