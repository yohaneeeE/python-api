# main.py or decisiontree_api.py
import os
import re
import io
import asyncio
from collections import OrderedDict
from typing import List, Optional, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pytesseract
from dotenv import load_dotenv

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
TESSERACT_PATH = os.getenv("TESSERACT_PATH")  # e.g. /usr/bin/tesseract
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------
# FastAPI App + CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates ")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your PHP site URL only
    allow_origins=["*"],  # or specify your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route for Render health check
@app.get("/")
def root():
    return {"message": "🚀 Career Prediction API is running successfully!"}

# ---------------------------
# Data Model + ML setup
# ---------------------------
class StudentInput(BaseModel):
    python: int
    sql: int
    java: int

model: Optional[RandomForestClassifier] = None
targetEncoder: Optional[LabelEncoder] = None

CS_CSV = "cs_students.csv"
if os.path.exists(CS_CSV):
    try:
        df = pd.read_csv(CS_CSV)
        features = ["Python", "SQL", "Java"]
        target = "Future Career"
        labelEncoders: Dict[str, LabelEncoder] = {}

        for col in features:
            if col in df.columns and df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                labelEncoders[col] = le

        if target in df.columns:
            targetEncoder = LabelEncoder()
            df[target] = targetEncoder.fit_transform(df[target].astype(str))
            X = df[features]
            y = df[target]

            model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            model.fit(X, y)
    except Exception as e:
        print("⚠️ Could not train model:", e)

# ---------------------------
# Utility Dictionaries
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "software", "coding", "development"],
    "databases": ["database", "sql", "dbms", "systems integration"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics"],
    "networking": ["networking", "networks", "cloud", "infrastructure"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "os", "architecture", "computer systems"]
}

bucketMap = {"programming": "Java", "databases": "SQL", "ai_ml": "Python"}

careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp", "Meta Frontend Dev"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer"],
    "Cloud Solutions Architect": ["AWS Solutions Architect", "Azure Fundamentals"],
    "Cybersecurity Specialist": ["CompTIA Security+", "Cisco CyberOps Associate"],
    "General Studies": ["Short IT courses to explore career interests"]
}

VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

# ---------------------------
# Helper Functions
# ---------------------------
def snap_to_valid_grade(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    return min(VALID_GRADES, key=lambda g: abs(g - val))

def grade_to_level(grade: Optional[float]) -> str:
    if grade is None:
        return "Unknown"
    if grade <= 1.75:
        return "Strong"
    elif grade <= 2.5:
        return "Average"
    else:
        return "Weak"

# ---------------------------
# OCR → Text → Grades Parser
# ---------------------------
def extractSubjectGrades(text: str):
    bucket_grades = {"Python": [3.0], "SQL": [3.0], "Java": [3.0]}
    # simplified placeholder parser
    return [], {}, {}, {}, bucket_grades

# ---------------------------
# Predict Function
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict):
    if model and targetEncoder:
        dfInput = pd.DataFrame([finalBuckets])
        proba = model.predict_proba(dfInput)[0]
        careers = [
            {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p) * 100, 2)}
            for i, p in enumerate(proba)
        ]
        return sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
    else:
        # fallback heuristic
        return [
            {"career": "Software Engineer", "confidence": 90.0},
            {"career": "Data Scientist", "confidence": 85.0}
        ]

# ---------------------------
# Certificates
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    if not certFiles:
        return [{"info": "No certificates uploaded"}]
    return [{"file": c.filename, "suggestions": ["Certificate added successfully."]} for c in certFiles]

# ---------------------------
# Routes
# ---------------------------
@app.post("/ocrPredict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes)).convert("RGB")
        text = await asyncio.to_thread(pytesseract.image_to_string, img)

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text.strip())
        careerOptions = predictCareerWithSuggestions(finalBuckets)
        certResults = analyzeCertificates(certificateFiles)

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}

