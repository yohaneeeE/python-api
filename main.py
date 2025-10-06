# main.py — cleaned & fixed
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
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route for Render health check
@app.get("/")
def root():
    return {"message": "🚀 Career Prediction API is running successfully!"}

# ---------------------------
# Data Model + ML setup (optional)
# ---------------------------
class StudentInput(BaseModel):
    python: int
    sql: int
    java: int

model: Optional[RandomForestClassifier] = None
targetEncoder: Optional[LabelEncoder] = None

# If you have a cs_students.csv and want to train at startup, enable this:
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
            print("✅ Structured model trained from cs_students.csv")
        else:
            print("ℹ️ cs_students.csv present but no target column — model disabled")
    except Exception as e:
        print("⚠️ Could not train model:", e)
else:
    print("ℹ️ cs_students.csv not found — structured model disabled (heuristic will be used)")

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
# (Simple robust placeholder: returns numeric finalBuckets)
# ---------------------------
def extractSubjectGrades(text: str):
    """
    Parse OCR text (rough) and return:
    (subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets)
    finalBuckets is a dict with numeric values (1.0 - 5.0)
    """
    # Very tolerant parsing: attempt to find subject lines with grades
    subjects_structured = []
    rawSubjects = OrderedDict()
    normalizedText = {}
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    if not text:
        # Default fallback
        finalBuckets = {"Python": 3.0, "SQL": 3.0, "Java": 3.0}
        return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # ignore obviously irrelevant lines
        low = line.lower()
        if any(kw in low for kw in ["student", "report", "university", "republic"]):
            continue

        # crude extraction: find last numeric token (grade) in line
        tokens = re.split(r'\s+', line)
        nums = [re.sub(r'[^0-9.]', '', t) for t in tokens if re.search(r'\d', t)]
        grade_val = None
        if nums:
            # try last numeric token as grade
            try:
                cand = float(nums[-1])
                cand_norm = snap_to_valid_grade(_normalize_grade_str(cand))
                grade_val = cand_norm
            except Exception:
                grade_val = None

        # subject is the non-numeric prefix
        subj_tokens = [t for t in tokens if not re.fullmatch(r'[^A-Za-z]*\d+[^A-Za-z]*', t)]
        subj_name = " ".join([t for t in subj_tokens if not re.search(r'\d', t)])
        subj_name = subj_name.strip() or "Unknown Subject"
        subj_name = re.sub(r'[^\w\s]', ' ', subj_name).strip().title()

        # map simple keyword to bucket
        lower = subj_name.lower()
        if "python" in lower:
            bucket_grades["Python"].append(grade_val if grade_val is not None else 3.0)
        if "sql" in lower or "database" in lower:
            bucket_grades["SQL"].append(grade_val if grade_val is not None else 3.0)
        if "java" in lower:
            bucket_grades["Java"].append(grade_val if grade_val is not None else 3.0)

        mappedSkills[subj_name] = grade_to_level(grade_val)
        subjects_structured.append({
            "description": subj_name,
            "grade": grade_val,
            "raw_line": line
        })
        rawSubjects[subj_name] = grade_val
        normalizedText[subj_name] = subj_name

    # compute averages -> numeric finalBuckets
    finalBuckets = {}
    for k in ("Python", "SQL", "Java"):
        vals = [v for v in bucket_grades.get(k, []) if isinstance(v, (int, float))]
        if vals:
            finalBuckets[k] = round(sum(vals) / len(vals), 2)
        else:
            finalBuckets[k] = 3.0

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# helper used above
def _normalize_grade_str(num):
    try:
        raw = float(num)
    except Exception:
        return None
    # attempt sensible conversions
    candidates = [raw, raw / 10.0, raw / 100.0]
    valid = [c for c in candidates if 1.0 <= c <= 5.0]
    if valid:
        chosen = min(valid, key=lambda x: abs(x - 2.5))
        return round(chosen, 2)
    if 0.0 < raw <= 5.0:
        return round(raw, 2)
    return None

# ---------------------------
# Predict Function
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict):
    """
    finalBuckets: {"Python":1.5, "SQL":3.0, "Java":2.0}
    Returns list of career dicts (career, confidence, suggestion, certificates)
    """
    careers = []
    if model is not None and targetEncoder is not None:
        try:
            dfInput = pd.DataFrame([{
                "Python": float(finalBuckets.get("Python", 3.0)),
                "SQL": float(finalBuckets.get("SQL", 3.0)),
                "Java": float(finalBuckets.get("Java", 3.0)),
            }])
            proba = model.predict_proba(dfInput)[0]
            careers = [
                {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p) * 100, 2)}
                for i, p in enumerate(proba)
            ]
            careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
        except Exception as e:
            print("Warning: structured model prediction failed:", e)
            careers = []

    # fallback heuristic if model not available or failed
    if not careers:
        # simple heuristic ranks careers by sum of relevant buckets (lower is better)
        careerSkillMap = {
            "Software Engineer": ["programming", "databases"],
            "Data Scientist": ["ai_ml", "programming", "databases"],
            "Cloud Solutions Architect": ["networking", "databases", "programming"],
            "Web Developer": ["webdev", "programming", "databases"],
        }
        heuristics = []
        for career, skills in careerSkillMap.items():
            # map skill names to buckets
            score = 0.0
            for s in skills:
                if s == "programming":
                    score += finalBuckets.get("Java", 3.0)
                elif s == "databases":
                    score += finalBuckets.get("SQL", 3.0)
                elif s == "ai_ml":
                    score += finalBuckets.get("Python", 3.0)
                else:
                    score += 3.0
            heuristics.append({"career": career, "score": score})
        heuristics = sorted(heuristics, key=lambda x: x["score"])
        max_score = heuristics[-1]["score"] if heuristics else 1.0
        careers = []
        for h in heuristics[:3]:
            conf = max(0.0, (max_score - h["score"]) / max_score) * 100 if max_score else 50.0
            careers.append({"career": h["career"], "confidence": round(conf, 2)})

    # add suggestions + certificate recs
    for c in careers:
        name = c["career"]
        suggestions = []
        # generic suggestions based on buckets
        for k, val in finalBuckets.items():
            if val <= 1.75:
                suggestions.append(f"Strong in {k} — consider advanced projects or certifications.")
            elif val <= 2.5:
                suggestions.append(f"Average in {k} — do hands-on practice and small projects.")
            else:
                suggestions.append(f"Weak in {k} — take foundational courses and practice more.")
        c["suggestion"] = " ".join(suggestions[:6])
        c["certificates"] = careerCertSuggestions.get(name, ["Consider general IT certifications."])

    return careers

# ---------------------------
# Certificates
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    if not certFiles:
        return [{"info": "No certificates uploaded"}]
    results = []
    for c in certFiles:
        results.append({"file": c.filename, "suggestions": ["Certificate received"]})
    return results

# ---------------------------
# Routes
# ---------------------------
@app.post("/ocrPredict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes)).convert("RGB")

        # run pytesseract in thread to avoid blocking
        text = await asyncio.to_thread(pytesseract.image_to_string, img)
        # debug: print first 120 chars of extracted text
        print("OCR Extracted (preview):", (text or "")[:120])

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades((text or "").strip())
        careerOptions = predictCareerWithSuggestions(finalBuckets)
        certResults = analyzeCertificates(certificateFiles)

        return {
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "General Studies",
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        # don't expose internal stack in production — here we return for debugging
        print("Error in /ocrPredict:", e)
        return {"error": str(e)}
