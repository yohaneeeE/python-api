# main.py — Career Prediction API (OCR + PDF/DOCX/TXT + Certificates)
import os
import io
import re
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
import fitz  # PyMuPDF
from docx import Document

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
TESSERACT_PATH = os.getenv("TESSERACT_PATH")  # e.g., /usr/bin/tesseract
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="Career Prediction API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Career Prediction API is running!"}

# ---------------------------
# Model and Data Loading
# ---------------------------
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
            print("✅ Structured model trained from cs_students.csv")
        else:
            print("ℹ️ cs_students.csv present but no target column — model disabled")
    except Exception as e:
        print("⚠️ Could not train model:", e)
else:
    print("ℹ️ cs_students.csv not found — using heuristics only")

# ---------------------------
# Subject Groups & Buckets
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

TEXT_FIXES = {
    "tras beaives bstaegt": "Elective 5",
    "wage system integration and rotate 2 es": "System Integration and Architecture 2",
    "aot sten ainsaton and marenance": "System Administration and Maintenance",
    "capa capstone pret and research 2 es": "Capstone Project and Research 2",
    "mathnats nthe modem oa es": "Mathematics in the Modern World",
    "advan database systems": "Advance Database Systems"
}

REMOVE_LIST = [
    "stone project ad reset",
    "student",
    "report of grades",
    "unknown subject",
    "category", "communications", "class", "united", "student no", "fullname",
]

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

def _normalize_grade_str(num):
    try:
        raw = float(num)
    except Exception:
        return None
    candidates = [raw, raw / 10.0, raw / 100.0]
    valid = [c for c in candidates if 1.0 <= c <= 5.0]
    if valid:
        chosen = min(valid, key=lambda x: abs(x - 2.5))
        return round(chosen, 2)
    if 0.0 < raw <= 5.0:
        return round(raw, 2)
    return None

def normalize_subject(subj: str) -> str:
    s = subj.lower().strip()
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)
    s = re.sub(r'[^\w\s]', ' ', s)
    for bad in REMOVE_LIST:
        if bad in s:
            return None
    return s.title() if s else None

def simple_bucket(subj: str) -> Optional[str]:
    s = subj.lower()
    for bucket, keywords in subjectGroups.items():
        for kw in keywords:
            if kw in s:
                return bucketMap.get(bucket)
    return None

# ---------------------------
# Extract Subjects & Grades
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured = []
    rawSubjects = OrderedDict()
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for raw_line in lines:
        line = raw_line.strip()
        if not line or any(kw in line.lower() for kw in ["student", "report", "university", "republic"]):
            continue

        tokens = re.split(r'\s+', line)
        nums = [re.sub(r'[^0-9.]', '', t) for t in tokens if re.search(r'\d', t)]
        grade_val = None
        if nums:
            try:
                cand = float(nums[-1])
                grade_val = snap_to_valid_grade(_normalize_grade_str(cand))
            except:
                grade_val = None

        subj_tokens = [t for t in tokens if not re.fullmatch(r'[^A-Za-z]*\d+[^A-Za-z]*', t)]
        subj_name = " ".join([t for t in subj_tokens if not re.search(r'\d', t)])
        subj_name = normalize_subject(subj_name) or "Unknown Subject"

        # Add to bucket grades
        bucket = simple_bucket(subj_name)
        if bucket:
            bucket_grades[bucket].append(grade_val if grade_val is not None else 3.0)

        mappedSkills[subj_name] = grade_to_level(grade_val)
        subjects_structured.append({"description": subj_name, "grade": grade_val, "raw_line": line})
        rawSubjects[subj_name] = grade_val

    finalBuckets = {}
    for k in ("Python", "SQL", "Java"):
        vals = [v for v in bucket_grades.get(k, []) if isinstance(v, (int, float))]
        finalBuckets[k] = round(sum(vals) / len(vals), 2) if vals else 3.0

    return subjects_structured, rawSubjects, mappedSkills, finalBuckets

# ---------------------------
# Predict Career & Suggestions
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict):
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
                {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100,2)}
                for i, p in enumerate(proba)
            ]
            careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
        except:
            careers = []

    if not careers:
        heuristics = [
            {"career": "Software Engineer", "score": finalBuckets.get("Java", 3.0)},
            {"career": "Web Developer", "score": (finalBuckets.get("Java",3.0)+finalBuckets.get("SQL",3.0))/2},
            {"career": "Data Scientist", "score": finalBuckets.get("Python", 3.0)}
        ]
        max_score = max(h["score"] for h in heuristics)
        careers = []
        for h in heuristics:
            conf = max(0.0,(max_score-h["score"])/max_score*100) if max_score else 50.0
            careers.append({"career": h["career"], "confidence": round(conf,2)})

    # Add suggestions and certificates
    for c in careers:
        name = c["career"]
        suggestions = []
        for k, val in finalBuckets.items():
            if val <= 1.75:
                suggestions.append(f"Strong in {k} — advanced projects or certifications recommended.")
            elif val <= 2.5:
                suggestions.append(f"Average in {k} — practice and small projects recommended.")
            else:
                suggestions.append(f"Weak in {k} — foundational courses and practice needed.")
        c["suggestion"] = " ".join(suggestions)
        c["certificates"] = careerCertSuggestions.get(name, ["Consider general IT certifications."])

    return careers

# ---------------------------
# Certificates Analysis
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    if not certFiles:
        return [{"info":"No certificates uploaded"}]

    suggestionsMap = {
        "aws":"AWS certification boosts Cloud/DevOps roles",
        "ccna":"CCNA boosts Networking/System admin roles",
        "datascience":"Data Science certificate aligns with AI/ML",
        "webdev":"Web Dev cert enhances frontend/backend profile",
        "python":"Python cert supports Data Science, AI, Software Eng"
    }

    results = []
    for cert in certFiles:
        certName = cert.filename.lower()
        matched = [msg for key, msg in suggestionsMap.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds career value."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Extract text from files
# ---------------------------
def extract_text_from_file(upload: UploadFile) -> str:
    filename = upload.filename.lower()
    file_bytes = upload.file.read()

    if filename.endswith(".pdf"):
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return pytesseract.image_to_string(img)

# ---------------------------
# Routes
# ---------------------------
@app.post("/filePredict")
async def filePredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        text = await asyncio.to_thread(extract_text_from_file, file)
        subjects_structured, rawSubjects, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets)
        certResults = analyzeCertificates(certificateFiles)

        return {
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "General Studies",
            "careerOptions": careerOptions,
            "finalBuckets": finalBuckets,
            "subjects_structured": subjects_structured,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}

# Backward compatibility route
@app.post("/ocrPredict")
async def ocrPredict_redirect(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    return await filePredict(file, certificateFiles)
