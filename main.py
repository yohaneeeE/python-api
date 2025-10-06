# main_fixed_reliable.py — Reliable Career Prediction API with Certificates 🚀
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
import fitz  # PyMuPDF for PDFs
from docx import Document
from rapidfuzz import fuzz

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
TESSERACT_PATH = os.getenv("TESSERACT_PATH")
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------------
# FastAPI App + CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (Reliable)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🚀 Career Prediction API is running reliably!"}

# ---------------------------
# ML Model Setup
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
            print("✅ Structured model trained from cs_students.csv")
    except Exception as e:
        print("⚠️ Could not train model:", e)

# ---------------------------
# Dictionaries / Heuristics
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
    "stone project ad reset", "student", "report of grades", "unknown subject",
    "category", "communications", "class", "united", "student no", "fullname"
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

def normalize_subject(subj: str) -> Optional[str]:
    s = subj.lower().strip()
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)
    s = re.sub(r'[^\w\s]', ' ', s)
    for bad in REMOVE_LIST:
        if bad in s:
            return None
    return s.title() if s else None

def fuzzy_bucket(subj: str):
    """Return main bucket key from subject string."""
    subj = subj.lower()
    for bucket, keywords in subjectGroups.items():
        for kw in keywords:
            if fuzz.partial_ratio(kw, subj) > 80:
                return bucketMap.get(bucket)
    return None

# ---------------------------
# OCR / Text → Grades Parser
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured = []
    rawSubjects = OrderedDict()
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        if any(kw in line.lower() for kw in ["student", "report", "university", "republic"]):
            continue

        # Extract number from line
        nums = re.findall(r"\d+(?:\.\d+)?", line.replace(",", "."))
        grade_val = None
        if nums:
            try:
                cand = float(nums[-1])
                if cand > 5.0:  # normalize 0-100 scale to 1-5
                    cand = 5.0 - (cand / 100) * 4
                grade_val = snap_to_valid_grade(cand)
            except:
                grade_val = None

        # Extract subject name
        subj_tokens = re.split(r'\d+', line)
        subj_name = " ".join(subj_tokens).strip()
        subj_name = normalize_subject(subj_name) or "Unknown Subject"

        # Map to bucket
        bucket = fuzzy_bucket(subj_name)
        if bucket:
            bucket_grades[bucket].append(grade_val if grade_val is not None else 3.0)

        mappedSkills[subj_name] = grade_to_level(grade_val)
        subjects_structured.append({"description": subj_name, "grade": grade_val, "raw_line": line})
        rawSubjects[subj_name] = grade_val

    finalBuckets = {}
    for k in ["Python", "SQL", "Java"]:
        vals = [v for v in bucket_grades[k] if isinstance(v, (int, float))]
        finalBuckets[k] = round(sum(vals)/len(vals), 2) if vals else 3.0

    return subjects_structured, rawSubjects, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict):
    careers = []
    if model is not None and targetEncoder is not None:
        try:
            dfInput = pd.DataFrame([finalBuckets])
            proba = model.predict_proba(dfInput)[0]
            careers = [
                {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)}
                for i, p in enumerate(proba)
            ]
            careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
        except:
            careers = []

    if not careers:
        # Weighted heuristic
        career_weights = {
            "Software Engineer": 0.6*finalBuckets["Java"] + 0.4*finalBuckets["Python"],
            "Web Developer": 0.5*finalBuckets["Java"] + 0.5*finalBuckets["SQL"],
            "Data Scientist": 0.7*finalBuckets["Python"] + 0.3*finalBuckets["SQL"]
        }
        min_score = min(career_weights.values())
        careers = []
        for career, score in career_weights.items():
            conf = max(0.0, (5.0 - score)/5.0) * 100
            careers.append({"career": career, "confidence": round(conf, 2)})

    for c in careers:
        name = c["career"]
        suggestions = []
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
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    if not certFiles:
        return [{"info": "No certificates uploaded"}]

    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns well with AI/ML and Data Scientist roles.",
        "webdev": "Web Development certificate enhances your frontend/backend developer profile.",
        "python": "Python certification supports Data Science, AI, and Software Engineering careers."
    }

    results = []
    for cert in certFiles:
        certName = cert.filename.lower().replace("_", " ").replace("-", " ")
        matched = [msg for key, msg in certificateSuggestions.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds value to your career profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# File Text Extraction
# ---------------------------
def extract_text_from_file(upload: UploadFile) -> str:
    filename = upload.filename.lower()
    file_bytes = upload.file.read()

    if filename.endswith(".pdf"):
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text("text")
                if not page_text.strip():
                    # fallback OCR
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
        return text

    elif filename.endswith((".docx", ".doc")):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        # fallback: image
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
            "mappedSkills": mappedSkills,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ocrPredict")
async def ocrPredict_redirect(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    return await filePredict(file, certificateFiles)
