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

# Configure Tesseract executable path if provided; otherwise leave default (Render typically has /usr/bin/tesseract)
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Initialize Gemini client if available and key present
genai_client = None
if _HAS_GENAI and GEMINI_API_KEY:
    genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# Input Schema (not used directly in endpoint but kept for reference)
# ---------------------------
class StudentInput(BaseModel):
    python: int
    sql: int
    java: int

# ---------------------------
# Train Structured Data Model
# ---------------------------
# NOTE: Ensure bsit_students.csv is present in working directory on Render
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
app = FastAPI(title="Career Prediction API (COG + Certificates 🚀)")

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
# OCR Fixes
# ---------------------------
VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]


# Known OCR misreads to fix (add more as you discover them)
TEXT_FIXES = {
    "lective": "Elective",
    "hective": "Elective",
    "pen aire": "PE",
    "pathfit": "PE",
    "grmmunication": "Communication",
    "cobege": "College",
    "tras beaives bstaegt": "Elective 5",
    "wage system integration and rotate 2 es": "System Integration and Architecture 2",
    "aot sten ainsaton and marenance": "System Administration and Maintenance",
    "capa capstone pret and research 2 es": "Capstone Project and Research 2",
    "mathnats nthe modem oa es": "Mathematics in the Modern World",
    "advan database systems": "Advance Database Systems",
    "capstone project and research 1 spparont cepsre": "Capstone Project and Research 1",
    "web systems and technologies 2 soxtsrowebsystemsbtechroiogies": "Web Systems and Technologies 2",
    "rane foreign languoge 2": "Foreign Language 2",
    "Networking 1 2": "Networking 2",
    "panik at lpunen 255": "Panitikan at Lipunan",
    "lifeand works of rizal": "Life and Works of Rizal",
    "conder cote soman cagesuntcanes": "Data Structure and Algorithms",
    "negate proganmingandteomoege": "Integrative Programming and Technologies 1",
    "foreign langage": "Foreign Language",
    "hunan computer terface": "Human Computer Interface",
    "infomation anogerent": "Information Management",
    "toot": "Object-Oriented Programming 1",
    "lective": "elective 4",
    "hective": "elective",
    "pen aire": "pe",
    "pathfit": "pe",
    "grmmunication": "communication",
    "cobege": "college",
    "phystal edeation": "physical education",
    "inveductonto computing ws": "introduction to computing",
    "inveductonto computing": "introduction to computing",
    "rio harare system ard saving": "hardware system and servicing",
    "hardware system ard saving": "hardware system and servicing",
    "camper prararining": "computer programming",
    "camper prararin": "computer programming",
    "readhgs npop history": "readings in philippine history",
    "scene technology and sooty": "science technology and society",
    "scene technology and sooty": "science technology and society",
    "atari": "art appreciation",
    "natonl sncetrhing pega": "national service training program",
    "diserete sturt for it": "discrete structures for it",
    "networking": "networking 1",
    "understanding the se": "understanding the self",
    "understanding The sef": "understanding the self",
    "Understanding The Selff": "understanding the self",
    "purposve communication": "purposive communication",
    "mathematics in the modem world so": "mathematics in the modern world"

}

# Things that should NEVER appear (noise / random OCR junk)
REMOVE_LIST = [
    "stone project ad reset",
    "catege ommuniatons crass uniteamed",
    "student",
    "acaserie eer agpy gna",
    "unknown subject",
    "category", "communications", "class", "united", "student no", "fullname",
    "report of grades", "republic", "city of", "wps", "office"
]

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

def clean_subject_text(desc: str) -> str:
    d = (desc or "").lower()

    # --- Fix PE (PE / PathFit) ---
    if "pen aire" in d or "pathfit" in d:
        return "PE"

    # --- Fix Elective with numbers ---
    if "lective" in d or "hective" in d:
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

def normalize_code(text: str) -> Optional[str]:
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
    # mappedSkills now stores a dict for each subject: {'level': 'Strong', 'bucket': 'Python' or None}
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
                    bucket_grades.setdefault(assigned_bucket, []).append(gradeVal)
                break

        # store both level and bucket to make later lookup reliable
        mappedSkills[subjDesc] = {
            "level": grade_to_level(gradeVal) if gradeVal is not None else "Unknown",
            "bucket": assigned_bucket  # may be None if no bucket matched
        }


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
        "Python": finalBuckets.get("Python", 3.0),
        "SQL": finalBuckets.get("SQL", 3.0),
        "Java": finalBuckets.get("Java", 3.0),
    }])

    proba = model.predict_proba(dfInput)[0]
    # model.classes_ contains encoded labels (integers), so we enumerate those indices
    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        suggestions = []
        # find subjects that map to each skill bucket (mappedSkills entries with bucket == skill)
        for skill, grade in finalBuckets.items():
            if grade is None:
                continue
            subjMatches = [subj for subj, meta in mappedSkills.items() if meta.get("bucket") == skill]
            # Because grade is numeric and lower is better (1.0 best), treat >=2.75 as weak
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
        certName = (cert.filename or "").lower()
        matched = [msg for key, msg in certificateSuggestions.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds additional value to your career profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Gemini Enhancement (optional)
# ---------------------------
async def enhance_with_gemini(careerOptions,):
    if not genai_client:
        return None
    try:
        prompt = f"""
You are a helpful career advisor for BSIT students.
Given this student's analysis:
- Top career predictions: {careerOptions}

Provide 3 concise, practical, and personalized suggestions (mention technologies, projects, or certs).
Keep output under 2-3 sentences and use a motivational friendly tone and make no redundancy in your answers for every bullet.
"""
        response = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip() if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"(Gemini unavailable: {e})"

# ---------------------------
# Routes
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        imageBytes = await file.read()
        try:
            img = Image.open(io.BytesIO(imageBytes))
        except UnidentifiedImageError:
            return {"error": "Uploaded file is not a supported image."}
        # For safety, convert to RGB (some PIL formats are paletted)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Run OCR in a thread (blocking)
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

        # Gemini enhancement (optional)
        gemini_enhancement = await enhance_with_gemini(careerOptions,) if genai_client else None

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "geminiSuggestions": gemini_enhancement,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        # Don't leak huge stack traces to users — return error message
        return {"error": str(e)}

# Basic health route
@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API running."}

