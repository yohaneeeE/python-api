# decisiontree_api.py  (fixed)
import os
from dotenv import load_dotenv
import re
import io
from collections import OrderedDict
from typing import List, Optional, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pytesseract
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Load .env file (optional)
load_dotenv()

# Read variables
TESSERACT_PATH = os.getenv("TESSERACT_PATH")  # e.g. /usr/bin/tesseract
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

# If user provided a specific tesseract path, set it.
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
# Try to load structured dataset and train model (if available)
# ---------------------------
model: Optional[RandomForestClassifier] = None
targetEncoder: Optional[LabelEncoder] = None

CS_CSV = "cs_students.csv"
if os.path.exists(CS_CSV):
    try:
        df = pd.read_csv(CS_CSV)
        features = ["Python", "SQL", "Java"]
        target = "Future Career"

        data = df.copy()
        labelEncoders: Dict[str, LabelEncoder] = {}

        for col in features:
            if col in data.columns and data[col].dtype == "object":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                labelEncoders[col] = le

        if target in data.columns:
            targetEncoder = LabelEncoder()
            data[target] = targetEncoder.fit_transform(data[target].astype(str))

            X = data[features]
            y = data[target]

            model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            model.fit(X, y)
        else:
            model = None
            targetEncoder = None
    except Exception as e:
        # If training fails, continue with model=None
        print("Warning: could not train structured model:", e)
        model = None
        targetEncoder = None
else:
    print(f"Info: {CS_CSV} not found — structured model will be disabled.")

# ---------------------------
# FastAPI App with CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

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

def grade_to_level(grade: Optional[float]) -> str:
    if grade is None:
        return "Unknown"
    # lower numeric grade is better (1.00 best)
    if grade <= 1.75:
        return "Strong"
    elif grade <= 2.5:
        return "Average"
    else:
        return "Weak"

def snap_to_valid_grade(val: Optional[float]) -> Optional[float]:
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

def clean_subject_text(desc: str) -> Optional[str]:
    if not desc:
        return None
    d = desc.strip().lower()

    # Direct fixes (search for keys in TEXT_FIXES)
    for wrong, right in TEXT_FIXES.items():
        if wrong in d:
            d = d.replace(wrong, right.lower())

    # PE special cases
    if "pathfit" in d or "pen aire" in d or d.strip() == "pe":
        return "PE"

    # Elective special case: try to capture trailing digit
    if "elective" in d:
        m = re.search(r'(\d{1,2})', d)
        if m:
            num = m.group(1)[-1]
            return f"Elective {num}"
        return "Elective"

    # Purposive Communication
    if "purposive" in d and "communication" in d:
        return "Purposive Communication"

    # remove extra punctuation and multiple spaces and title-case
    d = re.sub(r'[^\w\s]', ' ', d)
    d = re.sub(r'\s{2,}', ' ', d).strip()
    if not d or len(d) < 2:
        return None
    return d.title()

# ---------------------------
# Helpers
# ---------------------------
def classify_subject(desc: str) -> str:
    if not desc:
        return "Unknown"
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

def normalize_code(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return re.sub(r'\s+', '', text.upper())

def _normalize_grade_str(num_str: Optional[str]) -> Optional[float]:
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if s == "":
        return None
    try:
        raw = float(s)
    except Exception:
        return None

    # Try some plausible conversions (raw could be 125 -> 1.25 or 125 -> 12.5)
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
    """
    Parse OCR text into structured subjects with grades and units.
    Returns: subjects_structured (list), rawSubjects (OrderedDict), normalizedText (dict), mappedSkills (dict), finalBuckets (dict)
    """
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

        # normalize whitespace and remove weird separators
        clean = re.sub(r'[\t\r\f\v]+', ' ', line)
        clean = re.sub(r'[^\w\.\-\s]', ' ', clean)   # keep letters, numbers, dot, dash, underscore
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 2:
            continue

        # --- detect course code (handles "IT 312", "IT312", "E10", "PCM 101") ---
        subjCode = None
        if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
            subjCode = f"{parts[0].upper()} {parts[1]}"
            parts = parts[2:]
        elif re.match(r'^[A-Z]{1,4}\d{1,3}$', parts[0].upper()):
            subjCode = parts[0].upper()
            parts = parts[1:]

        if not parts:
            continue

        # Remove trailing textual remark (e.g., "Passed")
        remarks = None
        if parts and parts[-1].isalpha():
            remarks = parts[-1]
            parts = parts[:-1]
            if not parts:
                continue

        # Collect numeric tokens with positions (to find grade and units)
        float_tokens = []
        for i, tok in enumerate(parts):
            token_clean = re.sub(r'[^0-9.]', '', tok)
            if token_clean and re.search(r'\d', token_clean):
                try:
                    rawf = float(token_clean)
                    float_tokens.append((i, token_clean, rawf))
                except Exception:
                    continue

        # Decide grade and units:
        gradeVal = None
        unitsVal = None
        grade_idx = None

        if len(float_tokens) >= 2:
            prev_idx, prev_tok, prev_raw = float_tokens[-2]
            last_idx, last_tok, last_raw = float_tokens[-1]
            grade_idx = prev_idx
            gradeVal = _normalize_grade_str(prev_tok)
            gradeVal = snap_to_valid_grade(gradeVal)
            try:
                unitsVal = float(last_raw)
            except Exception:
                unitsVal = None
        elif len(float_tokens) == 1:
            idx, tok, rawf = float_tokens[0]
            grade_idx = idx
            gradeVal = _normalize_grade_str(tok)
            gradeVal = snap_to_valid_grade(gradeVal)
            unitsVal = None
        else:
            # no numeric token → not a subject row
            continue

        # Build description tokens before grade_idx
        desc_tokens = parts[:grade_idx] if grade_idx is not None else parts[:]
        # If first token is just numeric code like '312', remove it
        if desc_tokens and re.fullmatch(r'\d+', desc_tokens[0]):
            desc_tokens = desc_tokens[1:]

        subjDesc_raw = " ".join(desc_tokens).strip()
        subjDesc_clean = clean_subject_text(subjDesc_raw)
        if not subjDesc_clean:
            subjDesc_clean = subjCode or "Unknown Subject"

        subjDesc = subjDesc_clean
        subjKey = subjDesc if not subjCode else f"{subjCode} {subjDesc}"
        category = classify_subject(subjDesc)

        # determine mapping to skill bucket (for ML only)
        assigned_bucket = None
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
                    bucket_grades.setdefault(assigned_bucket, []).append(gradeVal)
                break

        # store subject skill level (Weak/Average/Strong) for UI
        mappedSkills[subjDesc] = grade_to_level(gradeVal) if gradeVal is not None else "Unknown"

        # store
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

    # average bucket grades -> finalBuckets numeric values
    finalBuckets = {}
    for b, grades in bucket_grades.items():
        if grades:
            finalBuckets[b] = round(sum(grades) / len(grades), 2)
        else:
            finalBuckets[b] = 3.0

    # ensure keys exist
    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction with Smarter Suggestions
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    """
    Return top career options + suggestions.
    If structured model is not available, return generic suggestions based on bucket scores.
    """
    # If structured model available, use it to produce probabilities
    careers = []

    if model is not None and targetEncoder is not None:
        try:
            dfInput = pd.DataFrame([{
                "Python": finalBuckets["Python"],
                "SQL": finalBuckets["SQL"],
                "Java": finalBuckets["Java"],
            }])
            proba = model.predict_proba(dfInput)[0]
            careers = [
                {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p) * 100, 2)}
                for i, p in enumerate(proba)
            ]
            careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
        except Exception as e:
            # fallback if predict_proba fails
            print("Warning: model prediction failed:", e)
            model_local = None

    # If model not available or empty result, create heuristic career list
    if not careers:
        # Simple heuristic — rank careers by sum of relevant bucket strengths (lower grade better)
        bucket_score = lambda skill_list: sum([ (3.0 if finalBuckets.get(k) is None else finalBuckets[k]) for k in skill_list ])
        heuristics = []
        for career, skills in careerSkillMap.items():
            heuristics.append({"career": career, "score": bucket_score(skills)})
        # lower score implies better (since grade numeric lower is stronger)
        heuristics = sorted(heuristics, key=lambda x: x["score"])
        # map to career dict with synthetic confidence
        careers = []
        max_score = heuristics[-1]["score"] if heuristics else 1.0
        for h in heuristics[:3]:
            # invert score to confidence-like value
            conf = max(0.0, (max_score - h["score"]) / max_score) * 100
            careers.append({"career": h["career"], "confidence": round(conf, 2)})

    # Build suggestions and certificate recommendations
    for c in careers:
        career_name = c["career"]
        suggestions = []
        # Recommend based on finalBuckets numeric grades (1.0 best, 5.0 worst)
        for skill_name, grade in finalBuckets.items():
            if grade is None:
                continue
            # interpret grade thresholds
            if grade <= 1.75:
                # strong
                suggestions.append(f"You're strong in {skill_name}. Consider advanced projects and certifications to showcase this skill.")
            elif grade <= 2.5:
                suggestions.append(f"You're average in {skill_name}. Focus on practical exercises and small projects to improve.")
            else:
                suggestions.append(f"You should strengthen fundamentals in {skill_name} with targeted courses and practice.")
        # Career-specific hints
        if "Developer" in career_name or "Engineer" in career_name:
            suggestions.append("Work on small real-world projects and add them to a portfolio (GitHub).")
        if "Data" in career_name or "AI" in career_name:
            suggestions.append("Build Python data/ML mini-projects and document experiments.")
        if "Database" in career_name or "Architect" in career_name:
            suggestions.append("Practice SQL and database design; consider cloud fundamentals for deployment.")
        if not suggestions:
            suggestions.append("Focus on improving relevant subjects and obtain certificates where possible.")

        c["suggestion"] = " ".join(suggestions[:8])
        c["certificates"] = careerCertSuggestions.get(career_name, ["Consider general IT certifications."])

    return careers

# ---------------------------
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns well with AI/ML and Data Scientist roles.",
        "webdev": "Web Development certificate enhances your frontend/backend developer profile.",
        "python": "Python certification supports Data Science, AI, and Software Engineering careers."
    }
    if not certFiles:
        return [{"info": "No certificates uploaded"}]
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
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes)).convert("RGB")

        # run pytesseract in background thread to avoid blocking event loop
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

        certResults = analyzeCertificates(certificateFiles)

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
        # return error message for debugging — consider logging in production
        return {"error": str(e)}
