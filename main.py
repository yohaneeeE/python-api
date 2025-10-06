"""
main.py — Combined, fixed Career Prediction API
Place this at the repo root. Designed to be run by:
    uvicorn main:app --host 0.0.0.0 --port 8000

Features:
- Accepts PDF/DOCX/TXT/PNG/JPG uploads
- OCR for images (pytesseract)
- Text extraction from PDF/DOCX/TXT
- Subject/grade normalization + bucket mapping (Python/SQL/Java)
- Structured RandomForest prediction (from cs_students.csv)
- TF-IDF text-based fallback predictor
- Certificate filename analysis
- Optional MySQL saving controlled by env vars
"""

import os
import io
import re
import json
import math
import traceback
from collections import OrderedDict
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ML & data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# OCR & file parsing
from PIL import Image
import pytesseract
import docx
from PyPDF2 import PdfReader

# Optional MySQL saving
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except Exception:
    MYSQL_AVAILABLE = False

# ---------------------------
# Configuration
# ---------------------------
CSV_PATH = os.getenv("CSV_PATH", "app/models/cs_students.csv")  # or "models/cs_students.csv"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# MySQL envs (optional)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Career Prediction API (OCR + TOR parsing + Certs)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------------------
# Utilities: grade normalization & text fixes
# ---------------------------
VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

def snap_to_valid_grade(val: Optional[float]):
    if val is None:
        return None
    # find closest valid
    return min(VALID_GRADES, key=lambda g: abs(g - val))

def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if s == "":
        return None
    try:
        raw = float(s)
    except:
        return None

    # Accept numbers like 125 -> 1.25, 12 -> 1.2 etc.
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

def grade_to_level(grade: Optional[float]) -> str:
    if grade is None:
        return "Unknown"
    if grade <= 1.75:
        return "Strong"
    elif grade <= 2.5:
        return "Average"
    else:
        return "Weak"

# Known OCR wrong mappings (extended from your original)
TEXT_FIXES = {
    "advan database systems": "Advance Database Systems",
    "camper prararining": "Computer Programming",
    "purposve communication": "Purposive Communication",
    "mathematics in the modem world": "Mathematics in the Modern World",
    "aot sten ainsaton and marenance": "System Integration and Maintenance",
    # add more cases as you see them
}

REMOVE_LIST = [
    "student", "report of grades", "republic", "city of", "wps", "fullname",
    "student no", "academic year", "date printed", "gwa"
]

def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    raw = desc or ""
    s = raw.lower().strip()
    s = re.sub(r'[_]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    if not s:
        return None

    for bad in REMOVE_LIST:
        if bad in s:
            return None

    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct.lower())

    # Elective detection
    if "elective" in s:
        m = re.search(r'\b(\d{1,2})\b', s)
        num = m.group(1) if m else None
        return f"Elective {num}" if num else "Elective"

    # PE
    if s == "pe" or s.startswith("pe ") or "pathfit" in s:
        if code:
            mm = re.search(r'(\d{1,3})', code)
            if mm:
                return f"PE {mm.group(1)}"
        return "PE"

    s = s.strip()
    if len(s) < 3:
        return None

    return s.title()

# ---------------------------
# Subject groups + bucket map (from your originals)
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "object oriented", "software", "coding", "development", "elective"],
    "databases": ["database", "sql", "dbms", "information systems", "data management"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics"],
    "networking": ["networking", "networks", "cloud", "infrastructure"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "os", "architecture", "computer systems"]
}

bucketMap = {
    "programming": "Java",
    "databases": "SQL",
    "ai_ml": "Python"
}

# subject -> certificates mapping (shortened for brevity)
subjectCertMap = {
    "computer programming": ["PCAP – Python Certified Associate", "Oracle Certified Java Programmer"],
    "information management": ["Oracle Database SQL Associate", "Microsoft SQL Server Certification"],
    "web systems and technologies": ["FreeCodeCamp Responsive Web Design", "Meta Front-End Developer Certificate"],
    "networking 1": ["Cisco CCNA", "CompTIA Network+"],
    "data structure and algorithms": ["HackerRank Skills Certification (DSA)"],
    "introduction to computing": ["IC3 Digital Literacy Certification", "CompTIA IT Fundamentals+"],
    "hardware system and servicing": ["CompTIA A+", "PC Hardware Technician Certification"],
}

careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp Responsive Web Design", "Meta Frontend Dev"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer Certificate"],
    "Database Administrator": ["Oracle SQL Associate"],
    "Cloud Solutions Architect": ["AWS Solutions Architect"],
    "Cybersecurity Specialist": ["CompTIA Security+", "Cisco CyberOps Associate"],
    "General Studies": ["Short IT courses to explore career interests"]
}

# ---------------------------
# OCR & text extraction helpers
# ---------------------------
def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception:
        return ""

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for p in reader.pages:
            try:
                page_text = p.extract_text() or ""
            except Exception:
                page_text = ""
            text += page_text + "\n"
        return text
    except Exception:
        return ""

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(docx_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text_from_txt_bytes(txt_bytes: bytes) -> str:
    try:
        return txt_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return str(txt_bytes)

def extract_text_from_upload(file: UploadFile) -> str:
    """Detect file type by filename and return extracted text."""
    filename = (file.filename or "").lower()
    contents = file.file.read()  # file.file is a SpooledTemporaryFile
    # Reset stream position for potential reuse
    try:
        file.file.seek(0)
    except Exception:
        pass

    if filename.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image_bytes(contents)
    if filename.endswith(".pdf"):
        return extract_text_from_pdf_bytes(contents)
    if filename.endswith(".docx"):
        return extract_text_from_docx_bytes(contents)
    if filename.endswith(".txt"):
        return extract_text_from_txt_bytes(contents)

    # fallback: try treating as text
    try:
        return contents.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------------------------
# Subject & grade extraction (robust from original)
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
        # skip header junk
        if any(kw in low for kw in ["fullname", "student", "report of grades", "academic year"]):
            continue

        # normalize whitespace & remove stray punctuation (but keep dots/digits)
        clean = re.sub(r'[\t\r\f\v]+', ' ', line)
        clean = re.sub(r'[^\w\.\-\s]', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 1:
            continue

        # detect course code
        subjCode = None
        if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
            subjCode = f"{parts[0].upper()} {parts[1]}"
            parts = parts[2:]
        elif re.match(r'^[A-Z]{1,4}\d{1,3}$', parts[0].upper()):
            subjCode = parts[0].upper()
            parts = parts[1:]

        if not parts:
            continue

        # remove trailing alpha remark
        remarks = None
        if parts and parts[-1].isalpha():
            remarks = parts[-1]
            parts = parts[:-1]
            if not parts:
                continue

        # collect numeric tokens to find grade and units
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
            # not a subject line - skip
            continue

        # build desc tokens before grade_idx
        desc_tokens = parts[:grade_idx] if grade_idx is not None else parts[:]
        if desc_tokens and re.fullmatch(r'\d+', desc_tokens[0]):
            desc_tokens = desc_tokens[1:]
        subjDesc_raw = " ".join(desc_tokens).strip()
        if not subjDesc_raw:
            subjDesc_raw = subjCode or "Unknown Subject"

        subjDesc_clean = normalize_subject(subjCode, subjDesc_raw)
        if subjDesc_clean is None:
            continue

        subjDesc = subjDesc_clean
        subjKey = subjDesc

        category = "Major Subject" if "elective" in subjDesc.lower() else (
            "IT Subject" if any(k in subjDesc.lower() for k in [
                "programming", "database", "data", "system", "integration", "architecture",
                "software", "network", "computing", "information", "security", "java",
                "python", "sql", "web", "algorithm"
            ]) else "Minor Subject"
        )

        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
                    bucket_grades[assigned_bucket].append(gradeVal)
                break

        mappedSkills[subjDesc] = grade_to_level(gradeVal) if gradeVal is not None else "Unknown"

        subjects_structured.append({
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
# Model training / load
# ---------------------------
# We'll try to load structured CSV model; if missing, we fallback to text-based TF-IDF classifier
structured_model = None
targetEncoder = None
labelEncoders = {}
text_model = None
vectorizer = None

def train_or_load_models():
    global structured_model, targetEncoder, labelEncoders, text_model, vectorizer
    # Text-based example docs (fallback)
    docs = [
        "Math Physics Programming Data Structures",
        "Networking Security Database Computer Systems",
        "UI Design Multimedia Graphic Design HTML CSS"
    ]
    text_labels = ["Software Engineer", "Network Specialist", "UI/UX Designer"]

    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(docs)
    text_model = RandomForestClassifier(n_estimators=50, random_state=42)
    text_model.fit(X_text, text_labels)

    # Attempt to train structured model from CSV
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            features = [c for c in ["GPA", "Python", "SQL", "Java", "Interested Domain"] if c in df.columns]
            target = "Future Career" if "Future Career" in df.columns else (df.columns[-1] if len(df.columns) > 0 else None)
            if target is None:
                return

            data = df.copy()
            for col in features:
                if data[col].dtype == "object":
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    labelEncoders[col] = le
            targetEncoder = LabelEncoder()
            data[target] = targetEncoder.fit_transform(data[target].astype(str))

            X = data[features]
            y = data[target]
            structured_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            structured_model.fit(X.fillna(3.0), y)
        except Exception:
            # If loading/training fails, structured_model remains None and we fall back to text model
            structured_model = None

# train at startup
train_or_load_models()

# ---------------------------
# Career prediction with suggestions
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    # structured prediction path (if model available)
    careers = []
    try:
        if structured_model is not None:
            # Build input row using keys that model expects: try to match Python/SQL/Java or other features
            df_in = pd.DataFrame([{k: finalBuckets.get(k, 3.0) for k in ["Python", "SQL", "Java"]}])
            proba = structured_model.predict_proba(df_in)[0]
            # Create career list using targetEncoder if available (else index names)
            if targetEncoder is not None:
                classes = targetEncoder.inverse_transform(range(len(proba)))
            else:
                classes = [f"Career_{i}" for i in range(len(proba))]
            careers = [{"career": classes[i], "confidence": round(float(p)*100, 2)} for i, p in enumerate(proba)]
            careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
    except Exception:
        careers = []

    # fallback: use text model by synthesizing pseudo-document from normalizedText
    if not careers:
        try:
            global vectorizer, text_model
            combined = " ".join(list(normalizedText.values()))
            if vectorizer is None or text_model is None:
                # a very simple fallback classification
                careers = [{"career": "General Studies", "confidence": 50.0}]
            else:
                X_in = vectorizer.transform([combined])
                pred = text_model.predict_proba(X_in)[0]
                # text_model.classes_ exists when trained
                labels = list(text_model.classes_) if hasattr(text_model, "classes_") else [f"Career_{i}" for i in range(len(pred))]
                careers = [{"career": labels[i], "confidence": round(float(p)*100, 2)} for i, p in enumerate(pred)]
                careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
        except Exception:
            careers = [{"career": "General Studies", "confidence": 50.0}]

    # Build suggestions + certificates based on mappedSkills
    for c in careers:
        suggestions = []
        cert_recs = []
        for subj, level in mappedSkills.items():
            subj_lower = subj.lower()
            if level == "Strong":
                suggestions.append(f"Excellent performance in {subj}! Keep building projects.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)
            elif level == "Average":
                suggestions.append(f"Good performance in {subj}. Additional practice or short courses recommended.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)
            elif level == "Weak":
                suggestions.append(f"Consider strengthening fundamentals in {subj}. Use tutorials and exercises.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)

        if "Engineer" in c["career"] or "Developer" in c["career"] or "Software" in c["career"]:
            suggestions.append("Build small coding projects and host them on GitHub.")
        if "Data" in c["career"] or "AI" in c["career"] or "Scientist" in c["career"]:
            suggestions.append("Try Python + data projects and Kaggle practice notebooks.")

        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Focus on IT-related subjects for stronger alignment."
        c["certificates"] = list(dict.fromkeys(cert_recs)) if cert_recs else careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])

    return careers

# ---------------------------
# Certificate analysis helper
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns with AI/ML and Data Scientist roles.",
        "webdev": "Web Development certificate enhances frontend/backend developer profile.",
        "python": "Python certification supports Data Science & Software Engineering careers."
    }
    if not certFiles:
        return [{"info": "No certificates uploaded"}]

    for cert in certFiles:
        name = cert.filename.lower()
        matched = [msg for key, msg in certificateSuggestions.items() if key in name]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds value to your profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Optional save to MySQL
# ---------------------------
def save_result_to_mysql(payload: dict):
    if not MYSQL_AVAILABLE or not DB_HOST or not DB_USER or not DB_PASS or not DB_NAME:
        return {"status": "skipped", "reason": "MySQL not configured in env vars or mysql-connector not installed"}
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS career_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255),
                prediction JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
        """)
        insert_q = "INSERT INTO career_predictions (name, email, prediction) VALUES (%s, %s, %s)"
        name = payload.get("name")
        email = payload.get("email")
        prediction_json = json.dumps(payload.get("prediction", {}))
        cur.execute(insert_q, (name, email, prediction_json))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------
# Routes
# ---------------------------
@app.post("/ocrPredict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None), saveToDb: Optional[bool] = Form(False), name: Optional[str] = Form(None), email: Optional[str] = Form(None)):
    """
    Upload a file (pdf/docx/txt/png/jpg) and optional certificate files.
    Returns:
      - careerPrediction (top)
      - careerOptions (top 3 with confidence)
      - subjects_structured (list)
      - mappedSkills
      - finalBuckets
      - certificates analysis
    """
    try:
        # Extract text
        text = extract_text_from_upload(file)
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text)

        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)
        if not careerOptions:
            careerOptions = [{
                "career": "General Studies",
                "confidence": 50.0,
                "suggestion": "Add more subjects or improve grades for a better match.",
                "certificates": careerCertSuggestions["General Studies"]
            }]

        certResults = analyzeCertificates(certificateFiles) if certificateFiles else [{"info": "No certificates uploaded"}]

        result = {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults,
            "extracted_text_snippet": text[:200]
        }

        if saveToDb:
            payload = {"name": name, "email": email, "prediction": result}
            db_res = save_result_to_mysql(payload)
            result["db_save"] = db_res

        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": {"structured_model": bool(structured_model), "text_model": bool(text_model)}}

# ---------------------------
# If run directly
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
