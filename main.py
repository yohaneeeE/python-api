# main.py
"""
Merged Career Prediction API
Combines features from:
 - decisiontree.py
 - decisiontree_api.py
 - image.py
 - main.py (your earlier versions)

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
 - POST /ocrPredict  (file: UploadFile, certificateFiles: optional multiple UploadFile)
 - POST /predict     (file(s): UploadFile; simpler interface)
 - GET  /health
"""

import os
import io
import re
import json
import traceback
from collections import OrderedDict
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import docx

# Optional MySQL connector (install if you want DB save)
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except Exception:
    MYSQL_AVAILABLE = False

# ---------------------------
# Configuration (env)
# ---------------------------
CSV_PATH = os.getenv("CSV_PATH", "app/models/cs_students.csv")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Career Prediction API (merged)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ---------------------------
# Grade normalization helpers (from your originals)
# ---------------------------
VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

def snap_to_valid_grade(val: Optional[float]):
    if val is None:
        return None
    return min(VALID_GRADES, key=lambda g: abs(g - val))

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

def grade_to_level(grade: Optional[float]) -> str:
    if grade is None:
        return "Unknown"
    if grade <= 1.75:
        return "Strong"
    elif grade <= 2.5:
        return "Average"
    else:
        return "Weak"

# OCR fixes & normalizations (shortened + extendable)
TEXT_FIXES = {
    "advan database systems": "Advance Database Systems",
    "camper prararining": "Computer Programming",
    "purposve communication": "Purposive Communication",
    "mathematics in the modem world": "Mathematics in the Modern World",
    "aot sten ainsaton and marenance": "System Integration and Maintenance",
}

REMOVE_LIST = [
    "student", "report of grades", "republic", "city of", "wps",
    "fullname", "student no", "academic year", "date printed", "gwa"
]

# Subject groups & bucket map (from your originals)
subjectGroups = {
    "programming": [
        "programming", "java", "oop", "object oriented",
        "software", "coding", "development", "elective"
    ],
    "databases": [
        "database", "sql", "dbms", "information systems", "data management"
    ],
    "ai_ml": [
        "python", "machine learning", "ai", "data mining", "analytics"
    ],
    "networking": ["networking", "networks", "cloud", "infrastructure"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "os", "architecture", "computer systems"]
}

bucketMap = {
    "programming": "Java",
    "databases": "SQL",
    "ai_ml": "Python"
}

# Subject->certificate recommendations (shortened from your mapping)
subjectCertMap = {
    "computer programming": [
        "PCAP – Python Certified Associate",
        "Oracle Certified Java Programmer"
    ],
    "object-oriented programming": [
        "Oracle Java SE Programmer Certification",
        "Python OOP Certification"
    ],
    "information management": [
        "Oracle Database SQL Associate",
        "Microsoft SQL Server Certification"
    ],
    "web systems and technologies": [
        "FreeCodeCamp Responsive Web Design",
        "Meta Front-End Developer Certificate"
    ],
    "networking 1": ["Cisco CCNA", "CompTIA Network+"],
    "data structure and algorithms": ["HackerRank Skills Certification (DSA)"],
    "introduction to computing": ["IC3 Digital Literacy Certification", "CompTIA IT Fundamentals+"]
}

careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp Responsive Web Design", "Meta Frontend Dev"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer Certificate"],
    "Database Administrator": ["Oracle SQL Associate"],
    "Cybersecurity Specialist": ["CompTIA Security+"],
    "General Studies": ["Short IT courses to explore career interests"]
}

# ---------------------------
# File -> Text extraction helpers (robust)
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

def extract_text_from_upload_file(file: UploadFile) -> str:
    filename = (file.filename or "").lower()
    contents = file.file.read()
    # reset for potential reuse
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

    # fallback
    try:
        return contents.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------------------------
# Subject & grade extraction (robust copy from your originals)
# ---------------------------
def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    raw = desc or ""
    s = raw.lower().strip()
    s = re.sub(r'[_]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    if not s or len(s) < 3:
        return None

    for bad in REMOVE_LIST:
        if bad in s:
            return None

    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct.lower())

    if "elective" in s:
        m = re.search(r'\b(\d{1,2})\b', s)
        num = m.group(1) if m else None
        return f"Elective {num}" if num else "Elective"

    if s == "pe" or s.startswith("pe ") or "pathfit" in s:
        if code:
            mm = re.search(r'(\d{1,3})', code)
            if mm:
                return f"PE {mm.group(1)}"
        return "PE"

    return s.title()

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
        if any(kw in low for kw in ["fullname", "student", "report of grades", "academic year", "date printed"]):
            continue

        clean = re.sub(r'[\t\r\f\v]+', ' ', line)
        clean = re.sub(r'[^\w\.\-\s]', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 1:
            continue

        subjCode = None
        if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
            subjCode = f"{parts[0].upper()} {parts[1]}"
            parts = parts[2:]
        elif re.match(r'^[A-Z]{1,4}\d{1,3}$', parts[0].upper()):
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
# Model training/loading
# ---------------------------
structured_model = None
targetEncoder = None
labelEncoders = {}
text_model = None
vectorizer = None

def train_or_load_models():
    global structured_model, targetEncoder, labelEncoders, text_model, vectorizer
    # Simple TF-IDF fallback
    docs = [
        "Math Physics Programming Data Structures",
        "Networking Security Database Computer Systems",
        "UI Design Multimedia Graphic Design HTML CSS"
    ]
    labels = ["Software Engineer", "Network Specialist", "UI/UX Designer"]
    vectorizer = TfidfVectorizer()
    X_text = vectorizer.fit_transform(docs)
    text_model = RandomForestClassifier(n_estimators=50, random_state=42)
    text_model.fit(X_text, labels)

    # Try load CSV for structured model
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # expected features if present
            features = [c for c in ["GPA", "Python", "SQL", "Java", "Interested Domain"] if c in df.columns]
            target_col = "Future Career" if "Future Career" in df.columns else (df.columns[-1] if len(df.columns) > 0 else None)
            if target_col is None:
                return

            data = df.copy()
            for col in features:
                if data[col].dtype == "object":
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    labelEncoders[col] = le
            targetEncoder = LabelEncoder()
            data[target_col] = targetEncoder.fit_transform(data[target_col].astype(str))

            X = data[features].fillna(3.0)
            y = data[target_col]
            structured_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            structured_model.fit(X, y)
        except Exception:
            structured_model = None

# Run at startup
train_or_load_models()

# ---------------------------
# Career prediction & suggestion logic
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    careers = []
    try:
        if structured_model is not None:
            # prepare input. only include columns model expects (try Python/SQL/Java/GPA if available)
            cols = structured_model.feature_names_in_ if hasattr(structured_model, "feature_names_in_") else ["Python","SQL","Java"]
            row = {c: finalBuckets.get(c, 3.0) for c in cols}
            df_in = pd.DataFrame([row])
            proba = structured_model.predict_proba(df_in)[0]
            # get class labels
            if targetEncoder is not None:
                # targetEncoder may not be in global scope due to training; attempt inverse on indices
                classes = targetEncoder.inverse_transform(list(range(len(proba))))
            else:
                classes = [f"Career_{i}" for i in range(len(proba))]
            careers = [{"career": classes[i], "confidence": round(float(p)*100,2)} for i,p in enumerate(proba)]
            careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
    except Exception:
        careers = []

    # fallback: use text_model
    if not careers:
        try:
            combined = " ".join(list(normalizedText.values()))
            if vectorizer is None or text_model is None:
                careers = [{"career": "General Studies", "confidence": 50.0}]
            else:
                X_in = vectorizer.transform([combined])
                if hasattr(text_model, "predict_proba"):
                    proba = text_model.predict_proba(X_in)[0]
                    labels = list(text_model.classes_)
                    careers = [{"career": labels[i], "confidence": round(float(p)*100,2)} for i,p in enumerate(proba)]
                    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]
                else:
                    pred = text_model.predict(X_in)[0]
                    careers = [{"career": pred, "confidence": 80.0}]
        except Exception:
            careers = [{"career":"General Studies","confidence":50.0}]

    # suggestions + certs from mappedSkills
    for c in careers:
        suggestions = []
        cert_recs = []
        for subj, level in mappedSkills.items():
            subj_lower = subj.lower()
            if level == "Strong":
                suggestions.append(f"Excellent performance in {subj}! Build a portfolio or projects.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)
            elif level == "Average":
                suggestions.append(f"Good performance in {subj}. Consider short courses and practical exercises.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)
            elif level == "Weak":
                suggestions.append(f"Consider improving fundamentals in {subj}. Use tutorials and practice tasks.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)

        # Career specific hints
        if any(k in c["career"].lower() for k in ["engineer","developer","software"]):
            suggestions.append("Start building small coding projects; publish on GitHub.")
        if any(k in c["career"].lower() for k in ["data","scientist","ai"]):
            suggestions.append("Practice Python + data projects; try Kaggle or hands-on tutorials.")

        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Improve related subjects for better alignment."
        c["certificates"] = list(dict.fromkeys(cert_recs)) if cert_recs else careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])

    return careers

# ---------------------------
# Certificate filename analysis
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud & DevOps roles.",
        "ccna": "Your CCNA boosts Networking opportunities.",
        "datascience": "Data Science certificate aligns with Data Scientist roles.",
        "webdev": "Web Dev certificate helps Frontend/Fullstack roles.",
        "python": "Python certification supports Data Science & Software Engineering."
    }
    if not certFiles:
        return [{"info":"No certificates uploaded"}]
    for cert in certFiles:
        name = cert.filename.lower()
        matched = [msg for key,msg in certificateSuggestions.items() if key in name]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds value to your profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Optional DB save
# ---------------------------
def save_result_to_mysql(payload: dict):
    if not MYSQL_AVAILABLE or not DB_HOST or not DB_USER or not DB_PASS or not DB_NAME:
        return {"status":"skipped","reason":"MySQL not configured or connector missing"}
    try:
        conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT or 3306, user=DB_USER, password=DB_PASS, database=DB_NAME)
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
        return {"status":"ok"}
    except Exception as e:
        return {"status":"error","error":str(e)}

# ---------------------------
# Routes
# ---------------------------
@app.post("/ocrPredict")
async def ocrPredict(
    file: UploadFile = File(...),
    certificateFiles: Optional[List[UploadFile]] = File(None),
    saveToDb: Optional[bool] = Form(False),
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None)
):
    """
    Upload one main file (PDF/DOCX/TXT/PNG/JPG) and optional certificate files.
    Returns structured subjects, mapped skills, final buckets, career predictions, and certificate analysis.
    """
    try:
        # Extract text
        text = extract_text_from_upload_file(file)
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text)

        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)
        if not careerOptions:
            careerOptions = [{
                "career":"General Studies",
                "confidence":50.0,
                "suggestion":"Add more subjects or improve grades.",
                "certificates": careerCertSuggestions["General Studies"]
            }]

        certResults = analyzeCertificates(certificateFiles) if certificateFiles else [{"info":"No certificates uploaded"}]

        result = {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults,
            "extracted_text_snippet": text[:300]
        }

        if saveToDb:
            payload = {"name":name, "email":email, "prediction": result}
            db_res = save_result_to_mysql(payload)
            result["db_save"] = db_res

        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

@app.post("/predict-multi")
async def predict_multi(files: List[UploadFile] = File(...)):
    """
    Accept multiple files (PDF/DOCX/TXT/PNG/JPG). Combines extracted text from all and produces a single analysis.
    """
    try:
        combined_text = ""
        for f in files:
            combined_text += "\n" + extract_text_from_upload_file(f)
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(combined_text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)
        return JSONResponse({
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "General Studies",
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

@app.get("/health")
async def health():
    return {"status":"ok", "models_loaded": {"structured_model": bool(structured_model), "text_model": bool(text_model)}}

# ---------------------------
# Run with uvicorn when executed directly
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level="info")
