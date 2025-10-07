# decisiontree_api.py
"""
Fixed and hardened version of your career-prediction API.
- Removes unused/Windows-only imports
- Supports images, PDF, DOCX, TXT uploads
- Safe handling when cs_students.csv is missing (fallback training data)
- Adds root route for health check
- Prints OCR/debug logs (visible in Render logs)
- Keeps original feature set: OCR -> subject parsing -> bucket averages -> ML predict -> suggestions

Deploy with:
- Build Command (Render): apt-get update && apt-get install -y tesseract-ocr && pip install -r requirements.txt
- Start Command (Render): uvicorn decisiontree_api:app --host 0.0.0.0 --port 10000

Requirements (requirements.txt):
fastapi
uvicorn
pillow
pytesseract
pandas
scikit-learn
python-multipart
PyPDF2
python-docx

"""

import io
import re
import shutil
import asyncio
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pytesseract

# Configure tesseract binary if available
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("⚠️ Tesseract not found in PATH. Make sure it is installed on the server.")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Utility helpers and constants
# ---------------------------
VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

TEXT_FIXES = {
    # keep the map you had; trimmed here for brevity
    "inveductonto computing": "introduction to computing",
    "phystal edeation": "physical education",
}

REMOVE_LIST = [
    "student", "fullname", "student no", "report of grades", "republic", "city of"
]

subjectGroups = {
    "programming": ["programming", "java", "oop", "object oriented", "software", "coding", "development"],
    "databases": ["database", "sql", "dbms", "data management"],
    "ai_ml": ["python", "machine learning", "ai"],
}

bucketMap = {"programming": "Java", "databases": "SQL", "ai_ml": "Python"}

careerCertSuggestions = {
    "Software Engineer": ["AWS Cloud Practitioner", "Oracle Java SE"],
    "Web Developer": ["FreeCodeCamp", "Meta Frontend Dev"],
    "Data Scientist": ["Google Data Analytics", "TensorFlow Developer Cert."],
    "General Studies": ["Short IT courses to explore career interests"]
}

# ---------------------------
# ML model training (load cs_students.csv if available)
# If csv missing or broken, train a small fallback model.
# ---------------------------
try:
    df = pd.read_csv("cs_students.csv")
    print("Loaded cs_students.csv with shape:", df.shape)
except Exception as e:
    print("⚠️ Could not load cs_students.csv — using fallback training data. (", e, ")")
    df = pd.DataFrame({
        "Python": [1.5, 2.5, 3.0, 2.0],
        "SQL": [2.0, 2.5, 3.2, 2.1],
        "Java": [1.8, 2.4, 3.1, 2.0],
        "Future Career": ["Software Engineer", "Data Analyst", "Database Admin", "Web Developer"]
    })

features = ["Python", "SQL", "Java"]
target = "Future Career"

# ensure features exist and fill missing columns if needed
for f in features:
    if f not in df.columns:
        df[f] = 3.0

if target not in df.columns:
    df[target] = "General Studies"

# encode categorical features if any
labelEncoders = {}
for col in features:
    if df[col].dtype == "object":
        le_col = LabelEncoder()
        df[col] = le_col.fit_transform(df[col])
        labelEncoders[col] = le_col

# encode target
targetEncoder = LabelEncoder()
df[target] = targetEncoder.fit_transform(df[target].astype(str))

X = df[features]
y = df[target]

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
model.fit(X, y)
print("Model trained. Classes:", list(targetEncoder.classes_))

# ---------------------------
# Helper functions: normalization, parsing, OCR, suggestions
# ---------------------------

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
    if 0.0 < raw <= 5.0:
        return round(raw, 2)
    return None


def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    s = (desc or "").lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    if not s:
        return None
    for bad in REMOVE_LIST:
        if bad in s:
            return None
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)
    if len(s) < 3:
        return None
    return s.title()


def extractSubjectGrades(text: str):
    subjects_structured = []
    rawSubjects = OrderedDict()
    normalizedText = {}
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for raw_line in lines:
        low = raw_line.lower()
        if any(kw in low for kw in ["course", "description", "student", "fullname"]):
            continue
        clean = re.sub(r'[^\w\.\-\s]', ' ', raw_line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue
        parts = clean.split()
        if len(parts) < 2:
            continue
        # find numeric tokens
        float_tokens = []
        for i, tok in enumerate(parts):
            token_clean = re.sub(r'[^0-9.]', '', tok)
            if token_clean and re.search(r'\d', token_clean):
                try:
                    rawf = float(token_clean)
                    float_tokens.append((i, token_clean, rawf))
                except:
                    continue
        if not float_tokens:
            continue
        # assume last numeric token is units, prev is grade (heuristic)
        gradeVal = None
        unitsVal = None
        if len(float_tokens) >= 2:
            prev_idx, prev_tok, prev_raw = float_tokens[-2]
            gradeVal = _normalize_grade_str(prev_tok)
            gradeVal = snap_to_valid_grade(gradeVal)
        else:
            idx, tok, rawf = float_tokens[-1]
            gradeVal = _normalize_grade_str(tok)
            gradeVal = snap_to_valid_grade(gradeVal)
        if gradeVal is None:
            continue
        # description tokens before grade index
        grade_idx = float_tokens[-2][0] if len(float_tokens) >= 2 else float_tokens[-1][0]
        desc_tokens = parts[:grade_idx]
        if desc_tokens and re.fullmatch(r'\d+', desc_tokens[0]):
            desc_tokens = desc_tokens[1:]
        subjDesc_raw = " ".join(desc_tokens).strip() or "Unknown Subject"
        subjDesc = normalize_subject(None, subjDesc_raw)
        if subjDesc is None:
            continue
        # classify
        category = "IT Subject" if any(k in subjDesc.lower() for k in ["programming", "database", "java", "python", "sql"]) else "Minor Subject"
        # mapping to buckets
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket:
                    bucket_grades[assigned_bucket].append(gradeVal)
                break
        mappedSkills[subjDesc] = ("Strong" if gradeVal <= 1.75 else ("Average" if gradeVal <= 2.5 else "Weak"))
        subjects_structured.append({
            "description": subjDesc,
            "grade": gradeVal,
            "units": None,
            "remarks": None,
            "category": category
        })
        rawSubjects[subjDesc] = gradeVal
        normalizedText[subjDesc] = subjDesc

    finalBuckets = {}
    for b, grades in bucket_grades.items():
        finalBuckets[b] = round(sum(grades)/len(grades), 2) if grades else 3.0
    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets


# ---------------------------
# Prediction + suggestions
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"],
    }])

    try:
        proba = model.predict_proba(dfInput)[0]
    except Exception:
        # fallback: use predict and assign 100% to that class
        pred_idx = int(model.predict(dfInput)[0])
        proba = [0.0] * len(targetEncoder.classes_)
        proba[pred_idx] = 1.0

    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    it_keywords = ["programming", "database", "data", "system", "software", "network", "java", "python", "sql"]

    for c in careers:
        suggestions = []
        cert_recs = []
        for subj, level in mappedSkills.items():
            subj_lower = subj.lower()
            if not any(k in subj_lower for k in it_keywords):
                continue
            if level == "Strong":
                suggestions.append(f"Excellent performance in {subj}! Keep it up.")
                   # Suggest certificates based on career domain
    for key, certs in careerCertSuggestions.items():
        if key.lower() in predictedCareer.lower():
            topCareerSuggestions = certs

    return {
        "careerPrediction": predictedCareer,
        "careerOptions": list(careerCertSuggestions.keys()),
        "suggestedCertifications": topCareerSuggestions,
        "subjects_structured": subjects_structured,
        "finalBuckets": finalBuckets
    }


            elif level == "Average":
                suggestions.append(f"Good progress in {subj}, but you can still improve.")
            else:
                suggestions.append(f"Needs improvement in {subj}.")
        if "Developer" in c["career"] or "Engineer" in c["career"]:
            suggestions.append("Build small coding projects to apply your knowledge.")
        if "Data" in c["career"] or "AI" in c["career"]:
            suggestions.append("Try Python/ML projects to enhance your portfolio.")
        c["suggestion"] = " ".join(suggestions) if suggestions else "Focus on IT-related subjects for stronger career alignment."
        c["certificates"] = cert_recs if cert_recs else careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])

    return careers


# ---------------------------
# Certificate analysis
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns well with AI/ML roles.",
        "webdev": "Web Dev certificate enhances frontend/backend developer profile.",
        "python": "Python certification supports Data Science and Software Engineering careers."
    }
    for cert in certFiles:
        name = cert.filename.lower()
        matched = [msg for key, msg in certificateSuggestions.items() if key in name]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds additional value to your career profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results


# ---------------------------
# Routes
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Career Prediction API running. Use POST /ocrPredict with file=..."}


@app.post("/ocrPredict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        print("Received file:", file.filename)
        contents = await file.read()

        # try multiple extractors depending on file type
        text = ""
        fname = (file.filename or "").lower()
        try:
            if fname.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                img = Image.open(io.BytesIO(contents)).convert('RGB')
                text = await asyncio.to_thread(pytesseract.image_to_string, img)
            elif fname.endswith('.pdf'):
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(contents))
                pages = [p.extract_text() or "" for p in reader.pages]
                text = "\n".join(pages)
            elif fname.endswith('.docx'):
                from docx import Document
                doc = Document(io.BytesIO(contents))
                text = "\n".join([p.text for p in doc.paragraphs])
            elif fname.endswith('.txt'):
                text = contents.decode(errors='ignore')
            else:
                # fallback: try OCR anyway
                img = Image.open(io.BytesIO(contents)).convert('RGB')
                text = await asyncio.to_thread(pytesseract.image_to_string, img)
        except Exception as e:
            print("Extractor error:", e)

        print("OCR/Extractor output (first 400 chars):", text[:400].replace('\n', ' '))

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades((text or "").strip())
        print("Parsed subjects:", subjects_structured)

        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        if not careerOptions:
            careerOptions = [{
                "career": "General Studies",
                "confidence": 50.0,
                "suggestion": "Add more subjects or improve grades for a better match.",
                "certificates": careerCertSuggestions.get("General Studies", [])
            }]

        certResults = []
        if certificateFiles:
            certResults = analyzeCertificates(certificateFiles or [])
        else:
            certResults = [{"info": "No certificates uploaded"}]

        response = {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }

        print("Returning response summary:", {"career": response["careerPrediction"], "subjects_count": len(subjects_structured)})
        return response

    except Exception as e:
        print("Unhandled error in /ocrPredict:", e)
        return {"error": str(e)}
