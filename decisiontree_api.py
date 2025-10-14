# filename: decisiontree_api.py

import re
import io
import os
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pytesseract
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# Gemini AI Integration
# ---------------------------
try:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    client = None
    print(f"⚠️ Gemini client not initialized: {e}")

async def improve_prediction_with_gemini(prediction_text: str) -> str:
    """Enhance the main career suggestion using Gemini for grammar + extra relevant tips."""
    if not client:
        return prediction_text

    prompt = f"""
    You are a career advisor AI. Improve the following text:
    - Fix grammar and typos
    - Keep a formal, professional tone (no emojis)
    - Add 2–3 related IT career insights or advice based on context
    - Output only the improved version

    Text:
    {prediction_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return prediction_text

# ---------------------------
# Windows Tesseract path (adjust if needed)
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---------------------------
# Input Schema
# ---------------------------
class StudentInput(BaseModel):
    python: int
    sql: int
    java: int

# ---------------------------
# Train Structured Data Model
# ---------------------------
df = pd.read_csv("cs_students.csv")

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
app = FastAPI(title="Career Prediction API (Gemini + OCR Normalization + Certificates)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Subject Classification and Settings
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "object oriented", "software", "coding", "development", "elective"],
    "databases": ["database", "sql", "dbms", "systems integration", "information systems", "data management"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics", "security", "assurance"],
    "networking": ["networking", "networks", "cloud", "infrastructure"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "os", "architecture", "computer systems"]
}

bucketMap = {"programming": "Java", "databases": "SQL", "ai_ml": "Python"}

ignore_keywords = [
    "course", "description", "final", "remarks", "re-exam", "units",
    "fullname", "year level", "program", "college", "student no",
    "academic year", "date printed", "gwa", "credits", "republic", "city",
    "report", "gender", "bachelor", "semester", "university"
]

# ---------------------------
# Certificate Maps
# ---------------------------
subjectCertMap = {
    "computer programming": ["PCAP – Python Certified Associate", "Oracle Certified Java Programmer", "C++ Certified Associate Programmer"],
    "object-oriented programming": ["Oracle Java SE Programmer Certification", "C# Programming Certification", "Python OOP Certification"],
    "information management": ["Oracle Database SQL Associate", "Microsoft SQL Server Certification", "MongoDB Certified Developer Associate"],
    "advance database systems": ["PostgreSQL Professional Certification", "MongoDB Certified Developer Associate", "Oracle MySQL Professional"],
    "web systems and technologies": ["FreeCodeCamp Responsive Web Design", "Meta Front-End Developer Certificate", "W3C Front-End Web Developer Certificate"],
    "system integration and architecture": ["AWS Solutions Architect", "Azure Fundamentals", "Google Cloud Associate Engineer"],
    "system administration and maintenance": ["CompTIA Linux+", "Windows Server Admin", "RHCSA"],
    "networking 1": ["Cisco CCNA", "CompTIA Network+", "Juniper JNCIA"],
    "networking 2": ["Cisco CCNP", "CompTIA Security+", "Fortinet NSE"],
    "data structure and algorithms": ["HackerRank DSA", "Google Kickstart", "Coderbyte Algorithms"],
    "discrete structures for it": ["Math for CS (MITx)", "Coursera Discrete Math"],
    "human computer interface": ["Google UX Design", "Adobe UX", "Interaction Design Foundation"],
    "introduction to computing": ["IC3 Digital Literacy", "CompTIA ITF+"],
    "hardware system and servicing": ["CompTIA A+", "PC Hardware Technician"],
    "capstone project and research": ["Agile Scrum", "PMP", "Google Project Management"]
}

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
# Grade Helpers
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
# OCR fixes & helpers (used by normalize_subject)
# ---------------------------
TEXT_FIXES = {
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
    "atari": "art appreciation",
    "natonl sncetrhing pega": "national service training program",
    "diserete sturt for it": "discrete structures for it",
    "networking": "networking 1",
    "understanding the se": "understanding the self",
    "purposve communication": "purposive communication",
    "mathematics in the modem world so": "mathematics in the modern world"
}

REMOVE_LIST = [
    "stone project ad reset",
    "catege ommuniatons crass uniteamed",
    "student",
    "acaserie eer agpy gna",
    "unknown subject",
    "category", "communications", "class", "united", "student no", "fullname",
    "report of grades", "republic", "city of", "wps", "office"
]

def normalize_subject_simple(raw_desc: str) -> Optional[str]:
    """
    Lightweight normalization: apply text-fixes, remove junk tokens, return Title-case or None.
    """
    if not raw_desc:
        return None
    s = raw_desc.lower().strip()
    # replace obvious OCR mistakes
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)
    # remove punctuation, extra spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    if not s:
        return None
    # filter remove-list
    for bad in REMOVE_LIST:
        if bad in s:
            return None
    # if too short, drop
    if len(s) < 3:
        return None
    return s.title()

# ---------------------------
# OCR Parsing Function (fixed)
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
        if not line or any(kw in line.lower() for kw in ignore_keywords):
            continue

        clean = re.sub(r'[\t\r\f\v]+', ' ', line)
        clean = re.sub(r'[^\w\.\-\s]', ' ', clean)   # allow dot/dash/space
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        # find tokens that look like grades (1.00, 1.75, 2, 3.00 etc.)
        float_tokens = []
        for i, tok in enumerate(parts):
            # sanitize token (remove trailing punctuation)
            tok_clean = tok.strip().strip('.,;:')
            if re.fullmatch(r'\d+(\.\d+)?', tok_clean):
                try:
                    f = float(tok_clean)
                    float_tokens.append((i, tok_clean, f))
                except:
                    continue

        if not float_tokens:
            continue

        # assume last numeric token is the grade (common in TORs)
        idx, tok_str, grade_raw = float_tokens[-1]
        gradeVal = snap_to_valid_grade(grade_raw)
        # description tokens = everything before the grade token
        desc_tokens = parts[:idx]
        subj_raw = " ".join(desc_tokens).strip()
        # fallback if subj_raw empty: try whole line except grade
        if not subj_raw:
            subj_raw = " ".join(parts[:-1]).strip()

        subj_clean = normalize_subject_simple(subj_raw) or ("Unknown Subject")
        subjDesc = subj_clean

        # map to buckets (by keywords in subject)
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned = bucketMap.get(group)
                if assigned and gradeVal is not None:
                    bucket_grades[assigned].append(gradeVal)
                break

        # populate structures
        subjects_structured.append({
            "description": subjDesc,
            "grade": gradeVal,
            "raw_line": raw_line
        })
        rawSubjects[subjDesc] = gradeVal
        normalizedText[subjDesc] = subjDesc
        mappedSkills[subjDesc] = grade_to_level(gradeVal) if gradeVal is not None else "Unknown"

    # compute final buckets average (default 3.0 when empty)
    finalBuckets = {}
    for k in ("Python", "SQL", "Java"):
        vals = bucket_grades.get(k, [])
        finalBuckets[k] = round(sum(vals)/len(vals), 2) if vals else 3.0

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Prediction Logic
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"],
    }])
    proba = model.predict_proba(dfInput)[0]
    careers = [{"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)} for i, p in enumerate(proba)]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        suggestions = []
        for subj, level in mappedSkills.items():
            if level == "Strong":
                suggestions.append(f"Excellent performance in {subj}. Explore certifications to enhance your credibility.")
            elif level == "Average":
                suggestions.append(f"Good progress in {subj}. More practice could help you reach excellence.")
            elif level == "Weak":
                suggestions.append(f"Consider reviewing {subj} to strengthen your fundamentals.")

        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Focus on core IT subjects."
        c["certificates"] = careerCertSuggestions.get(c["career"], ["General IT certifications recommended."])
    return careers

# ---------------------------
# Certificate File Analysis
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator roles.",
        "datascience": "Your Data Science certificate aligns with AI/ML careers.",
        "webdev": "Your Web Development certificate enhances frontend/backend profiles.",
        "python": "Your Python certification supports Data Science and Software Engineering."
    }
    for cert in certFiles:
        certName = cert.filename.lower()
        matched = [msg for key, msg in certificateSuggestions.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds value to your career."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Routes
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        text = await asyncio.to_thread(pytesseract.image_to_string, img)

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text.strip())
        careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        # ✅ Enhance the top suggestion using Gemini (if available)
        if careerOptions and careerOptions[0].get("suggestion"):
            top_text = careerOptions[0]["suggestion"]
            improved_text = await improve_prediction_with_gemini(top_text)
            careerOptions[0]["suggestion"] = improved_text

        if not careerOptions:
            careerOptions = [{
                "career": "General Studies",
                "confidence": 50.0,
                "suggestion": "Add more subjects or improve grades for a better match.",
                "certificates": careerCertSuggestions["General Studies"]
            }]

        certResults = analyzeCertificates(certificateFiles or []) if certificateFiles else [{"info": "No certificates uploaded"}]

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
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "gemini_connected": bool(client), "model_trained": True}
