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
else:
    # fallback for Windows default install
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
# Load and Train Structured Data Model
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
# Certificate Mappings
# ---------------------------
subjectCertMap = {
    "computer programming": [
        "PCAP – Python Certified Associate",
        "Oracle Certified Java Programmer",
        "C++ Certified Associate Programmer"
    ],
    "object-oriented programming": [
        "Oracle Java SE Programmer Certification",
        "C# Programming Certification (Microsoft)",
        "Python OOP Certification"
    ],
    "integrative programming and technologies": [
        "Full-Stack Web Developer Certificate (The Odin Project)",
        "Meta Full-Stack Developer Certificate",
        "JavaScript Specialist Certification"
    ],
    "information management": [
        "Oracle Database SQL Associate",
        "Microsoft SQL Server Certification",
        "MongoDB Certified Developer Associate"
    ],
    "advance database systems": [
        "PostgreSQL Professional Certification",
        "MongoDB Certified Developer Associate",
        "Oracle MySQL Professional"
    ],
    "web systems and technologies": [
        "FreeCodeCamp Responsive Web Design",
        "Meta Front-End Developer Certificate",
        "W3C Front-End Web Developer Certificate"
    ],
    "system integration and architecture": [
        "AWS Solutions Architect",
        "Microsoft Azure Fundamentals",
        "Google Cloud Associate Engineer"
    ],
    "system administration and maintenance": [
        "CompTIA Linux+",
        "Microsoft Certified: Windows Server Administration",
        "Red Hat Certified System Administrator (RHCSA)"
    ],
    "networking 1": ["Cisco CCNA", "CompTIA Network+", "Juniper JNCIA"],
    "networking 2": ["Cisco CCNP", "CompTIA Security+", "Fortinet NSE Certification"],
    "data structure and algorithms": [
        "HackerRank Skills Certification (DSA)",
        "Google Kickstart Participation",
        "Coderbyte Algorithmic Certificate"
    ],
    "discrete structures for it": [
        "Mathematics for Computer Science (MITx)",
        "Coursera Discrete Math Specialization"
    ],
    "human computer interface": [
        "Google UX Design Certificate",
        "Adobe Certified Professional: UX Design",
        "Interaction Design Foundation Certificate"
    ],
    "science technology and society": [
        "Ethics in AI & Data Science (Coursera)",
        "Technology & Society Certificate"
    ],
    "introduction to computing": ["IC3 Digital Literacy Certification", "CompTIA IT Fundamentals+"],
    "hardware system and servicing": ["CompTIA A+", "PC Hardware Technician Certification"],
    "capstone project and research": [
        "Agile Scrum Certification",
        "Project Management Professional (PMP)",
        "Google Project Management Certificate"
    ]
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
# OCR & Subject Normalization Helpers
# ---------------------------
VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

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

TEXT_FIXES = {
    "tras beaives bstaegt": "Elective 5",
    "wage system integration and rotate 2 es": "System Integration and Architecture 2",
    "aot sten ainsaton and marenance": "System Administration and Maintenance",
    "capa capstone pret and research 2 es": "Capstone Project and Research 2",
    "mathnats nthe modem oa es": "Mathematics in the Modern World",
    "advan database systems": "Advance Database Systems",
    "capstone project and research 1 spparont cepsre": "Capstone Project and Research 1",
    "web systems and technologies 2 soxtsrowebsystemsbtechroiogies": "Web Systems and Technologies 2",
    "rane foreign languoge 2": "Foreign Language 2"
}

REMOVE_LIST = [
    "stone project ad reset",
    "student",
    "report of grades",
    "unknown subject",
    "category", "communications", "class", "united", "student no", "fullname",
]

def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    if not desc:
        return None
    s = desc.lower().strip()
    for bad in REMOVE_LIST:
        if bad in s:
            return None
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)
    return s.title() if s else None

# ---------------------------
# Extract Subjects & Grades
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

def extractSubjectGrades(text: str):
    subjects_structured = []
    rawSubjects = OrderedDict()
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if any(kw in line.lower() for kw in ignore_keywords):
            continue
        tokens = re.split(r'\s+', line)
        nums = [re.sub(r'[^0-9.]', '', t) for t in tokens if re.search(r'\d', t)]
        grade_val = None
        if nums:
            try:
                grade_val = snap_to_valid_grade(float(nums[-1]))
            except:
                grade_val = None
        subj_tokens = [t for t in tokens if not re.fullmatch(r'[^A-Za-z]*\d+[^A-Za-z]*', t)]
        subj_name = normalize_subject(None, " ".join([t for t in subj_tokens if not re.search(r'\d', t)]))
        if not subj_name:
            continue
        bucket = None
        for group, keywords in subjectGroups.items():
            if any(k in subj_name.lower() for k in keywords):
                bucket = bucketMap.get(group)
                break
        if bucket and grade_val is not None:
            bucket_grades[bucket].append(grade_val)
        mappedSkills[subj_name] = grade_to_level(grade_val)
        subjects_structured.append({"description": subj_name, "grade": grade_val})
        rawSubjects[subj_name] = grade_val

    finalBuckets = {k: round(sum(v)/len(v), 2) if v else 3.0 for k, v in bucket_grades.items()}
    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    return subjects_structured, rawSubjects, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction & Suggestions
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, mappedSkills: dict):
    careers = []
    if model and targetEncoder:
        try:
            dfInput = pd.DataFrame([{
                "Python": finalBuckets.get("Python",3.0),
                "SQL": finalBuckets.get("SQL",3.0),
                "Java": finalBuckets.get("Java",3.0)
            }])
            proba = model.predict_proba(dfInput)[0]
            careers = [
                {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100,2)}
                for i,p in enumerate(proba)
            ]
            careers = sorted(careers, key=lambda x:x["confidence"], reverse=True)[:3]
        except:
            careers = []

    if not careers:
        # fallback heuristic
        heuristics = [
            {"career": "Software Engineer", "score": finalBuckets.get("Java",3.0)},
            {"career": "Web Developer", "score": (finalBuckets.get("Java",3.0)+finalBuckets.get("SQL",3.0))/2},
            {"career": "Data Scientist", "score": finalBuckets.get("Python",3.0)}
        ]
        max_score = max(h["score"] for h in heuristics)
        for h in heuristics:
            conf = max(0.0,(max_score-h["score"])/max_score*100) if max_score else 50.0
            careers.append({"career": h["career"], "confidence": round(conf,2)})

    # suggestions
    for c in careers:
        suggestions = []
        for k,v in finalBuckets.items():
            if v <= 1.75:
                suggestions.append(f"Strong in {k} — consider advanced projects/certifications.")
            elif v <= 2.5:
                suggestions.append(f"Average in {k} — practice and small projects recommended.")
            else:
                suggestions.append(f"Weak in {k} — foundational courses and practice needed.")
        c["suggestion"] = " ".join(suggestions)
        c["certificates"] = careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])
    return careers

# ---------------------------
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    if not certFiles:
        return [{"info":"No certificates uploaded"}]
    results = []
    suggestionsMap = {
        "aws":"AWS certificate boosts Cloud/DevOps roles",
        "ccna":"CCNA boosts Networking/System admin roles",
        "datascience":"Data Science certificate aligns with AI/ML",
        "webdev":"Web Dev cert enhances frontend/backend profile",
        "python":"Python cert supports Data Science, AI, Software Eng"
    }
    for cert in certFiles:
        name = cert.filename.lower()
        matched = [msg for k,msg in suggestionsMap.items() if k in name]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds career value."]
        results.append({"file":cert.filename,"suggestions":matched})
    return results

# ---------------------------
# Routes
# ---------------------------
@app.post("/filePredict")
async def filePredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        text = await asyncio.to_thread(extract_text_from_file, file)
        subjects_structured, rawSubjects, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, mappedSkills)
        certResults = analyzeCertificates(certificateFiles)
        return {
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "General Studies",
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ocrPredict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    return await filePredict(file, certificateFiles)
