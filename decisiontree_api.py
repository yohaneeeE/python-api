# filename: decisiontree_api.py

import re
import io
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
import json
from google import genai
import os
from pdf2image import convert_from_bytes
import cv2
import numpy as np

# Initialize Gemini client
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    client = None
    print(f"Gemini client not initialized: {e}")

# Windows Tesseract path (adjust if needed)
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
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Subject → Certificates Mapping
# ---------------------------
subjectCertMap = {
    # Core Programming
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
    # Databases
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
    # Web & Systems
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
    # Networking & Security
    "networking 1": [
        "Cisco CCNA",
        "CompTIA Network+",
        "Juniper JNCIA"
    ],
    "networking 2": [
        "Cisco CCNP",
        "CompTIA Security+",
        "Fortinet NSE Certification"
    ],
    # Algorithms & Data
    "data structure and algorithms": [
        "HackerRank Skills Certification (DSA)",
        "Google Kickstart Participation",
        "Coderbyte Algorithmic Certificate"
    ],
    "discrete structures for it": [
        "Mathematics for Computer Science (MITx)",
        "Coursera Discrete Math Specialization"
    ],
    # AI & Emerging Tech
    "human computer interface": [
        "Google UX Design Certificate",
        "Adobe Certified Professional: UX Design",
        "Interaction Design Foundation Certificate"
    ],
    "science technology and society": [
        "Ethics in AI & Data Science (Coursera)",
        "Technology & Society Certificate"
    ],
    # General IT Foundation
    "introduction to computing": [
        "IC3 Digital Literacy Certification",
        "CompTIA IT Fundamentals+"
    ],
    "hardware system and servicing": [
        "CompTIA A+",
        "PC Hardware Technician Certification"
    ],
    # Capstone / Research (Optional Guidance)
    "capstone project and research": [
        "Agile Scrum Certification",
        "Project Management Professional (PMP)",
        "Google Project Management Certificate"
    ]
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
# OCR Fixes & Helpers
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
    "understanding The sef": "understanding the self",
    "Understanding The Selff": "understanding the self",
    "purposve communication": "purposive communication",
    "mathematics in the modem world so": "mathematics in the modern world"
}

REMOVE_LIST = [
    "stone project ad reset", "catege ommuniatons crass uniteamed", "student",
    "acaserie eer agpy gna", "unknown subject", "category", "class", "united",
    "student no", "fullname", "report of grades", "republic", "city of", "wps",
    "office","total grades","section","ocr","passed","failed","ungraded",
    "bsit","major","year level","academic year","grade","total subject",
    "credit","units","average"
]

# ---------------------------
# Normalization Functions
# ---------------------------
def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    raw = desc or ""
    s = raw.lower().strip()
    s = re.sub(r'[_]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    if not s:
        return None

    for bad in REMOVE_LIST:
        pattern = r'\b' + re.escape(bad) + r'\b'
        if re.search(pattern, s):
            return None

    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)

    if "elective" in s:
        m = re.search(r'\b(\d{1,2})\b', s) or (re.search(r'(\d)', code) if code else None)
        num = m.group(1)[-1] if m else None
        return f"Elective {num}" if num else "Elective"

    if "pe" in s or "pathfit" in s:
        if code:
            m = re.search(r'(\d{1,3})', code)
            if m:
                return f"PE {m.group(1)}"
        return "PE"

    if "purposive" in s and "communication" in s:
        return "Purposive Communication"

    s = s.strip()
    if len(s) < 2 and s not in ["pe", "it", "os"]:
        return None

    return s.title()

def normalize_code(text: str) -> Optional[str]:
    if not text:
        return None
    return re.sub(r'\s+', '', text.upper())

def _normalize_grade_str(num_str: str):
    s = str(num_str or "").replace(',', '.').replace('l','1').replace('O','0')
    s = re.sub(r'[^0-9.]', '', s).strip()
    if s == "":
        return None
    try:
        raw = float(s)
    except:
        return None

    candidates = [raw, raw/10, raw/100]
    valid = [c for c in candidates if 1.0 <= c <= 5.0]
    if valid:
        return round(min(valid, key=lambda x: abs(x-2.5)),2)
    if 0.0 < raw <=5.0:
        return round(raw,2)
    return round(raw,2)

# ---------------------------
# Gemini Cleanup
# ---------------------------
async def improve_subjects_with_gemini(subjects: dict, skills: dict):
    if not client:
        return subjects, skills

    prompt = f"""
    You are an academic data cleaner AI.
    Fix spelling and capitalization for subjects, keep skill levels unchanged.
    Input: {json.dumps({"subjects": subjects, "skills": skills}, ensure_ascii=False)}
    Output JSON only.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        cleaned_text = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        cleaned = json.loads(cleaned_text)
        return cleaned.get("subjects", subjects), cleaned.get("skills", skills)
    except:
        return subjects, skills

# ---------------------------
# OCR Extraction
# ---------------------------
async def extractSubjectGrades(text: str):
    subjects_structured, rawSubjects, normalizedText, mappedSkills = [], OrderedDict(), {}, {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if any(kw in low for kw in ignore_keywords) and len(line.split()) < 5:
            continue

        clean = re.sub(r'[\t\r\f\v]+',' ', line)
        clean = re.sub(r'[^\w\.\-\s]',' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 2:
            continue

        subjCode = None
        if len(parts)>=2 and parts[0].isalpha() and parts[1].isdigit():
            subjCode = f"{parts[0].upper()} {parts[1]}"
            parts = parts[2:]
        elif re.match(r'^[A-Z]{1,4}\d{1,3}$', parts[0].upper()):
            subjCode = parts[0].upper()
            parts = parts[1:]

        if not parts:
            continue

        remarks=None
        if parts[-1].isalpha():
            remarks=parts[-1]
            parts=parts[:-1]
            if not parts: continue

        float_tokens=[]
        for i,tok in enumerate(parts):
            tok = tok.replace(',', '.').replace('l','1').replace('O','0')
            token_clean = re.sub(r'[^0-9.]','',tok)
            if token_clean and re.search(r'\d', token_clean):
                try: float_tokens.append((i,token_clean,float(token_clean)))
                except: continue

        gradeVal, unitsVal, grade_idx = None, None, None
        if len(float_tokens)>=2:
            prev_idx, prev_tok, prev_raw = float_tokens[-2]
            last_idx, last_tok, last_raw = float_tokens[-1]
            grade_idx=prev_idx
            gradeVal = snap_to_valid_grade(_normalize_grade_str(prev_tok))
            unitsVal = float(last_raw)
        elif len(float_tokens)==1:
            idx, tok, rawf=float_tokens[0]
            grade_idx=idx
            gradeVal = snap_to_valid_grade(_normalize_grade_str(tok))
            unitsVal=None
        else:
            continue

        desc_tokens = parts[:grade_idx] if grade_idx is not None else parts[:]
        if desc_tokens and re.fullmatch(r'\d+', desc_tokens[0]):
            desc_tokens=desc_tokens[1:]
        subjDesc_raw=" ".join(desc_tokens).strip() or subjCode or "Unknown Subject"
        subjDesc = normalize_subject(subjCode, subjDesc_raw)
        if subjDesc is None: continue

        subjKey=subjDesc
        category = "Major Subject" if "elective" in subjDesc.lower() else (
            "IT Subject" if any(k in subjDesc.lower() for k in [
                "programming","database","data","system","integration","architecture",
                "software","network","computing","information","security","java","python","sql","web","algorithm"
            ]) else "Minor Subject"
        )

        lower_desc=subjDesc.lower()
        for group,keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket=bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
                    bucket_grades[assigned_bucket].append(gradeVal)
                break

        mappedSkills[subjDesc]=grade_to_level(gradeVal) if gradeVal is not None else "Unknown"
        subjects_structured.append({
            "code": subjCode,
            "subject": subjDesc,
            "grade": gradeVal,
            "units": unitsVal,
            "remarks": remarks,
            "category": category
        })

        rawSubjects[subjDesc_raw]={
            "grade": gradeVal,
            "units": unitsVal,
            "remarks": remarks
        }

    subjects_structured, mappedSkills = await improve_subjects_with_gemini(subjects_structured, mappedSkills)
    return subjects_structured, rawSubjects, bucket_grades, mappedSkills

# ---------------------------
# OCR from PDF/Image
# ---------------------------
async def read_pdf(file: UploadFile):
    contents = await file.read()
    pages = convert_from_bytes(contents, dpi=300)
    images=[]
    for p in pages:
        img = p.convert("L")  # grayscale
        arr = np.array(img)
        _, thresh = cv2.threshold(arr, 150, 255, cv2.THRESH_BINARY)
        images.append(Image.fromarray(thresh))
    text=""
    for img in images:
        text += pytesseract.image_to_string(img, lang="eng") + "\n"
    return text

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/predict")
async def predict_career(file: UploadFile=File(...)):
    text = await read_pdf(file)
    subjects, rawSubjects, bucket_grades, mappedSkills = await extractSubjectGrades(text)

    avgGrades={}
    for k,v in bucket_grades.items():
        avgGrades[k]=round(sum(v)/len(v),2) if v else None

    bucket_preds={k:(labelEncoders.get(k,None).inverse_transform([int(round(v))])[0] if k in labelEncoders and v else "Unknown") for k,v in avgGrades.items()}

    career_scores={}
    for career, certs in careerCertSuggestions.items():
        career_scores[career]=sum(
            [1 for bucket,val in bucket_grades.items() if val and val<=2.5]
        )

    top_careers=sorted(career_scores.items(), key=lambda x:-x[1])[:3]
    results=[]
    for c,s in top_careers:
        certs=careerCertSuggestions.get(c,[])
        results.append({
            "career":c,
            "score":s,
            "certificates":certs
        })

    return {
        "subjects": subjects,
        "skills": mappedSkills,
        "career_matches": results
    }
