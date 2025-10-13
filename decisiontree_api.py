# filename: decisiontree_api.py

import re
import io
import json
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
from google import genai

# ---------------------------
# Gemini Client
# ---------------------------
client = genai.Client()  # reads GEMINI_API_KEY from environment

async def improveSubjectsWithGemini(subjects: dict, mappedSkills: dict):
    """
    Fix typos, normalize capitalization, and add 3-4 sentence career suggestions.
    Returns updated subjects and mappedSkills.
    """
    prompt = f"""
You are an expert career counselor AI. Subjects and skill levels:
{subjects} with skills {mappedSkills}.
1. Correct typos or spellings in subject names.
2. Normalize capitalization and suggest proper subject names.
3. Provide 3–4 sentence career advice per subject based on skill level (Strong/Average/Weak).
Return JSON like: {{
  "subjects": {{subject_name: corrected_name}},
  "skills": {{
      subject_name: {{
          "level": "Strong/Average/Weak",
          "suggestion": "Your 3-4 sentence advice here."
      }}
  }}
}}
"""
    try:
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )
        parsed = json.loads(response.text)
        updatedSubjects = parsed.get("subjects", {})
        updatedSkills = parsed.get("skills", {})
        return updatedSubjects, updatedSkills
    except Exception:
        # fallback
        return subjects, {k: {"level": v, "suggestion": ""} for k, v in mappedSkills.items()}


# ---------------------------
# Tesseract Path
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
# Train ML Model
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
# FastAPI App
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
    "academic year", "date printed", "gwa", "credits", "republic", "city", "report",
    "gender", "bachelor", "semester", "university"
]

# ---------------------------
# Subject → Certificates Mapping
# ---------------------------
subjectCertMap = {
    # Core Programming
    "computer programming": ["PCAP – Python Certified Associate", "Oracle Certified Java Programmer", "C++ Certified Associate Programmer"],
    "object-oriented programming": ["Oracle Java SE Programmer Certification", "C# Programming Certification (Microsoft)", "Python OOP Certification"],
    "integrative programming and technologies": ["Full-Stack Web Developer Certificate (The Odin Project)", "Meta Full-Stack Developer Certificate", "JavaScript Specialist Certification"],
    # Databases
    "information management": ["Oracle Database SQL Associate", "Microsoft SQL Server Certification", "MongoDB Certified Developer Associate"],
    "advance database systems": ["PostgreSQL Professional Certification", "MongoDB Certified Developer Associate", "Oracle MySQL Professional"],
    # Web & Systems
    "web systems and technologies": ["FreeCodeCamp Responsive Web Design", "Meta Front-End Developer Certificate", "W3C Front-End Web Developer Certificate"],
    "system integration and architecture": ["AWS Solutions Architect", "Microsoft Azure Fundamentals", "Google Cloud Associate Engineer"],
    "system administration and maintenance": ["CompTIA Linux+", "Microsoft Certified: Windows Server Administration", "Red Hat Certified System Administrator (RHCSA)"],
    # Networking & Security
    "networking 1": ["Cisco CCNA", "CompTIA Network+", "Juniper JNCIA"],
    "networking 2": ["Cisco CCNP", "CompTIA Security+", "Fortinet NSE Certification"],
    # Algorithms & Data
    "data structure and algorithms": ["HackerRank Skills Certification (DSA)", "Google Kickstart Participation", "Coderbyte Algorithmic Certificate"],
    "discrete structures for it": ["Mathematics for Computer Science (MITx)", "Coursera Discrete Math Specialization"],
    # AI & Emerging Tech
    "human computer interface": ["Google UX Design Certificate", "Adobe Certified Professional: UX Design", "Interaction Design Foundation Certificate"],
    "science technology and society": ["Ethics in AI & Data Science (Coursera)", "Technology & Society Certificate"],
    # General IT Foundation
    "introduction to computing": ["IC3 Digital Literacy Certification", "CompTIA IT Fundamentals+"],
    "hardware system and servicing": ["CompTIA A+", "PC Hardware Technician Certification"],
    # Capstone / Research
    "capstone project and research": ["Agile Scrum Certification", "Project Management Professional (PMP)", "Google Project Management Certificate"]
}

# ---------------------------
# Hardcoded Career Certificates
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
# Helpers
# ---------------------------
VALID_GRADES = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 5.00]

def grade_to_level(grade: float) -> str:
    if grade is None: return "Unknown"
    if grade <= 1.75: return "Strong"
    if grade <= 2.5: return "Average"
    return "Weak"

def snap_to_valid_grade(val: float):
    if val is None: return None
    return min(VALID_GRADES, key=lambda g: abs(g - val))

# ---------------------------
# OCR Fixes & Normalization
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
    "report of grades", "republic", "city of", "wps", "office", "bsit"
]

def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    s = desc.lower().strip()
    for bad in REMOVE_LIST:
        if bad in s: return None
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s: s = s.replace(wrong, correct)
    if "elective" in s:
        m = re.search(r'\d+', s)
        num = m.group(0)[-1] if m else None
        return f"Elective {num}" if num else "Elective"
    if "pe" in s or "pathfit" in s: return "PE"
    if "purposive" in s and "communication" in s: return "Purposive Communication"
    if len(s) < 3: return None
    return s.title()

def _normalize_grade_str(num_str: str):
    s = re.sub(r'[^0-9.]', '', str(num_str or '')).strip()
    if not s: return None
    try:
        raw = float(s)
    except:
        return None
    candidates = [raw, raw/10.0, raw/100.0]
    valid = [c for c in candidates if 1.0 <= c <= 5.0]
    return round(min(valid, key=lambda x: abs(x-2.5)), 2) if valid else round(raw,2)

# ---------------------------
# OCR Extraction
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
        if any(kw in line.lower() for kw in ignore_keywords): continue

        clean = re.sub(r'[^\w\.\-\s]', ' ', line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean: continue

        parts = clean.split()
        if len(parts) < 2: continue

        subjCode = None
        if len(parts)>=2 and parts[0].isalpha() and parts[1].isdigit():
            subjCode = f"{parts[0].upper()} {parts[1]}"
            parts = parts[2:]
        elif re.match(r'^[A-Z]{1,4}\d{1,3}$', parts[0].upper()):
            subjCode = parts[0].upper()
            parts = parts[1:]

        if not parts: continue

        # find numeric tokens (grade/unit)
        float_tokens = []
        for i, tok in enumerate(parts):
            tok_clean = re.sub(r'[^0-9.]', '', tok)
            if tok_clean: float_tokens.append((i,tok_clean,float(tok_clean)))

        if not float_tokens: continue
        idx, tok, rawf = float_tokens[0]
        gradeVal = snap_to_valid_grade(_normalize_grade_str(tok))
        unitsVal = float(float_tokens[-1][2]) if len(float_tokens)>1 else None
        grade_idx = idx

        desc_tokens = parts[:grade_idx] if grade_idx is not None else parts[:]
        if desc_tokens and re.fullmatch(r'\d+', desc_tokens[0]): desc_tokens = desc_tokens[1:]
        subjDesc_raw = " ".join(desc_tokens).strip() or subjCode or "Unknown Subject"

        subjDesc_clean = normalize_subject(subjCode, subjDesc_raw)
        if not subjDesc_clean: continue

        subjDesc = subjDesc_clean
        category = "Major Subject" if "elective" in subjDesc.lower() else "IT Subject"
        mappedSkills[subjDesc] = grade_to_level(gradeVal)

        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
                    bucket_grades[assigned_bucket].append(gradeVal)
                break

        subjects_structured.append({
            "description": subjDesc,
            "grade": gradeVal,
            "units": float(unitsVal) if unitsVal else None,
            "remarks": None,
            "category": category
        })

        rawSubjects[subjDesc] = gradeVal
        normalizedText[subjDesc] = subjDesc

    finalBuckets = {}
    for b, grades in bucket_grades.items():
        finalBuckets[b] = round(sum(grades)/len(grades),2) if grades else 3.0

    for k in ("Python","SQL","Java"): finalBuckets.setdefault(k,3.0)

    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"]
    }])
    proba = model.predict_proba(dfInput)[0]
    careers = [{"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100,2)} for i,p in enumerate(proba)]
    careers = sorted(careers,key=lambda x:x["confidence"],reverse=True)[:3]

    it_keywords = ["programming","database","data","system","integration","architecture","software","network","computing","information","security","java","python","sql","web","algorithm","ai","machine learning"]

    for c in careers:
        suggestions=[]
        cert_recs=[]
        for subj, level in mappedSkills.items():
            if not any(k in subj.lower() for k in it_keywords): continue
            if level=="Strong":
                suggestions.append(f"Excellent performance in {subj}! Keep it up.")
            elif level=="Average":
                suggestions.append(f"Good progress in {subj}, you can improve further.")
            elif level=="Weak":
                suggestions.append(f"Work on strengthening your foundation in {subj}.")
            for key, certs in subjectCertMap.items():
                if key in subj.lower(): cert_recs.extend(certs)

        if "Developer" in c["career"] or "Engineer" in c["career"]:
            suggestions.append("Build small coding projects to apply your knowledge.")
        if "Data" in c["career"] or "AI" in c["career"]:
            suggestions.append("Try Python/ML projects to enhance your data portfolio.")

        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Focus on IT-related subjects."
        c["certificates"] = cert_recs if cert_recs else careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])
    return careers

# ---------------------------
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results=[]
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps career paths.",
        "ccna": "Your CCNA boosts Networking and Systems Administrator opportunities.",
        "datascience": "Data Science certificate aligns well with AI/ML and Data Scientist roles.",
        "webdev": "Web Development certificate enhances your frontend/backend profile.",
        "python": "Python certification supports Data Science, AI, and Software Engineering careers."
    }
    for cert in certFiles:
        certName = cert.filename.lower()
        matched = [msg for key,msg in certificateSuggestions.items() if key in certName]
        if not matched: matched = [f"Certificate '{cert.filename}' adds additional value to your career profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# API Route
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        text = await asyncio.to_thread(pytesseract.image_to_string,img)

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text.strip())

        # Gemini enhancement
        updatedSubjects, updatedSkills = await improveSubjectsWithGemini(normalizedText, mappedSkills)
        careerOptions = predictCareerWithSuggestions(finalBuckets, updatedSubjects, {k:v["level"] for k,v in updatedSkills.items()})

        if not careerOptions:
            careerOptions=[{"career":"General Studies","confidence":50.0,"suggestion":"Add more subjects or improve grades.","certificates":careerCertSuggestions["General Studies"]}]

        certResults = analyzeCertificates(certificateFiles or []) if certificateFiles else [{"info":"No certificates uploaded"}]

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": updatedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}
