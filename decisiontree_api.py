# filename: decisiontree_api.py
import os
import re
import io
import json
import asyncio
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# ---------------------------
# Gemini client support (try multiple package variants)
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Try new "from google import genai" style
genai_new = None
genai_legacy = None
genai_client = None
try:
    # modern package (user sample)
    from google import genai as genai_new
    try:
        genai_client = genai_new.Client()  # will pick up GEMINI_API_KEY from env
    except Exception:
        genai_client = None
except Exception:
    genai_new = None

if genai_client is None:
    try:
        # fallback to legacy package name
        import google.generativeai as genai_legacy
        if GEMINI_API_KEY:
            try:
                genai_legacy.configure(api_key=GEMINI_API_KEY)
            except Exception:
                # ignore configure failure, will handle later
                pass
    except Exception:
        genai_legacy = None

# ---------------------------
# Core imports and config
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
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates + Gemini enhancer)")

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
    "introduction to computing": [
        "IC3 Digital Literacy Certification",
        "CompTIA IT Fundamentals+"
    ],
    "hardware system and servicing": [
        "CompTIA A+",
        "PC Hardware Technician Certification"
    ],
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

# Known OCR misreads to fix (add more as you discover them)
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

def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    raw = desc or ""
    s = raw.lower().strip()

    # remove underscores, stray punctuation and multiple spaces
    s = re.sub(r'[_]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    if not s:
        return None

    for bad in REMOVE_LIST:
        if bad in s:
            return None

    # Replace known OCR misreads
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)

    # Elective special-case: try preserve trailing elective number
    if "elective" in s:
        num = None
        m = re.search(r'\b(\d{1,2})\b', s)
        if m:
            num = m.group(1)[-1]
        return f"Elective {num}" if num else "Elective"

    # PE / Pathfit
    if s.strip() == "pe" or "pe " in s or "pathfit" in s or s.startswith("pe "):
        if code:
            m = re.search(r'(\d{1,3})', code)
            if m:
                return f"PE {m.group(1)}"
        return "PE"

    # Purposive Communication
    if "purposive" in s and "communication" in s:
        return "Purposive Communication"

    s = s.strip()
    if len(s) < 3:
        return None

    return s.title()

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
# OCR Preprocessing helpers
# ---------------------------
def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Basic preprocessing to improve tesseract accuracy."""
    # Convert to grayscale, normalize contrast, denoise, threshold, upscale
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    # adaptive threshold-like: simple point threshold
    img = img.point(lambda x: 0 if x < 140 else 255, mode="1")
    w, h = img.size
    if max(w, h) < 2000:
        scale = 1.8
    else:
        scale = 1.2
    img = img.resize((int(w * scale), int(h * scale)))
    return img

async def ocr_image_to_text(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    img = preprocess_image_for_ocr(img)
    text = await asyncio.to_thread(pytesseract.image_to_string, img, config="--oem 3 --psm 6 -l eng")
    # keep safe characters
    text = re.sub(r'[^A-Za-z0-9.\n\s\-]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# ---------------------------
# OCR Extraction (original robust parser kept)
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
        category = None
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
# Career Prediction with Smarter Suggestions (original logic preserved)
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"],
    }])

    proba = model.predict_proba(dfInput)[0]
    careers = [
        {"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p)*100, 2)}
        for i, p in enumerate(proba)
    ]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    it_keywords = [
        "programming", "database", "data", "system", "integration", "architecture",
        "software", "network", "computing", "information", "security",
        "java", "python", "sql", "web", "algorithm", "ai", "machine learning"
    ]

    for c in careers:
        suggestions = []
        cert_recs = []

        for subj, level in mappedSkills.items():
            subj_lower = subj.lower()

            if not any(k in subj_lower for k in it_keywords):
                continue

            if level == "Strong":
                suggestions.append(f"Excellent performance in {subj}! Keep it up.")
                suggestions.append(f"Since you're strong in {subj}, consider certifications to prove your skill.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)

            elif level == "Average":
                suggestions.append(f"Good progress in {subj}, but you can still improve.")
                suggestions.append(f"Extra practice or online short courses in {subj} could help you excel.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)

            elif level == "Weak":
                suggestions.append(f"You need to strengthen your foundation in {subj}.")
                suggestions.append(f"Study resources, tutorials, and practice exercises in {subj} are highly recommended.")
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)

        if "Developer" in c["career"] or "Engineer" in c["career"]:
            suggestions.append("Build small coding projects to apply your knowledge.")
        if "Data" in c["career"] or "AI" in c["career"]:
            suggestions.append("Try Python/ML projects to enhance your data science portfolio.")

        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Focus on IT-related subjects for stronger career alignment."
        c["certificates"] = cert_recs if cert_recs else careerCertSuggestions.get(
            c["career"], ["Consider general IT certifications."]
        )

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
        certName = cert.filename.lower()
        matched = [msg for key, msg in certificateSuggestions.items() if key in certName]
        if not matched:
            matched = [f"Certificate '{cert.filename}' adds additional value to your career profile."]
        results.append({"file": cert.filename, "suggestions": matched})
    return results

# ---------------------------
# Gemini helpers (optional enhancer)
# ---------------------------
async def gemini_clean_subjects_and_grades(ocr_text: str):
    """
    Use Gemini to clean OCR output and return a structured JSON:
    {"subjects":[{"name":"Subject Name","grade":1.75}, ...], "skills": {...}}
    If Gemini is not available, return {} (the caller will fall back to local parsing).
    """
    if genai_client:
        # new client style: client.models.generate_content
        try:
            # run in thread to avoid blocking
            resp = await asyncio.to_thread(
                genai_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=f"""
                Clean and structure this OCR transcript of student grades.
                - Fix typos and capitalization of subject names.
                - Correct misspellings.
                - Make sure all grades have decimals (e.g. 2 -> 2.00).
                - Return JSON exactly in this form:
                  {{
                    "subjects": [{{"name": "...", "grade": 1.75}}],
                    "skills": {{}}
                  }}
                Text:
                {ocr_text}
                """
            )
            # resp.text is expected
            cleaned = json.loads(resp.text.strip())
            return cleaned
        except Exception:
            return {}
    elif genai_legacy:
        # legacy package: genai_legacy.GenerativeModel(...)
        try:
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            resp = await asyncio.to_thread(model.generate_content, f"""
                Clean and structure this OCR transcript of student grades.
                - Fix typos and capitalization of subject names.
                - Correct misspellings.
                - Make sure all grades have decimals (e.g. 2 -> 2.00).
                - Return JSON exactly in this form:
                  {{
                    "subjects": [{{"name": "...", "grade": 1.75}}],
                    "skills": {{}}
                  }}
                Text:
                {ocr_text}
            """)
            cleaned = json.loads(resp.text.strip())
            return cleaned
        except Exception:
            return {}
    else:
        return {}

async def gemini_generate_career_suggestions(finalBuckets, subjects):
    """
    Ask Gemini for top 3-4 career matches and 3-4 sentence plain text suggestions
    for each (no emojis). Returns {"careers":[{"career":"...", "confidence":nn, "suggestion":"..."}]}
    """
    if genai_client:
        try:
            prompt = f"""
            Based on these average bucket grades:
            {finalBuckets}
            and these subjects:
            {subjects}

            Suggest 3-4 IT-related career paths and for each return:
            - career (short title)
            - confidence (0-100)
            - suggestion (3-4 sentences, plain text, no emojis)
            Return JSON like:
            {{
              "careers":[
                {{"career":"Software Engineer","confidence":90,"suggestion":"..."}},
                ...
              ]
            }}
            """
            resp = await asyncio.to_thread(
                genai_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=prompt
            )
            return json.loads(resp.text.strip())
        except Exception:
            return {}
    elif genai_legacy:
        try:
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Based on these average bucket grades:
            {finalBuckets}
            and these subjects:
            {subjects}

            Suggest 3-4 IT-related career paths and for each return:
            - career (short title)
            - confidence (0-100)
            - suggestion (3-4 sentences, plain text, no emojis)
            Return JSON like:
            {{
              "careers":[
                {{"career":"Software Engineer","confidence":90,"suggestion":"..."}},
                ...
              ]
            }}
            """
            resp = await asyncio.to_thread(model.generate_content, prompt)
            return json.loads(resp.text.strip())
        except Exception:
            return {}
    else:
        return {}

# ---------------------------
# Routes
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    """
    Main endpoint:
    - Run OCR (preprocess -> tesseract)
    - Parse with original extractSubjectGrades
    - Optionally ask Gemini to clean OCR text and/or generate career suggestions
    - Return combined results (original parsing + Gemini enhancements when available)
    """
    try:
        imageBytes = await file.read()
        # Step A: OCR with preprocessing
        ocr_text = await ocr_image_to_text(imageBytes)

        # Step B: Local parsing (deterministic)
        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(ocr_text)

        # Step C: Try to get Gemini cleaned structured subjects (non-blocking)
        gemini_data = {}
        try:
            gemini_data = await gemini_clean_subjects_and_grades(ocr_text)
        except Exception:
            gemini_data = {}

        # If Gemini returned structured subjects, merge/override cleaned values carefully
        if gemini_data and isinstance(gemini_data.get("subjects"), list) and gemini_data.get("subjects"):
            # Build lookup by name (lower) from gemini
            gmap = {}
            for s in gemini_data.get("subjects", []):
                name = s.get("name")
                grade = s.get("grade")
                if name:
                    gmap[name.lower()] = {"name": name, "grade": round(float(grade), 2) if grade is not None else None}

            # Replace descriptions and grades in subjects_structured where gemini suggests a correction
            for subj in subjects_structured:
                key = subj.get("description", "").lower()
                if key in gmap:
                    corrected = gmap[key]
                    # update description and grade if available
                    subj["description"] = corrected.get("name", subj["description"])
                    if corrected.get("grade") is not None:
                        subj["grade"] = snap_to_valid_grade(corrected.get("grade"))

            # Also refresh normalizedText and mappedSkills using corrected data
            normalizedText = {s["description"]: s["description"] for s in subjects_structured}
            mappedSkills = {s["description"]: grade_to_level(s["grade"]) for s in subjects_structured}

            # Recompute finalBuckets from corrected subject list
            # (ensure decimals and bucket mapping)
            bucket_grades = {"Python": [], "SQL": [], "Java": []}
            for s in subjects_structured:
                name = s.get("description", "").lower()
                grade = s.get("grade")
                for group, keywords in subjectGroups.items():
                    if any(k in name for k in keywords):
                        assigned = bucketMap.get(group)
                        if assigned and grade is not None:
                            bucket_grades[assigned].append(grade)
                        break
            for b in ("Python", "SQL", "Java"):
                finalBuckets[b] = round(sum(bucket_grades[b]) / len(bucket_grades[b]), 2) if bucket_grades[b] else finalBuckets.get(b, 3.0)

        # Step D: Get career options - prefer Gemini suggestions if available, else local predict
        gemini_careers = {}
        try:
            gemini_careers = await gemini_generate_career_suggestions(finalBuckets, normalizedText)
        except Exception:
            gemini_careers = {}

        if gemini_careers and isinstance(gemini_careers.get("careers"), list) and gemini_careers.get("careers"):
            careerOptions = gemini_careers["careers"]
        else:
            careerOptions = predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills)

        # Step E: Certificate file analysis (preserve original behavior)
        certResults = []
        if certificateFiles:
            certResults = analyzeCertificates(certificateFiles or [])
        else:
            certResults = [{"info": "No certificates uploaded"}]

        # Prepare final output
        return {
            "careerPrediction": careerOptions[0]["career"] if careerOptions else "General Studies",
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": normalizedText,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults,
            "rawOCR": ocr_text,
            "gemini_used": bool(genai_client or genai_legacy)
        }
    except Exception as e:
        return {"error": str(e)}

# Simple health check
@app.get("/")
def root():
    return {"status": "ok", "message": "Career Prediction API (merged rule-based + Gemini enhancer) is running."}
