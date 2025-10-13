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
import cv2
import numpy as np
import re
from PIL import Image
import os

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
    """
    Normalize and clean a subject description. Returns cleaned title-case desc or None (to drop).
    - code: the detected code (e.g. "IT 102") or None
    - desc: raw description tokens before grade
    """
    raw = desc or ""   # ignore course code in the displayed string
    s = raw.lower().strip()

    # remove underscores, stray punctuation and multiple spaces
    s = re.sub(r'[_]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

    if not s:
        return None

    # Remove obvious junk (contains any token from remove list)
    for bad in REMOVE_LIST:
        if bad in s:
            return None

    # Replace known OCR misreads
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)

    # Elective special-case: try preserve trailing elective number
    if "elective" in s:
        # try to grab an elective number from code or from the string
        num = None
        # look for a trailing digit token in s
        m = re.search(r'\b(\d{1,2})\b', s)
        if m:
            num = m.group(1)[-1]  # last digit
        elif code:
            m2 = re.search(r'(\d)', code)
            if m2:
                num = m2.group(1)
        return f"Elective {num}" if num else "Elective"

    # PE / Pathfit
    if s.strip() == "pe" or "pe " in s or "pathfit" in s or s.startswith("pe "):
        # Keep "PE" (optionally include number from code)
        if code:
            # try to extract number from code (E10 or PE 10)
            m = re.search(r'(\d{1,3})', code)
            if m:
                return f"PE {m.group(1)}"
        return "PE"

    # Purposive Communication
    if "purposive" in s and "communication" in s:
        # try to include code prefix if available
        return "Purposive Communication"

    # Trim obvious headings/columns like "student" etc already covered above
    # Final cleanup and Title case
    s = s.strip()
    # avoid leaving strings like '5' or single chars
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

async def improve_subjects_with_gemini(subjects: dict, skills: dict):
    """
    Use Gemini (new SDK) to clean, correct, and enhance subject names and skill mappings.
    - Fix typograhical errors and grammar.
    - Normalize capitalization (e.g., 'programming 1' → 'Programming 1').
    - Keep skill values (Strong, Average, Weak) unchanged.
    - Return clean, valid JSON with the same structure.
    """

    if not client:
        return subjects, skills

    prompt = f"""
    You are an academic data cleaner AI.
    The following are subjects and their corresponding skill levels extracted from OCR.
    Your task is to fix spelling mistakes, correct capitalization, and ensure the subject names
    are formatted clearly and professionally.

    Rules:
    - Keep all skill level values exactly as they are (“Strong”, “Average”, “Weak”).
    - Maintain the same structure as the input.
    - Do not add, remove, or rename fields.
    - 3 to 4 sentences
    - Output valid JSON only (no explanations, no markdown, no extra text).

    Example:
    Input:  {{ "subjects": {{"programming 1": "Strong"}}, "skills": {{"Python": "Average"}} }}
    Output: {{ "subjects": {{"Programming 1": "Strong"}}, "skills": {{"Python": "Average"}} }}

    Input:
    {{
        "subjects": {json.dumps(subjects, ensure_ascii=False)},
        "skills": {json.dumps(skills, ensure_ascii=False)}
    }}
    """

    try:
        # Use Gemini to clean the text
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        cleaned_text = response.text.strip()

        # Some Gemini outputs may include backticks or explanations — strip those
        cleaned_text = re.sub(r"^```json|```$", "", cleaned_text, flags=re.MULTILINE).strip()

        # Attempt to parse Gemini output as JSON
        cleaned = json.loads(cleaned_text)

        # Gracefully fallback to originals if any part is missing
        return cleaned.get("subjects", subjects), cleaned.get("skills", skills)

    except json.JSONDecodeError:
        print("Gemini returned invalid JSON — fallback to raw subjects.")
        return subjects, skills
    except Exception as e:
        print(f"Gemini subjects cleanup error: {e}")
        return subjects, skills


# ---------------------------
# OCR Extraction
# ---------------------------
async def extractSubjectGrades(text: str):

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
        # else leave subjCode None and treat tokens as description + numbers

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
                except:
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
            unitsVal = float(last_raw)
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
        if not subjDesc_raw:
            subjDesc_raw = subjCode or "Unknown Subject"

        # Normalize & filter subject name
        subjDesc_clean = normalize_subject(subjCode, subjDesc_raw)
        if subjDesc_clean is None:
            # filtered as junk
            continue

        subjDesc = subjDesc_clean
        subjKey = subjDesc   # ✅ no course code in keys
        category = None
        # classify after normalization
        category = "Major Subject" if "elective" in subjDesc.lower() else (
            "IT Subject" if any(k in subjDesc.lower() for k in [
                "programming", "database", "data", "system", "integration", "architecture",
                "software", "network", "computing", "information", "security", "java",
                "python", "sql", "web", "algorithm"
            ]) else "Minor Subject"
        )

        # determine mapping to skill bucket (for ML only)
        lower_desc = subjDesc.lower()
        for group, keywords in subjectGroups.items():
            if any(k in lower_desc for k in keywords):
                assigned_bucket = bucketMap.get(group)
                if assigned_bucket and gradeVal is not None:
                    bucket_grades[assigned_bucket].append(gradeVal)
                break

        # store subject skill level (Weak/Average/Strong) for UI
        mappedSkills[subjDesc] = grade_to_level(gradeVal) if gradeVal is not None else "Unknown"

        # store
        subjects_structured.append({
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

    for k in ("Python", "SQL", "Java"):
        finalBuckets.setdefault(k, 3.0)

    # 🧠 Clean extracted subjects and skills with Gemini
    improvedSubjects, improvedSkills = await improve_subjects_with_gemini(normalizedText, mappedSkills)

    return subjects_structured, rawSubjects, improvedSubjects, improvedSkills, finalBuckets


# ---------------------------
# Career Prediction with Smarter Suggestions (IT-only focus + Subject Certs)
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

    # Keywords to consider as IT-related
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

            # ✅ Skip non-IT related subjects
            if not any(k in subj_lower for k in it_keywords):
                continue  

            if level == "Strong":
                suggestions.append(f"Excellent performance in {subj}! Keep it up 🚀.")
                suggestions.append(f"Since you're strong in {subj}, consider certifications to prove your skill.")
                # If strong but no cert yet → recommend certs too
                for key, certs in subjectCertMap.items():
                    if key in subj_lower:
                        cert_recs.extend(certs)

            elif level == "Average":
                suggestions.append(f"Good progress in {subj}, but you can still improve 📘.")
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

        # Add career-specific hints
        if "Developer" in c["career"] or "Engineer" in c["career"]:
            suggestions.append("💻 Build small coding projects to apply your knowledge.")
        if "Data" in c["career"] or "AI" in c["career"]:
            suggestions.append("📊 Try Python/ML projects to enhance your data science portfolio.")

        # Attach suggestions + certs
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


async def preprocess_image(image_bytes):
    """Convert, enhance, and clean the uploaded image for better OCR accuracy."""
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize (increase resolution)
    scale_percent = 150
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise and sharpen
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)

    # Convert back to PIL Image for pytesseract
    processed_img = Image.fromarray(processed)

    # Optional: auto-rotate if text is sideways
    try:
        osd = pytesseract.image_to_osd(processed_img)
        rotation = int(re.search('(?<=Rotate: )\d+', osd).group(0))
        if rotation != 0:
            processed_img = processed_img.rotate(-rotation, expand=True)
    except Exception as e:
        print("Rotation detection failed:", e)

    return processed_img


async def fix_ocr_text_with_gemini(text: str):
    """Use Gemini to clean and fix OCR text errors."""
    if not client:
        return text

    prompt = f"""
    You are an OCR correction AI. The following text was extracted from a student's transcript of records.
    Fix typos, correct spacing, and preserve grades (e.g., 1.50, 2.75, 3.00).
    Output only the cleaned text, no explanations.

    Text:
    {text}
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini OCR correction error:", e)
        return text

# ---------------------------
# Routes
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
       @app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        processed_img = await preprocess_image(imageBytes)


        # ✅ Use better OCR config
        config = "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./- "
        text = await asyncio.to_thread(pytesseract.image_to_string, processed_img, config=config)

        # ✅ Auto-fix OCR text using Gemini
        text = await fix_ocr_text_with_gemini(text)


        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = await extractSubjectGrades(text.strip())
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
