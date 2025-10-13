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
from google import genai  # ✅ Gemini API

# ---------------------------
# Gemini Setup
# ---------------------------
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    client = None
    print(f"Gemini client not initialized: {e}")

async def improve_prediction_with_gemini(prediction_text: str) -> str:
    """
    Use Gemini to improve grammar, fix typos, and add extra related suggestions.
    Keeps the tone formal, avoids emojis.
    """
    if not client:
        return prediction_text

    prompt = f"""
    You are an expert career counselor AI. Improve and reformat the following career prediction summary:
    - Correct grammar and typos.
    - Add 2–3 more relevant suggestions that align with IT or the predicted career.
    - Maintain a professional and encouraging tone.
    - Do not use emojis or slang.
    - Keep the output concise but helpful.

    Prediction Summary:
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
app = FastAPI(title="Career Prediction API (Gemini Enhanced)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Subject Groups & Mappings
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
    "course", "description", "final", "remarks", "re-exam", "units", "fullname", "year level", "program",
    "college", "student no", "academic year", "date printed", "gwa", "credits", "republic", "city",
    "report", "gender", "bachelor", "semester", "university"
]

# ---------------------------
# Certificates Mapping
# ---------------------------
subjectCertMap = {
    "computer programming": ["PCAP – Python Certified Associate", "Oracle Certified Java Programmer", "C++ Certified Associate Programmer"],
    "object-oriented programming": ["Oracle Java SE Programmer Certification", "C# Programming Certification (Microsoft)", "Python OOP Certification"],
    "integrative programming and technologies": ["Full-Stack Web Developer Certificate (The Odin Project)", "Meta Full-Stack Developer Certificate", "JavaScript Specialist Certification"],
    "information management": ["Oracle Database SQL Associate", "Microsoft SQL Server Certification", "MongoDB Certified Developer Associate"],
    "advance database systems": ["PostgreSQL Professional Certification", "MongoDB Certified Developer Associate", "Oracle MySQL Professional"],
    "web systems and technologies": ["FreeCodeCamp Responsive Web Design", "Meta Front-End Developer Certificate", "W3C Front-End Web Developer Certificate"],
    "system integration and architecture": ["AWS Solutions Architect", "Microsoft Azure Fundamentals", "Google Cloud Associate Engineer"],
    "system administration and maintenance": ["CompTIA Linux+", "Microsoft Certified: Windows Server Administration", "Red Hat Certified System Administrator (RHCSA)"],
    "networking 1": ["Cisco CCNA", "CompTIA Network+", "Juniper JNCIA"],
    "networking 2": ["Cisco CCNP", "CompTIA Security+", "Fortinet NSE Certification"],
    "data structure and algorithms": ["HackerRank Skills Certification (DSA)", "Google Kickstart Participation", "Coderbyte Algorithmic Certificate"],
    "discrete structures for it": ["Mathematics for Computer Science (MITx)", "Coursera Discrete Math Specialization"],
    "human computer interface": ["Google UX Design Certificate", "Adobe Certified Professional: UX Design", "Interaction Design Foundation Certificate"],
    "science technology and society": ["Ethics in AI & Data Science (Coursera)", "Technology & Society Certificate"],
    "introduction to computing": ["IC3 Digital Literacy Certification", "CompTIA IT Fundamentals+"],
    "hardware system and servicing": ["CompTIA A+", "PC Hardware Technician Certification"],
    "capstone project and research": ["Agile Scrum Certification", "Project Management Professional (PMP)", "Google Project Management Certificate"]
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
# OCR Extraction & Parsing
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
        clean = re.sub(r'[^\w\.\-\s]', ' ', clean)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 2:
            continue

        float_tokens = [(i, tok, float(tok)) for i, tok in enumerate(parts) if re.fullmatch(r'\d+(\.\d+)?', tok)]
        if not float_tokens:
            continue

        gradeVal = float_tokens[-1][2]
        gradeVal = snap_to_valid_grade(gradeVal)
        subjDesc = " ".join(parts[:-1]).title()

        if any(k in subjDesc.lower() for k in subjectGroups["ai_ml"]):
            bucket_grades["Python"].append(gradeVal)
        elif any(k in subjDesc.lower() for k in subjectGroups["databases"]):
            bucket_grades["SQL"].append(gradeVal)
        elif any(k in subjDesc.lower() for k in subjectGroups["programming"]):
            bucket_grades["Java"].append(gradeVal)

        mappedSkills[subjDesc] = grade_to_level(gradeVal)
        subjects_structured.append({"description": subjDesc, "grade": gradeVal})

    finalBuckets = {k: (round(sum(v) / len(v), 2) if v else 3.0) for k, v in bucket_grades.items()}
    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, normalizedText: dict, mappedSkills: dict):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"],
    }])
    proba = model.predict_proba(dfInput)[0]
    careers = [{"career": targetEncoder.inverse_transform([i])[0], "confidence": round(float(p) * 100, 2)} for i, p in enumerate(proba)]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        suggestions = []
        for subj, level in mappedSkills.items():
            if level == "Strong":
                suggestions.append(f"You performed strongly in {subj}. Consider pursuing certifications or advanced projects related to it.")
            elif level == "Average":
                suggestions.append(f"You have good progress in {subj}. Taking extra tutorials or projects can further improve your skills.")
            elif level == "Weak":
                suggestions.append(f"Your foundation in {subj} needs strengthening. Review key concepts or enroll in refresher courses.")
        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Focus on core IT subjects."
        c["certificates"] = careerCertSuggestions.get(c["career"], ["Consider general IT certifications."])
    return careers

# ---------------------------
# Certificate Analysis
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
            matched = [f"Certificate '{cert.filename}' adds extra value to your portfolio."]
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

        # ✅ Enhance top suggestion using Gemini
        if careerOptions:
            raw_suggestion = careerOptions[0]["suggestion"]
            improved_suggestion = await improve_prediction_with_gemini(raw_suggestion)
            careerOptions[0]["suggestion"] = improved_suggestion

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
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}
