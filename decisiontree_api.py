# filename: decisiontree_api.py

import re
import io
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image, UnidentifiedImageError
import pytesseract
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------
# Gemini AI Integration (Optional)
# ---------------------------
try:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    client = None
    print(f"⚠️ Gemini client not initialized: {e}")

async def improve_prediction_with_gemini(prediction_text: str) -> str:
    """Enhance the main career suggestion using Gemini for grammar + insights."""
    if not client:
        return prediction_text

    prompt = f"""
    You are a career advisor AI. Improve the following text:
    - Fix grammar and typos
    - Keep a formal tone
    - Add 2–3 related IT career insights or advice
    Output only the improved version.

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
# Tesseract Path (adjust if needed)
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
# FastAPI App + CORS
# ---------------------------
app = FastAPI(title="Career Prediction API (Gemini + OCR + Certificates)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Subject Keywords
# ---------------------------
subjectGroups = {
    "programming": ["programming", "java", "oop", "object oriented", "software", "coding", "development", "elective"],
    "databases": ["database", "sql", "dbms", "information systems", "data management"],
    "ai_ml": ["python", "machine learning", "ai", "data mining", "analytics"],
    "networking": ["network", "cloud", "infrastructure"],
    "webdev": ["html", "css", "javascript", "frontend", "backend", "php", "web"],
    "systems": ["operating systems", "os", "architecture", "computer systems"]
}

bucketMap = {"programming": "Java", "databases": "SQL", "ai_ml": "Python"}

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
# ---------------------------
# Helper Functions
# ---------------------------
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
# OCR Subject Extraction
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured = []
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for raw_line in lines:
        line = raw_line.strip().lower()
        # Skip obvious non-grade lines
        if any(line.startswith(kw) for kw in ["course description", "remarks", "final", "student no", "gwa", "college", "university"]):
            continue

        # Clean OCR noise
        clean = re.sub(r'[^\w\.\-\s,]', ' ', line)
        clean = re.sub(r'\s{2,}', ' ', clean).strip()
        if not clean:
            continue

        # Normalize commas to dots
        clean = clean.replace(",", ".")

        tokens = clean.split()
        float_tokens = [(i, tok, float(tok)) for i, tok in enumerate(tokens) if re.fullmatch(r'\d+(\.\d+)?', tok)]
        if not float_tokens:
            continue

        gradeVal = float_tokens[-1][2]
        gradeVal = snap_to_valid_grade(gradeVal)
        subjDesc = " ".join(tokens[:-1]).title()

        # Assign subject to group
        for bucket, keywords in subjectGroups.items():
            if any(k in subjDesc.lower() for k in keywords):
                mappedBucket = bucketMap.get(bucket)
                if mappedBucket:
                    bucket_grades[mappedBucket].append(gradeVal)
                break

        mappedSkills[subjDesc] = grade_to_level(gradeVal)
        subjects_structured.append({"description": subjDesc, "grade": gradeVal})

    finalBuckets = {k: (round(sum(v) / len(v), 2) if v else 3.0) for k, v in bucket_grades.items()}

    print("✅ Extracted Subjects:", subjects_structured)
    print("✅ Skill Mapping:", mappedSkills)
    print("✅ Final Buckets:", finalBuckets)

    return subjects_structured, mappedSkills, finalBuckets

# ---------------------------
# Career Prediction
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, mappedSkills: dict):
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
                suggestions.append(f"Excellent performance in {subj}. Explore certifications.")
            elif level == "Average":
                suggestions.append(f"Good progress in {subj}. More practice could help you reach excellence.")
            elif level == "Weak":
                suggestions.append(f"Consider reviewing {subj} to strengthen fundamentals.")
        c["suggestion"] = " ".join(suggestions[:8]) if suggestions else "Focus on your IT subjects."
        c["certificates"] = careerCertSuggestions.get(c["career"], ["General IT certifications recommended."])
    return careers

# ---------------------------
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    certificateSuggestions = {
        "aws": "Your AWS certificate strengthens Cloud Architect and DevOps paths.",
        "ccna": "Your CCNA boosts Networking and Systems Admin roles.",
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
# API Routes
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = None):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        imageBytes = await file.read()
        try:
            img = Image.open(io.BytesIO(imageBytes))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Unable to process image.")

        text = await asyncio.to_thread(pytesseract.image_to_string, img)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text detected in the image.")

        subjects_structured, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets, mappedSkills)

        if careerOptions:
            top_text = careerOptions[0]["suggestion"]
            improved_text = await improve_prediction_with_gemini(top_text)
            careerOptions[0]["suggestion"] = improved_text

        certResults = analyzeCertificates(certificateFiles or []) if certificateFiles else [{"info": "No certificates uploaded"}]

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "mappedSkills": mappedSkills,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "gemini_connected": bool(client), "model_trained": True}
