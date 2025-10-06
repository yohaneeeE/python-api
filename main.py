# main.py — Full Career Prediction + Subject Analysis + Suggestions
import os
import io
import re
import asyncio
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pytesseract
from docx import Document
import fitz  # PyMuPDF

# ---------------------------
# Tesseract OCR Path
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="Career Prediction API 🚀")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Career Prediction API is running!"}

# ---------------------------
# Load & Train Model
# ---------------------------
model: Optional[RandomForestClassifier] = None
targetEncoder: Optional[LabelEncoder] = None
CS_CSV = "cs_students.csv"

if os.path.exists(CS_CSV):
    df = pd.read_csv(CS_CSV)
    features = ["Python", "SQL", "Java"]
    target = "Future Career"
    labelEncoders = {}

    for col in features:
        if df[col].dtype == "object":
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
        print("✅ Model trained successfully")

# ---------------------------
# Configuration
# ---------------------------
subjectGroups = {
    "programming": ["programming","java","oop","object oriented","software","coding","development","elective"],
    "databases": ["database","sql","dbms","systems integration","information systems","data management"],
    "ai_ml": ["python","machine learning","ai","data mining","analytics","security","assurance"],
    "networking":["networking","networks","cloud","infrastructure"],
    "webdev":["html","css","javascript","frontend","backend","php","web"],
    "systems":["operating systems","os","architecture","computer systems"]
}
bucketMap = {"programming":"Java","databases":"SQL","ai_ml":"Python"}
ignore_keywords = ["course","description","final","remarks","re-exam","units","fullname","year","student no"]

subjectCertMap = {
    "computer programming":["PCAP – Python Certified Associate","Oracle Java SE Programmer","C++ Certified Associate"],
    "information management":["Oracle SQL Associate","MS SQL Server Cert","MongoDB Developer Associate"]
}

careerCertSuggestions = {
    "Software Engineer":["AWS Cloud Practitioner","Oracle Java SE"],
    "Web Developer":["FreeCodeCamp","Meta Frontend Dev"],
    "Data Scientist":["Google Data Analytics","TensorFlow Dev Cert."],
    "Database Administrator":["Oracle SQL Associate","MS SQL Server"],
    "Cloud Solutions Architect":["AWS Solutions Architect","Azure Fundamentals"],
    "Cybersecurity Specialist":["CompTIA Security+","Cisco CyberOps Associate"],
    "General Studies":["Short IT courses to explore career interests"]
}

VALID_GRADES=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,5.0]

# ---------------------------
# Helper Functions
# ---------------------------
def snap_to_valid_grade(val: Optional[float]) -> Optional[float]:
    if val is None: return None
    return min(VALID_GRADES,key=lambda g: abs(g-val))

def grade_to_level(grade: Optional[float]) -> str:
    if grade is None: return "Unknown"
    if grade<=1.75: return "Strong"
    elif grade<=2.5: return "Average"
    return "Weak"

def normalize_subject(code: Optional[str], desc: str) -> Optional[str]:
    s = desc.lower().strip()
    if not s or any(k in s for k in ignore_keywords): return None
    return s.title()

# ---------------------------
# Extract Text from File
# ---------------------------
def extract_text_from_file(upload: UploadFile) -> str:
    filename = upload.filename.lower()
    file_bytes = upload.file.read()
    if filename.endswith(".pdf"):
        text=""
        with fitz.open(stream=file_bytes,filetype="pdf") as doc:
            for page in doc: text+=page.get_text("text")+"\n"
        return text
    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith(".txt"):
        return file_bytes.decode("utf-8",errors="ignore")
    else:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)

# ---------------------------
# Extract Subjects & Grades
# ---------------------------
def extractSubjectGrades(text: str):
    subjects_structured=[]
    rawSubjects=OrderedDict()
    mappedSkills={}
    bucket_grades={"Python":[],"SQL":[],"Java":[]}

    lines=[l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if any(kw in line.lower() for kw in ignore_keywords): continue
        tokens = re.split(r'\s+',line)
        nums=[re.sub(r'[^0-9.]','',t) for t in tokens if re.search(r'\d',t)]
        grade_val = snap_to_valid_grade(float(nums[-1])) if nums else None
        subj_name = normalize_subject(None," ".join([t for t in tokens if not re.search(r'\d',t)]))
        if not subj_name: continue

        # bucket mapping
        for group,kws in subjectGroups.items():
            if any(k in subj_name.lower() for k in kws):
                bucket = bucketMap.get(group)
                if bucket and grade_val: bucket_grades[bucket].append(grade_val)

        mappedSkills[subj_name]=grade_to_level(grade_val)
        subjects_structured.append({"description":subj_name,"grade":grade_val})
        rawSubjects[subj_name]=grade_val

    finalBuckets={k:round(sum(v)/len(v),2) if v else 3.0 for k,v in bucket_grades.items()}
    for k in ["Python","SQL","Java"]: finalBuckets.setdefault(k,3.0)
    return subjects_structured, rawSubjects, mappedSkills, finalBuckets

# ---------------------------
# Predict Career & Suggestions
# ---------------------------
def predictCareerWithSuggestions(finalBuckets: dict, mappedSkills: dict):
    careers=[]
    if model and targetEncoder:
        dfInput=pd.DataFrame([{
            "Python":finalBuckets.get("Python",3.0),
            "SQL":finalBuckets.get("SQL",3.0),
            "Java":finalBuckets.get("Java",3.0)
        }])
        proba=model.predict_proba(dfInput)[0]
        careers=[{"career":targetEncoder.inverse_transform([i])[0],"confidence":round(float(p)*100,2)} for i,p in enumerate(proba)]
        careers=sorted(careers,key=lambda x:x["confidence"],reverse=True)[:3]
    else:
        careers=[{"career":"Software Engineer","confidence":50.0},{"career":"Web Developer","confidence":45.0},{"career":"Data Scientist","confidence":40.0}]

    for c in careers:
        suggestions=[]
        for subj,level in mappedSkills.items():
            if level=="Strong": suggestions.append(f"Excellent in {subj}! Consider advanced projects or certifications.")
            elif level=="Average": suggestions.append(f"Good in {subj}, practice more for improvement.")
            else: suggestions.append(f"Needs improvement in {subj}, focus on fundamentals.")
        c["suggestion"]=" ".join(suggestions[:8])
        c["certificates"]=careerCertSuggestions.get(c["career"],["General IT certifications recommended."])
    return careers

# ---------------------------
# Analyze Uploaded Certificates
# ---------------------------
def analyzeCertificates(certFiles: Optional[List[UploadFile]]):
    if not certFiles: return [{"info":"No certificates uploaded"}]
    results=[]
    suggestionsMap={"aws":"Boosts Cloud/DevOps roles","ccna":"Networking/System admin","datascience":"AI/ML/Data Science","webdev":"Frontend/Backend dev","python":"Supports Data/AI/Software Eng"}
    for cert in certFiles:
        name=cert.filename.lower()
        matched=[msg for k,msg in suggestionsMap.items() if k in name]
        if not matched: matched=[f"Certificate '{cert.filename}' adds value."]
        results.append({"file":cert.filename,"suggestions":matched})
    return results

# ---------------------------
# API Route
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), certificateFiles: Optional[List[UploadFile]] = File(None)):
    try:
        text = await asyncio.to_thread(extract_text_from_file,file)
        subjects_structured, rawSubjects, mappedSkills, finalBuckets = extractSubjectGrades(text)
        careerOptions = predictCareerWithSuggestions(finalBuckets,mappedSkills)
        certResults = analyzeCertificates(certificateFiles)
        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": careerOptions,
            "subjects_structured": subjects_structured,
            "finalBuckets": finalBuckets,
            "certificates": certResults
        }
    except Exception as e:
        return {"error": str(e)}
