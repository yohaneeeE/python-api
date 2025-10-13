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
import cv2
import numpy as np

# ---------------------------
# Utility Helpers
# ---------------------------
def clean_for_json(data):
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_for_json(x) for x in data]
    if isinstance(data, (str, int, float)) or data is None:
        return data
    return str(data)


def preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess an image to improve OCR accuracy."""
    img_cv = np.array(img.convert("L"))  # grayscale
    img_cv = cv2.medianBlur(img_cv, 3)
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        thresh = cv2.warpAffine(thresh, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)

    thresh = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(thresh)


# ---------------------------
# Gemini Client
# ---------------------------
client = genai.Client()  # reads GEMINI_API_KEY from environment


async def improveSubjectsWithGemini(subjects: dict, mappedSkills: dict):
    prompt = f"""
You are an expert career counselor AI. Here are the subjects and skill levels:
{subjects} with skills {mappedSkills}.

Please:
1. Fix misspellings and normalize subject names.
2. For each subject, provide "level" and a short "suggestion".
3. Suggest top 3–5 career paths with confidence scores and relevant certificates.

Return only JSON in this format:
{{
  "subjects": {{"original": "corrected"}},
  "skills": {{"subject": {{"level": "...", "suggestion": "..."}}}},
  "career_options": [{{"career": "...", "confidence": 90, "suggestion": "...", "certificates": ["..."]}}]
}}
"""
    try:
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )
        parsed = json.loads(response.candidates[0].content.parts[0].text)
        updatedSubjects = parsed.get("subjects", {})
        updatedSkills = parsed.get("skills", {})
        careerOptions = parsed.get("career_options", [])

        # Clean updatedSkills
        cleanedSkills = {}
        for subj, val in updatedSkills.items():
            if isinstance(val, dict):
                cleanedSkills[subj] = {
                    "level": val.get("level", "Unknown"),
                    "suggestion": val.get("suggestion", "")
                }
            else:
                cleanedSkills[subj] = {"level": str(val), "suggestion": ""}
        return updatedSubjects, cleanedSkills, careerOptions
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return subjects, {k: {"level": v, "suggestion": ""} for k, v in mappedSkills.items()}, []


# ---------------------------
# Tesseract Path
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

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
# FastAPI Setup
# ---------------------------
app = FastAPI(title="Career Prediction API (TOR/COG + Certificates 🚀)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

# ---------------------------
# Helper Mappings
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
        clean = re.sub(r"[^\w\.\-\s]", " ", raw_line)
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 2:
            continue

        float_tokens = []
        for i, tok in enumerate(parts):
            tok_clean = re.sub(r"[^0-9.]", "", tok)
            if not tok_clean:
                continue
            try:
                float_tokens.append((i, float(tok_clean)))
            except:
                continue

        if not float_tokens:
            continue

        idx, gradeVal = float_tokens[0]
        subjDesc = " ".join(parts[:idx]).strip().title()
        gradeVal = snap_to_valid_grade(gradeVal)
        if not subjDesc or len(subjDesc) < 3:
            continue

        mappedSkills[subjDesc] = grade_to_level(gradeVal)
        subjects_structured.append({"description": subjDesc, "grade": gradeVal})
        rawSubjects[subjDesc] = gradeVal
        normalizedText[subjDesc] = subjDesc

    finalBuckets = {k: 3.0 for k in ["Python", "SQL", "Java"]}
    return subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets


# ---------------------------
# Career Prediction
# ---------------------------
def predictCareerWithSuggestions(finalBuckets, normalizedText, mappedSkills):
    dfInput = pd.DataFrame([{
        "Python": finalBuckets["Python"],
        "SQL": finalBuckets["SQL"],
        "Java": finalBuckets["Java"]
    }])
    proba = model.predict_proba(dfInput)[0]
    careers = [{"career": targetEncoder.inverse_transform([i])[0],
                "confidence": round(float(p)*100,2)} for i,p in enumerate(proba)]
    careers = sorted(careers, key=lambda x: x["confidence"], reverse=True)[:3]

    for c in careers:
        c["suggestion"] = "Focus on IT-related subjects."
        c["certificates"] = ["AWS", "SQL", "Java"]
    return careers


# ---------------------------
# Certificate Analysis
# ---------------------------
def analyzeCertificates(certFiles: List[UploadFile]):
    results = []
    for cert in certFiles:
        certName = cert.filename.lower()
        if "aws" in certName:
            msg = "Your AWS cert strengthens Cloud Architect path."
        elif "ccna" in certName:
            msg = "Your CCNA boosts Networking opportunities."
        else:
            msg = f"Certificate '{cert.filename}' adds career value."
        results.append({"file": cert.filename, "suggestions": [msg]})
    return results


# ---------------------------
# API Route
# ---------------------------
@app.post("/predict")
async def ocrPredict(file: UploadFile = File(...), certificateFiles: List[UploadFile] = File(None)):
    try:
        imageBytes = await file.read()
        img = Image.open(io.BytesIO(imageBytes))
        img = preprocess_image(img)

        text = await asyncio.to_thread(pytesseract.image_to_string, img, lang="eng", config="--psm 6")
        if len(text.strip()) < 10:
            text = await asyncio.to_thread(pytesseract.image_to_string, img)

        subjects_structured, rawSubjects, normalizedText, mappedSkills, finalBuckets = extractSubjectGrades(text)

        if not mappedSkills:
            print("⚠️ No subjects detected — skipping Gemini enhancement.")
            updatedSubjects, updatedSkills, careerOptions = normalizedText, mappedSkills, []
        else:
            updatedSubjects, updatedSkills, careerOptions = await improveSubjectsWithGemini(normalizedText, mappedSkills)

        mappedLevels = {k: (v["level"] if isinstance(v, dict) else v) for k, v in updatedSkills.items()}
        careerOptions = predictCareerWithSuggestions(finalBuckets, updatedSubjects, mappedLevels)

        if not careerOptions:
            careerOptions = [{"career": "General Studies", "confidence": 50.0,
                              "suggestion": "Add more subjects or improve grades.",
                              "certificates": ["General IT Certifications"]}]

        certResults = analyzeCertificates(certificateFiles or []) if certificateFiles else [{"info": "No certificates uploaded"}]

        print("🧾 OCR Extracted Text:", text[:300])
        print("🧠 Detected subjects:", list(mappedSkills.keys()))

        return {
            "careerPrediction": careerOptions[0]["career"],
            "careerOptions": clean_for_json(careerOptions),
            "subjects_structured": clean_for_json(subjects_structured),
            "rawSubjects": list(rawSubjects.items()),
            "normalizedText": clean_for_json(normalizedText),
            "mappedSkills": clean_for_json(updatedSkills),
            "finalBuckets": clean_for_json(finalBuckets),
            "certificates": certResults
        }

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return {"error": str(e)}
