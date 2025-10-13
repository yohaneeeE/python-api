import re
import io
import asyncio
from typing import Dict
from collections import defaultdict

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

# Optional Gemini integration
try:
    from vertexai.preview.generative_models import GenerativeModel
    client = GenerativeModel("gemini-1.5-flash")
except Exception:
    client = None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- OCR CLEANING UTILS --------------------

TEXT_FIXES = {
    "lT": "IT",
    "l1": "IT",
    "Il": "II",
    "ln": "In",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}

REMOVE_LIST = ["Passed", "FAILED", "INC", "Rem.", "Rem", "Remv", "N/A", "n/a", "None"]


def normalize_subject(subj: str) -> str:
    subj = subj.strip()
    for old, new in TEXT_FIXES.items():
        subj = subj.replace(old, new)
    subj = re.sub(r"[^A-Za-z0-9\s\-]", "", subj)
    subj = re.sub(r"\s{2,}", " ", subj)
    return subj.strip()


def snap_to_valid_grade(val):
    if val is None:
        return None
    for target in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0]:
        if abs(val - target) < 0.05:
            return target
    return None


# -------------------- GEMINI IMPROVEMENT UTILS --------------------

async def gemini_refine_text(category: str, text: str) -> str:
    """
    category: 'subjects', 'skills', or 'prediction'
    """
    if not client or not text.strip():
        return text.strip()

    prompts = {
        "subjects": f"""
        Clean and improve the following extracted academic subjects:
        - Fix OCR typos and capitalization.
        - Standardize course titles.
        - Return them as a clean comma-separated list, no markdown.
        Input:
        {text}
        """,
        "skills": f"""
        Clean and standardize the following extracted technical skills:
        - Fix spelling and capitalization.
        - Merge duplicates if any.
        - Return as a comma-separated list (no markdown).
        Input:
        {text}
        """,
        "prediction": f"""
        You are a professional career counselor AI.
        Improve the following career prediction summary:
        - Fix grammar, capitalization, and sentence structure.
        - Add 2–3 relevant career or learning recommendations.
        - Keep it formal and use plain text only (no markdown or emojis).
        Input:
        {text}
        """
    }

    try:
        response = await asyncio.to_thread(client.generate_content, prompts[category])
        return response.text.strip()
    except Exception:
        return text.strip()


# -------------------- SUBJECT & GRADE EXTRACTION --------------------

def extractSubjectGrades(ocr_text: str) -> Dict[str, float]:
    lines = ocr_text.split("\n")
    bucket_grades = defaultdict(list)
    all_subjects = []

    subjectGroups = {
        "ai_ml": ["ai", "machine", "intelligence", "learning", "python", "neural", "data mining"],
        "database": ["sql", "database", "dbms", "data management"],
        "programming": ["java", "oop", "object oriented", "programming", "software"],
    }

    for line in lines:
        clean = re.sub(r"[\t:]+", " ", line).strip()
        if not clean:
            continue

        parts = clean.split()
        if len(parts) < 2:
            continue

        # detect numeric tokens
        float_tokens = []
        for i, tok in enumerate(parts):
            cleaned = re.sub(r"[^0-9.]", "", tok)
            if not cleaned:
                continue
            try:
                val = float(cleaned)
                if 1.0 <= val <= 5.0:
                    float_tokens.append((i, tok, val))
            except:
                continue

        if not float_tokens:
            continue

        gradeVal = snap_to_valid_grade(float_tokens[-1][2])
        if gradeVal is None:
            continue

        subjTokens = parts[: float_tokens[-1][0]]
        subjDesc = normalize_subject(" ".join(subjTokens))
        if not subjDesc or len(subjDesc) < 3:
            continue

        all_subjects.append(subjDesc)
        lowdesc = subjDesc.lower()

        # assign to relevant category
        if any(k in lowdesc for k in subjectGroups["ai_ml"]):
            bucket_grades["Python"].append(gradeVal)
        elif any(k in lowdesc for k in subjectGroups["database"]):
            bucket_grades["SQL"].append(gradeVal)
        elif any(k in lowdesc for k in subjectGroups["programming"]):
            bucket_grades["Java"].append(gradeVal)

    # compute average grades per skill
    finalBuckets = {}
    for k in ("Python", "SQL", "Java"):
        if bucket_grades[k]:
            finalBuckets[k] = round(sum(bucket_grades[k]) / len(bucket_grades[k]), 2)
        else:
            finalBuckets[k] = 3.0

    return {"subjects": all_subjects, "buckets": finalBuckets}


# -------------------- CAREER PREDICTION --------------------

def predictCareer(buckets: Dict[str, float]) -> str:
    if not buckets:
        return "No subjects detected."

    scores = {
        "AI Engineer": 5 - buckets["Python"],
        "Data Analyst": 5 - ((buckets["SQL"] + buckets["Python"]) / 2),
        "Software Developer": 5 - ((buckets["Java"] + buckets["Python"]) / 2),
    }

    best_career = max(scores, key=scores.get)
    explanation = (
        f"Based on your academic performance in technical subjects, "
        f"your strongest potential aligns with the role of {best_career}. "
        f"Skill performance summary: {buckets}."
    )
    return explanation


# -------------------- FASTAPI ROUTES --------------------

@app.get("/")
async def root():
    return {"message": "Career Prediction API with Gemini enhancement is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Handle PDF or image upload, extract grades, and predict career with Gemini cleanup."""
    try:
        imageBytes = await file.read()

        # Handle both PDF and image
        if file.filename.lower().endswith(".pdf"):
            images = convert_from_bytes(imageBytes)
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
        else:
            img = Image.open(io.BytesIO(imageBytes))
            text = await asyncio.to_thread(pytesseract.image_to_string, img)

        extracted = extractSubjectGrades(text)
        buckets = extracted["buckets"]
        subjects_text = ", ".join(extracted["subjects"])

        # Raw prediction
        raw_prediction = predictCareer(buckets)

        # Apply Gemini refinements
        refined_subjects = await gemini_refine_text("subjects", subjects_text)
        refined_skills = await gemini_refine_text("skills", ", ".join(buckets.keys()))
        improved_prediction = await gemini_refine_text("prediction", raw_prediction)

        return {
            "status": "success",
            "file": file.filename,
            "subjects_raw": subjects_text,
            "subjects_refined": refined_subjects,
            "skills_refined": refined_skills,
            "buckets": buckets,
            "career_prediction": improved_prediction
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# -------------------- MAIN --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
