from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import uvicorn
import io
import re
import os
import pandas as pd

app = FastAPI()

# CORS for frontend JS to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# PART A: Text-based TF-IDF + RandomForest (existing)
# ---------------------------
docs = [
    "Math, Physics, Programming, Data Structures",
    "Networking, Security, Database, Computer Systems",
    "UI Design, Multimedia, Graphic Design, HTML CSS"
]
text_labels = ["Software Engineer", "Network Specialist", "UI/UX Designer"]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(docs)

text_model = RandomForestClassifier(random_state=42)
text_model.fit(X_train, text_labels)

# ---------------------------
# PART B: Structured-data model (trained from cs_students.csv if available)
# ---------------------------
structured_model = None
label_encoders = {}
target_le = None
structured_features = ["GPA", "Python", "SQL", "Java", "Interested Domain"]

def build_structured_model(csv_path: str = "cs_students.csv"):
    global structured_model, label_encoders, target_le
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        # Ensure expected columns exist
        for col in structured_features + ["Future Career"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column in CSV: {col}")

        data = df.copy()
        # Encode categorical features
        label_encoders = {}
        for col in structured_features:
            if data[col].dtype == "object":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le

        # Encode target
        target_le = LabelEncoder()
        data["Future Career"] = target_le.fit_transform(data["Future Career"].astype(str))

        X = data[structured_features]
        y = data["Future Career"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"Structured model trained from {csv_path} (accuracy on holdout ~ {acc*100:.2f}%)")

        structured_model = model
        # set globals
        globals()["label_encoders"] = label_encoders
        globals()["target_le"] = target_le
        return model
    except Exception as e:
        print("Could not build structured model:", e)
        return None

# Try to build structured model at startup (silent if csv missing)
build_structured_model()

# ---------------------------
# OCR helper: Extract text from image bytes
# ---------------------------
def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return text.strip()

# ---------------------------
# Parse structured features (GPA + skills + interest/domain) from OCR text
# ---------------------------
def parse_structured_features(extracted_text: str) -> dict:
    keywords = ["Python", "SQL", "Java", "C++", "HTML", "CSS", "Networking", "Security", "Data"]
    skills = {}
    for kw in keywords:
        skills[kw] = "Strong" if re.search(r'\b' + re.escape(kw) + r'\b', extracted_text, re.I) else "Weak"

    # GPA detection (simple heuristic)
    gpa_match = re.search(r'\b(?:GPA|CGPA|Grade Point Average)[\s:]*([0-4](?:\.\d{1,2})?)\b', extracted_text, re.I)
    gpa = float(gpa_match.group(1)) if gpa_match else None

    # Interested domain detection (simple keyword mapping)
    domain_map = {
        "Data Science": ["data science", "machine learning", "ml", "data"],
        "Networking": ["network", "networking", "router", "tcp", "udp"],
        "Security": ["security", "cyber", "cybersecurity", "penetration"],
        "UI/UX": ["ui", "ux", "design", "graphic", "multimedia"],
        "Web Development": ["html", "css", "javascript", "web", "frontend", "backend"],
        "Systems": ["operating system", "computer systems", "systems", "os"]
    }
    detected_domain = None
    lower_text = extracted_text.lower()
    for domain, keys in domain_map.items():
        for k in keys:
            if k in lower_text:
                detected_domain = domain
                break
        if detected_domain:
            break

    return {"GPA": gpa, "skills": skills, "Interested Domain": detected_domain}

# ---------------------------
# Helper: Encode structured example for prediction
# ---------------------------
def encode_structured_example(parsed: dict):
    if structured_model is None or not label_encoders:
        raise RuntimeError("Structured model or encoders not available")

    row = {}
    # GPA
    row["GPA"] = parsed.get("GPA") if parsed.get("GPA") is not None else 0.0

    # For skill columns that are expected as categorical in CSV, attempt to encode
    for col in ["Python", "SQL", "Java"]:
        val = parsed.get("skills", {}).get(col, "Weak")
        le = label_encoders.get(col)
        if le is not None:
            # If unseen label, fallback to most frequent class index 0
            if val in le.classes_:
                row[col] = int(le.transform([val])[0])
            else:
                row[col] = 0
        else:
            # if encoder not available (CSV numeric), try to map simple heuristics
            row[col] = 2 if val.lower() == "strong" else (1 if val.lower() == "average" else 0)

    # Interested Domain
    domain = parsed.get("Interested Domain") or "Unknown"
    le_domain = label_encoders.get("Interested Domain")
    if le_domain is not None:
        if domain in le_domain.classes_:
            row["Interested Domain"] = int(le_domain.transform([domain])[0])
        else:
            # fallback: try to match by substring to existing classes
            chosen = None
            for cls in le_domain.classes_:
                if cls.lower() in domain.lower() or domain.lower() in cls.lower():
                    chosen = cls
                    break
            row["Interested Domain"] = int(le_domain.transform([chosen])[0]) if chosen else 0
    else:
        # if no encoder, set 0
        row["Interested Domain"] = 0

    return pd.DataFrame([row], columns=structured_features)

# ---------------------------
# OCR + Prediction endpoint
# ---------------------------
@app.post("/predict/")
async def predict_career(tor_file: UploadFile = File(...)):
    try:
        contents = await tor_file.read()
        extracted_text = extract_text_from_image_bytes(contents)
        print(f"Extracted Text: {extracted_text}")

        structured_parsed = parse_structured_features(extracted_text)

        # Text-based prediction (TF-IDF + text_model)
        X_input = vectorizer.transform([extracted_text])
        text_prediction = text_model.predict(X_input)[0]

        response = {
            "career_prediction_text_model": text_prediction,
            "extracted_text": extracted_text,
            "structured_parsed": structured_parsed
        }

        # If structured model available, try structured prediction
        try:
            if structured_model is not None:
                new_student_df = encode_structured_example(structured_parsed)
                pred_idx = structured_model.predict(new_student_df)[0]
                structured_pred = target_le.inverse_transform([pred_idx])[0] if target_le is not None else str(pred_idx)
                response["career_prediction_structured_model"] = structured_pred
        except Exception as e:
            # don't fail entire endpoint if structured prediction fails
            response["structured_prediction_error"] = str(e)

        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional: run with `python main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
