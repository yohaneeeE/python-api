from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import uvicorn
import io

app = FastAPI()

# CORS for frontend JS to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample training data
docs = [
    "Math, Physics, Programming, Data Structures",
    "Networking, Security, Database, Computer Systems",
    "UI Design, Multimedia, Graphic Design, HTML CSS"
]
labels = ["Software Engineer", "Network Specialist", "UI/UX Designer"]

# Train model
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(docs)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, labels)

# OCR + Prediction
@app.post("/predict/")
async def predict_career(tor_file: UploadFile = File(...)):
    try:
        # Read and convert image
        contents = await tor_file.read()
        img = Image.open(io.BytesIO(contents))
        
        # OCR
        extracted_text = pytesseract.image_to_string(img).strip()
        print(f"Extracted Text: {extracted_text}")

        # Predict
        X_input = vectorizer.transform([extracted_text])
        prediction = model.predict(X_input)[0]

        return JSONResponse(content={
            "career_prediction": prediction,
            "extracted_text": extracted_text
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

