# Install required packages if not already installed
# pip install pytesseract pillow scikit-learn pandas matplotlib

import pytesseract
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# ---------------------------
# PART 1: OCR - Extract Text from TOR
# ---------------------------
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text.strip()

# Extracted TOR text (OCR)
try:
    tor_text = extract_text_from_image("student_tor.png")
except FileNotFoundError:
    tor_text = "Math Physics Programming"  # fallback if image missing
print("📄 Extracted TOR Text:\n", tor_text)

# ---------------------------
# PART 2: Text-based Prediction (TF-IDF + Simple Model)
# ---------------------------
docs = [
    "Math, Physics, Programming, Data Structures",      # Software Engineer
    "Networking, Security, Database, Computer Systems", # Network Specialist
    "UI Design, Multimedia, Graphic Design, HTML CSS"   # UI/UX Designer
]
text_labels = ["Software Engineer", "Network Specialist", "UI/UX Designer"]

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(docs)

text_model = RandomForestClassifier(n_estimators=50, random_state=42)
text_model.fit(X_text, text_labels)

X_tor = vectorizer.transform([tor_text])
text_prediction = text_model.predict(X_tor)[0]
print("\n🧠 Predicted Career from TOR Text (OCR):", text_prediction)

# ---------------------------
# PART 3: Structured Data Model with RandomForest
# ---------------------------
df = pd.read_csv("cs_students.csv")

# Feature and Target Columns
features = ["GPA", "Python", "SQL", "Java", "Interested Domain"]
target = "Future Career"

data = df.copy()

# Encode Categorical Features
label_encoders = {}
for col in features:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
data[target] = target_le.fit_transform(data[target])

# Train-test split (no stratify, works even if some classes have 1 record)
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simple RandomForest (fast, good defaults)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"\n✅ Structured Data - Accuracy: {acc*100:.2f}%")

# ---------------------------
# Predict New Structured Student Example
# ---------------------------
try:
    new_student = pd.DataFrame([{
        "GPA": 3.6,
        "Python": label_encoders["Python"].transform(["Strong"])[0],
        "SQL": label_encoders["SQL"].transform(["Average"])[0],
        "Java": label_encoders["Java"].transform(["Weak"])[0],
        "Interested Domain": label_encoders["Interested Domain"].transform(["Data Science"])[0]
    }])

    structured_prediction = model.predict(new_student)
    structured_career = target_le.inverse_transform(structured_prediction)[0]
    print("\n🎯 Predicted Career from Structured Input:", structured_career)
except Exception as e:
    print("\n⚠️ Could not predict new student due to unseen label or error:", e)

# ---------------------------
# PART 4: Visualize One Tree from Random Forest
# ---------------------------
plt.figure(figsize=(18, 10))
plot_tree(
    model.estimators_[0],
    feature_names=X.columns,
    class_names=target_le.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("📊 Sample Decision Tree from Random Forest Model")
plt.show()
