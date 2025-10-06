import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_PATH = "app/models/cs_students.csv"

# Structured Model
df = pd.read_csv(CSV_PATH)
features = ["Python", "SQL", "Java"]
target = "Future Career"
label_encoders = {}

data = df.copy()
for col in features:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

target_le = LabelEncoder()
data[target] = target_le.fit_transform(data[target])

X = data[features]
y = data[target]
structured_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
structured_model.fit(X, y)

# Text-based Model
docs = [
    "Math, Physics, Programming, Data Structures",
    "Networking, Security, Database, Computer Systems",
    "UI Design, Multimedia, Graphic Design, HTML CSS"
]
labels = ["Software Engineer", "Network Specialist", "UI/UX Designer"]
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(docs)
text_model = RandomForestClassifier(n_estimators=50, random_state=42)
text_model.fit(X_text, labels)
