PIL import Image

# ---------------------------
# OCR: Extract text from TOR image
# ---------------------------
def extract_text_from_image(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text.strip()

# Example: Load a scanned TOR image
tor_text = extract_text_from_image("student_tor.png")
print("Extracted Text from TOR Image:\n", tor_text)

# ---------------------------
# (Optional) Convert OCR text into structured features
# ---------------------------
# Here you need to parse subjects/skills/grades from text
# For example, check if keywords like "Java", "SQL", "Python" appear:
skills = {
    "Python": "Strong" if "Python" in tor_text else "Weak",
    "SQL": "Strong" if "SQL" in tor_text else "Weak",
    "Java": "Strong" if "Java" in tor_text else "Weak"
}

# Example: Create a student record for prediction
new_student = pd.DataFrame([{
    "GPA": 3.4,  # if GPA detected in text, fill here
    "Python": label_encoders["Python"].transform([skills["Python"]])[0],
    "SQL": label_encoders["SQL"].transform([skills["SQL"]])[0],
    "Java": label_encoders["Java"].transform([skills["Java"]])[0],
    "Interested Domain": label_encoders["Interested Domain"].transform(["Software Development"])[0]
}])

# Predict using trained Decision Tree
prediction = clf.predict(new_student)
predicted_career = target_le.inverse_transform(prediction)[0]
print("\nCareer Guidance Recommendation from TOR Image:", predicted_career)