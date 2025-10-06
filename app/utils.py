import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import docx
import io
import re
from collections import OrderedDict

# Subject normalization & OCR fixes
TEXT_FIXES = {
    "advan database systems": "Advance Database Systems",
    "camper prararining": "Computer Programming",
    "purposve communication": "Purposive Communication",
    # ... add more as needed
}

REMOVE_LIST = ["student", "report of grades", "republic", "city of", "wps"]

def extract_text_from_image(file_bytes):
    img = Image.open(file_bytes)
    text = pytesseract.image_to_string(img)
    return text.strip()

def extract_text_from_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_bytes):
    doc = docx.Document(file_bytes)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file_bytes):
    return file_bytes.read().decode("utf-8", errors="ignore")

def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(file.file)
    elif filename.endswith(".pdf"):
        return extract_text_from_pdf(file.file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file.file)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(file.file)
    else:
        raise ValueError("Unsupported file type.")

def normalize_subject(desc: str) -> str:
    s = desc.lower().strip()
    for wrong, correct in TEXT_FIXES.items():
        if wrong in s:
            s = s.replace(wrong, correct)
    if any(bad in s for bad in REMOVE_LIST):
        return None
    return s.title()

def extract_subjects_and_grades(text: str):
    subjects_structured = []
    mappedSkills = {}
    bucket_grades = {"Python": [], "SQL": [], "Java": []}

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        grade = None
        try:
            for tok in reversed(tokens):
                tok_val = float(re.sub(r'[^0-9.]','', tok))
                if 1.0 <= tok_val <= 5.0:
                    grade = tok_val
                    break
        except:
            grade = 3.0

        subj_desc_raw = " ".join(tokens[:-1]) if grade else " ".join(tokens)
        subj = normalize_subject(subj_desc_raw)
        if not subj:
            continue
        subjects_structured.append({"description": subj, "grade": grade})
        # Assign skill bucket
        l = subj.lower()
        if "python" in l or "ai" in l:
            bucket_grades["Python"].append(grade)
            mappedSkills[subj] = "Strong" if grade <= 1.75 else "Average" if grade <= 2.5 else "Weak"
        elif "sql" in l or "database" in l:
            bucket_grades["SQL"].append(grade)
            mappedSkills[subj] = "Strong" if grade <= 1.75 else "Average" if grade <= 2.5 else "Weak"
        elif "java" in l or "programming" in l:
            bucket_grades["Java"].append(grade)
            mappedSkills[subj] = "Strong" if grade <= 1.75 else "Average" if grade <= 2.5 else "Weak"

    finalBuckets = {k: round(sum(v)/len(v),2) if v else 3.0 for k,v in bucket_grades.items()}
    return subjects_structured, mappedSkills, finalBuckets
