# ==============================
# Base Image
# ==============================
FROM python:3.10-slim

# ==============================
# Set working directory
# ==============================
WORKDIR /app

# ==============================
# Install System Dependencies
# - Tesseract OCR
# - poppler-utils (for pdfplumber)
# - libglib2.0 & fonts for PaddleOCR
# - libgl1 (optional: fixes cv2 libGL issue)
# ==============================
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Copy dependency list
# ==============================
COPY requirements.txt .

# ==============================
# Install Python packages
# ==============================
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# Copy source code
# ==============================
COPY . .

# ==============================
# Expose FastAPI port
# ==============================
EXPOSE 8000

# ==============================
# Run app with Uvicorn
# ==============================
CMD ["uvicorn", "decisiontree_api:app", "--host", "0.0.0.0", "--port", "8000"]
