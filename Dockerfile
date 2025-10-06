# ===========================
# 1. Use official Python base
# ===========================
FROM python:3.11-slim

# ===========================
# 2. Set working directory
# ===========================
WORKDIR /app

# ===========================
# 3. Copy dependency files
# ===========================
COPY requirements.txt .

# ===========================
# 4. Install dependencies
# ===========================
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ===========================
# 5. Copy all app files
# ===========================
COPY . .

# ===========================
# 6. Expose FastAPI default port
# ===========================
EXPOSE 8000

# ===========================
# 7. Command to run the app
# ===========================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
