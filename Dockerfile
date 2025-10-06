# ================================
# Stage 1: Base Image
# ================================
FROM python:3.10-slim

# Prevents Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ================================
# Install system dependencies
# ================================
# - tesseract-ocr → OCR engine
# - libtesseract-dev + libleptonica-dev → needed for pytesseract
# - build tools for scikit-learn dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Copy and install Python dependencies
# ================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Copy project files
# ================================
COPY . .

# ================================
# Environment variables
# ================================
ENV PORT=8000
ENV HOST=0.0.0.0

# ================================
# Run the FastAPI app with Uvicorn
# ================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

