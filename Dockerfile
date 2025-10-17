# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install OS dependencies for OCR and PDF parsing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "decisiontree_api:app", "--host", "0.0.0.0", "--port", "8000"]
