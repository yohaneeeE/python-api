

# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Tesseract OCR
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pillow pytesseract scikit-learn pandas

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
# (Render expects CMD, not ENTRYPOINT)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
