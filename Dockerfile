FROM python:3.10-slim

RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Install Tesseract OCR and clean up
RUN apt-get update && apt-get install -y tesseract-ocr libgl1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn decisiontree_api:app --host 0.0.0.0 --port ${PORT}"]

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Render's port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["sh", "-c", "uvicorn decisiontree_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
