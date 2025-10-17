FROM python:3.10-slim

WORKDIR /app
COPY . /app
COPY cs_students.csv /app/cs_students.csv   # ✅ ensure CSV is copied

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "decisiontree_api:app", "--host", "0.0.0.0", "--port", "8000"]
