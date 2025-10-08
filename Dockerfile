FROM python:3.10-slim
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "decisiontree_api:app", "--host", "0.0.0.0", "--port", "8000"]

