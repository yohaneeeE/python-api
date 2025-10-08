FROM python:3.10-slim

RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn decisiontree_api:app --host 0.0.0.0 --port ${PORT}"]
