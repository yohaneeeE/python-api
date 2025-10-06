# Use lightweight Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy all files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
