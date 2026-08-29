FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire repository into the image
COPY . /app/

# Ensure runtime directories exist
RUN mkdir -p /app/database /app/models

EXPOSE 8000

# Start FastAPI backend pointing to app.py inside backend/
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]