# -----------------------------
# Dockerfile for LLaMA GGUF CPU FastAPI
# -----------------------------
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        git \
        curl \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy Python dependencies
COPY requirements.txt .

# Upgrade pip & install requirements
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

