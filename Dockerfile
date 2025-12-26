# 🐳 Dockerfile for PSX-up Fortune Teller
# Built for E2B Sandboxes & Cloud Deployments

FROM python:3.9-slim

# Install system dependencies (needed for lightgbm/xgboost)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directory for persistence
RUN mkdir -p data

# Expose FastAPI port
EXPOSE 8000

# Run the backend
CMD ["python", "backend/main.py"]
