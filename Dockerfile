FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure dataset and model artifacts are pre-generated
RUN python generate_dataset.py
RUN python fraud_detection.py

# Expose default HTTP port
EXPOSE 5000 7860

# Default environment variables
ENV PORT=5000
ENV HOST=0.0.0.0

CMD ["python", "app.py"]
