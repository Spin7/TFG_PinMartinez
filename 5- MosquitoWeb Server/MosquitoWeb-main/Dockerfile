FROM python:3.11-slim

# Install system-level GIS / image / OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure the models directory exists (models are downloaded at startup)
RUN mkdir -p models

EXPOSE 8000

# Python reads PORT from os.getenv() — no shell expansion needed
CMD ["python", "Server.py"]
