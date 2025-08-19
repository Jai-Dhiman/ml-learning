# Safety Text Classifier - Docker Container
# Optimized for JAX/Flax training and inference

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints logs data .cache

# Set environment variables
ENV PYTHONPATH=/app/src
ENV JAX_PLATFORM_NAME=cpu
ENV TOKENIZERS_PARALLELISM=false

# Expose ports
EXPOSE 7860 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Default command (can be overridden)
CMD ["python", "demo_app.py", "--host", "0.0.0.0", "--port", "7860"]