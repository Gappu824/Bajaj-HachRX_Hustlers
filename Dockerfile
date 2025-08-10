# Dockerfile - Optimized for performance and reliability
FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system dependencies in optimized order
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    # For document processing
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    # For Tika (Java dependency)
    default-jre \
    # For OCR with extended language support
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-mal \
    tesseract-ocr-hin \
    tesseract-ocr-osd \
    # For Excel and scientific computing
    libopenblas-dev \
    gfortran \
    # For image processing
    libjpeg-dev \
    libpng-dev \
    # For PDF processing
    poppler-utils \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app_user

# Copy requirements first (for better caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('punkt_tab'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('omw-1.4')"

# Copy application code
COPY ./app /code/app

# Create necessary directories with proper permissions
RUN mkdir -p /code/.cache /code/logs /tmp/uploads && \
    chown -R app_user:app_user /code && \
    chmod -R 755 /code

# Switch to non-root user
USER app_user

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/code
ENV HF_HOME=/code/.cache/huggingface
ENV TRANSFORMERS_CACHE=/code/.cache/transformers

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application with better configuration
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]