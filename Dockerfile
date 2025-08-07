# Dockerfile - Enhanced with all dependencies
# FROM python:3.11-slim
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    # For document processing
    libxml2-dev \
    libxslt-dev \
    libmagic1 \
    # For OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-chi-sim \
    # For PDF processing
    poppler-utils \
    # For image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # For Excel processing
    libopenblas-dev \
    gfortran \
    # Java for Tika
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Install Tika server
RUN mkdir -p /opt/tika && \
    wget -q https://archive.apache.org/dist/tika/2.9.0/tika-server-standard-2.9.0.jar \
    -O /opt/tika/tika-server.jar
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-11-jre-headless libopenjp2-7 && rm -rf /var/lib/apt/lists/*    

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY ./app /code/app

# Create cache directory
RUN mkdir -p /code/.cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV JAVA_HOME=/usr/lib/jvm/default-java

# # Create startup script
# # Replace the startup script section in Dockerfile with:
# RUN echo '#!/bin/bash\n\
# java -jar /opt/tika/tika-server.jar --port 9998 > /dev/null 2>&1 &\n\
# sleep 10\n\
# exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1\n\
# ' > /code/start.sh && chmod +x /code/start.sh
# Replace the startup script section in Dockerfile with:
RUN echo '#!/bin/bash\n\
echo "Starting Tika server..."\n\
java -jar /opt/tika/tika-server.jar --port 9998 > /var/log/tika.log 2>&1 &\n\
TIKA_PID=$!\n\
echo "Waiting for Tika to be ready..."\n\
for i in {1..30}; do\n\
    if curl -s http://localhost:9998/tika > /dev/null 2>&1; then\n\
        echo "Tika server ready!"\n\
        break\n\
    fi\n\
    sleep 1\n\
done\n\
if ! curl -s http://localhost:9998/tika > /dev/null 2>&1; then\n\
    echo "Error: Tika server failed to start"\n\
    cat /var/log/tika.log\n\
    exit 1\n\
fi\n\
echo "Starting FastAPI application..."\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1 --timeout-keep-alive 120\n\
' > /code/start.sh && chmod +x /code/start.sh

# Run application
CMD ["/code/start.sh"]