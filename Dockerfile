# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies required for the new packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file to leverage Docker layer caching
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Download NLTK data that we need
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy your application code into the container
COPY ./app /code/app

# Create directory for utils if cache warmup is needed
RUN mkdir -p /code/app/utils

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Command to run your application.
# Google Cloud Run expects applications to listen on port 8080.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]