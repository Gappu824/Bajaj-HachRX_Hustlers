# Use an official lightweight Python image
FROM python:3.12-slim

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set environment variables for better networking
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Install minimal system dependencies for networking and SSL
RUN apt-get update && apt-get install -y \
    # Essential networking and SSL utilities
    curl \
    wget \
    ca-certificates \
    openssl \
    # Build tools needed for some Python packages
    build-essential \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Update CA certificates
RUN update-ca-certificates

# Set the working directory inside the container
WORKDIR /code

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy just the requirements file to leverage Docker layer caching
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your application code into the container
COPY ./app /code/app

# Change ownership of the working directory to the non-root user
RUN chown -R appuser:appuser /code

# Switch to non-root user
USER appuser

# Expose port 8080 (Google Cloud Run default)
EXPOSE 8080

# Health check to ensure the application is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Command to run your application
# Google Cloud Run expects applications to listen on port 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]