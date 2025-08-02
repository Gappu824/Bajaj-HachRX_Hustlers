# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# Copy just the requirements file to leverage Docker layer caching
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your application code into the container
COPY ./app /code/app

# Command to run your application.
# Google Cloud Run expects applications to listen on port 8080.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]