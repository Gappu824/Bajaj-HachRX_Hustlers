#!/bin/bash
# start_services.sh - Start all required services

echo "Starting Enhanced RAG Pipeline Services..."

# Start Tika server in background
echo "Starting Apache Tika server..."
java -jar /opt/tika/tika-server.jar --port 9998 &
TIKA_PID=$!

# Wait for Tika to be ready
echo "Waiting for Tika to be ready..."
sleep 10

# Check if Tika is running
if ! curl -s http://localhost:9998/tika > /dev/null; then
    echo "Error: Tika server failed to start"
    exit 1
fi

echo "Tika server ready!"

# Start the FastAPI application
echo "Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1

# Cleanup on exit
trap "kill $TIKA_PID" EXIT