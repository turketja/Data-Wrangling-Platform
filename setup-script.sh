#!/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p notebooks
mkdir -p ui
mkdir -p chatbot

# Copy files to their respective directories
cp docker-compose.yml ./

# UI files
cp ui/Dockerfile ui/
cp ui/requirements.txt ui/
cp ui/app.py ui/

# Chatbot files
cp chatbot/Dockerfile chatbot/
cp chatbot/requirements.txt chatbot/
cp chatbot/app.py chatbot/

echo "Starting containers..."
docker-compose up -d

echo "Waiting for services to be available..."
sleep 10

echo "Setup complete! Access your data analysis platform at:"
echo "  - Main UI: http://localhost:8501"
echo "  - JupyterLab: http://localhost:8888"
echo "  - MinIO: http://localhost:9001"
echo ""
echo "The system is now ready to use. You can upload data files and start analyzing them."
