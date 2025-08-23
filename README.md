#  Methane Leak Detection - Complete Deployment Guide

## 📁 Project Structure

```
methane-detection-app/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── streamlit_app.py            # Main Streamlit application
├── main.py                     # FastAPI backend
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker compose for local development
├── .env.example               # Environment variables template
├── data/                      # Data directory
│   ├── 20mm/
│   ├── 25mm/
│   ├── 30mm/
│   └── 40mm/
├── models/                    # Saved models directory
├── output/                    # Output files directory
├── tests/                     # Test files
│   ├── __init__.py
│   ├── test_api.py
│   └── test_streamlit.py
└── deployment/                # Deployment configurations
    ├── streamlit_cloud.md
    ├── heroku_deploy.md
    └── aws_deploy.md
```

## 📦 requirements.txt

```txt
# Core dependencies
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
scipy>=1.9.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.15.0

# Data processing
joblib>=1.3.0
pickle5>=0.0.11; python_version < '3.8'

# Optional advanced features
umap-learn>=0.5.0
imbalanced-learn>=0.11.0

# File handling
python-multipart>=0.0.6
aiofiles>=23.2.0

# Development and testing
pytest>=7.0.0
httpx>=0.25.0
pytest-asyncio>=0.21.0

# Deployment
gunicorn>=21.0.0
```

## ⚙️ .streamlit/config.toml

```toml
[global]
developmentMode = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 🐳 Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data models output

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Default command (can be overridden)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🐳 docker-compose.yml

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app
    command: ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    
  fastapi:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - streamlit
      - fastapi
```

## 🔧 .env.example

```env
# Application settings
APP_NAME=Methane Leak Detection
APP_VERSION=1.0.0
DEBUG=false

# Model settings
TEMPERATURE_THRESHOLD=305
TARGET_LEAK_RATIO=0.4
SENSITIVITY_LEVEL=balanced

# API settings
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501

# Database (if needed for future extensions)
DATABASE_URL=sqlite:///./methane_detection.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Security (for production)
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

## 🚀 Local Development Setup

### Option 1: Direct Python Setup

```bash
# 1. Clone or create the project directory
mkdir methane-detection-app
cd methane-detection-app

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create directory structure
mkdir -p data/{20mm,25mm,30mm,40mm}
mkdir -p models output

# 6. Run Streamlit app
streamlit run streamlit_app.py

# 7. Run FastAPI (in another terminal)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd methane-detection-app

# 2. Build and run with Docker Compose
docker-compose up --build

# Access applications:
# - Streamlit: http://localhost:8501
# - FastAPI: http://localhost:8000
# - FastAPI Docs: http://localhost:8000/docs
```

## ☁️ Streamlit Cloud Deployment

### 1. Prepare Repository

Create a GitHub repository with the following structure:
```
your-repo/
├── streamlit_app.py
├── requirements.txt
├── .streamlit/config.toml
├── README.md
└── data/ (optional sample data)
```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Choose main file: `streamlit_app.py`
5. Advanced settings (optional):
   - Python version: 3.9
   - Environment variables if needed
6. Click "Deploy"

### 3. Update streamlit_app.py for Cloud

Add this configuration at the top of your Streamlit app:

```python
# For Streamlit Cloud deployment
import streamlit as st
import os

# Configure for cloud deployment
if 'streamlit' in os.