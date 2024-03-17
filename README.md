# API для проекта по предсказанию индекса качества водзуха

## Overview

API создано для получения индекса качества воздуха разными способами.

## Getting Started

### Prerequisites

- Docker
- Python 3.11 (при запуске без Docker)

### Running

1. **Building the Docker Compose**

   ```bash
   docker compose build

2. **Running the Docker Compose**

   ```bash
   docker compose up

3. **(OR) Running without Docker Compose**

   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install Redis
   sudo apt-get install redis-server
   
   # Run Redis
   redis-server
   
   cd app
   uvicorn main:app --reload

API теперь доступно http://localhost.

