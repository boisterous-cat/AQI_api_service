FROM python:3.11-slim

COPY ./app /app
COPY requirements.txt /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]