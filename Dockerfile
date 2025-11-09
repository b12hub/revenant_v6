# Start from a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the app port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
