FROM python:3.12-slim

# Install dependencies for network access and SSL
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Set environment variables
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the web service on container startup
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
