FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p logs models data/cs-test data/cs-prod

# Expose the port
EXPOSE 8080

# Run unit tests then start the application
CMD python run_tests.py && gunicorn --bind 0.0.0.0:8080 app:app
