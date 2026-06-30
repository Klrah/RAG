# Use Python 3.12 to ensure full compatibility with ML libraries like faiss-cpu
FROM python:3.12-slim

# Set environment variables for optimized Python execution
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Download the latest uv binary from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only the requirements file first to cache the dependency layer
COPY requirements.txt .

# Install dependencies using uv into the container's system Python
# This is much faster than pip and avoids creating unnecessary virtual environments inside Docker
# Change this line in your Dockerfile:
RUN uv pip install --system --no-cache -v -r requirements.txt

# Copy the rest of the application files (app.py, guardrails.py, ingest.py, and the data/indices folders)
COPY . .

# Expose the port Streamlit uses
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]