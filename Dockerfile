# Start from a slim Linux base with Python already installed
FROM python:3.11-slim

# Install SUMO's headless package (eclipse-sumo) via pip,
# plus any system libs it needs to run without a GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir eclipse-sumo traci

# Set a working directory inside the container
WORKDIR /app

# Copy a small test script in
COPY docker-test.py .

# Default command when the container starts
CMD ["python", "docker-test.py"]