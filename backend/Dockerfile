# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
# (Add any other needed dependencies here, e.g., build-essential for C extensions)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY ./app /app/app

# Make port 8000 available to the world outside this container (same as BACKEND_PORT in .env)
# Note: This doesn't publish the port, just exposes it.
# Publishing happens in docker-compose.yml
EXPOSE 8000

# Define environment variable (optional, can be set in docker-compose.yml)
# ENV NAME World

# Run main.py when the container launches
# Use uvicorn to run the FastAPI application
# The host 0.0.0.0 makes the server accessible from outside the container
# The port should match the EXPOSE instruction and the BACKEND_PORT in .env
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 