# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to the parent directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the parent directory contents into the container at /workspace
COPY .. /workspace

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]