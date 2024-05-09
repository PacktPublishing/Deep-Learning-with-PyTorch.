# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port number that the FastAPI application should run on
EXPOSE 9000

# Command to run the FastAPI application when the container starts
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000"]