# Use an official Python 3.9 image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Tell Docker that the container will listen on port 8000
EXPOSE 8000

# The command to run your Flask app when the container starts
CMD ["python", "app.py"]