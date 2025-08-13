# aggregate_deployment/Dockerfile

# 1. Start with an official Python base image.
# Using a slim version keeps our final image smaller.
FROM python:3.12-slim

# Ensure OpenMP runtime is available for scikit-learn wheels and keep image small
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container.
# All subsequent commands will run from this directory.
WORKDIR /app

# 3. Copy the requirements file into the container first.
# This is a Docker best practice. It allows Docker to cache the installed
# libraries unless the requirements file changes.
COPY requirements.txt .

# 4. Install the Python dependencies.
RUN python -m pip install --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of our application code and data into the container.
# This includes app.py and the models/ and data/ directories.
COPY . .

# 6. Expose the port the app runs on.
# We'll have gunicorn listen on port 5000 inside the container.
EXPOSE 5000

# 7. Define the command to run the application.
# This command starts the gunicorn server, which will serve our Flask app.
# It specifies 4 worker processes to handle requests.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]