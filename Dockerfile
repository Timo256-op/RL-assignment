# --- STAGE 1: Builder Stage (To install dependencies efficiently) ---
FROM python:3.10-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create the working directory
WORKDIR $APP_HOME

# Copy only the requirements file first to take advantage of Docker caching
COPY requirements.txt .

# Install system dependencies required for building Python packages
#RUN apk add --no-cache \
 #   build-base \
  #  libsdl2-dev

# Install Python dependencies in a virtual environment for better isolation
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Production Runtime Stage (Minimal image for deployment) ---
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV MODEL_PATH=/app/saved_models/ppo_teacher_alloc.zip

# Create the working directory
WORKDIR $APP_HOME

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# Copy scripts/console entrypoints (uvicorn, streamlit CLI, etc.)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the rest of the application code
COPY . $APP_HOME

# Ensure the trained model is available in the image
RUN mkdir -p $APP_HOME/saved_models/
COPY saved_models/ppo_teacher_alloc.zip $APP_HOME/saved_models/

# Expose the port used by the FastAPI application
EXPOSE 8000 8501

# Command to run the application using Uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
