# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY . .

# Verify files (optional, for debugging)
RUN ls -la

# Set default command to run your script
CMD ["python", "experiments_cpu.py"]