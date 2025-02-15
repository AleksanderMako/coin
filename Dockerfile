# Use verified ARM64 compatible CPU image
FROM arm64v8/python:3.9-slim-bullseye
RUN pip install torch==2.3.0 --extra-index-url https://download.pytorch.org/whl/cpu
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

# Copy application files
COPY . .

# Set default command
# CMD ["python", "weights_collector.py"]
CMD ["python", "main_cifar.py","-iid","5","-nl","5","-lss","28","-ld","iid_5_nl5_lss28_ni100","-ni","100"]
