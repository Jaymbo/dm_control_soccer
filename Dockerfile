# Distributed Optuna Worker for Curriculum MAPPO Soccer
# Builds a self-contained container that connects to a central Optuna storage
# and pulls training trials until no more are available.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for MuJoCo, dm-control, rendering and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-dev \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    libglib2.0-0 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency file first for better Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies. PyTorch CPU wheel is used as default fallback.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project code
COPY . /workspace/

# Run as non-root user for better security
RUN useradd -m -u 1000 worker && chown -R worker:worker /workspace
USER worker

# Default: run the worker entrypoint
ENTRYPOINT ["python", "-u", "worker_entrypoint.py"]
CMD ["--n-trials", "1000000", "--timeout", "0"]
