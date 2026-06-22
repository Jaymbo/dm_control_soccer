#!/bin/bash
# ============================================
# Worker Setup Script
# ============================================
# Usage: ./scripts/setup_worker.sh
#
# This script:
# 1. Installs Python dependencies
# 2. Configures environment
# 3. Starts the worker
# ============================================

set -e

echo "============================================"
echo "⚽ Multi-Agent Soccer - Worker Setup"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 1. Check Prerequisites
# ============================================
echo -e "${YELLOW}[1/5] Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python found: $PYTHON_VERSION${NC}"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}pip3 not found. Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3-pip
    echo -e "${GREEN}✓ pip3 installed${NC}"
else
    echo -e "${GREEN}✓ pip3 found${NC}"
fi

# Check git
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Git not found. Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y git
    echo -e "${GREEN}✓ Git installed${NC}"
else
    echo -e "${GREEN}✓ Git found${NC}"
fi

echo ""

# ============================================
# 2. Install Dependencies
# ============================================
echo -e "${YELLOW}[2/5] Installing dependencies...${NC}"

# Install system dependencies
echo -e "${YELLOW}Installing system packages...${NC}"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
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
    libpng-dev

echo -e "${GREEN}✓ System packages installed${NC}"

# Install Python dependencies
echo -e "${YELLOW}Installing Python packages (this may take 2-3 minutes)...${NC}"
pip3 install --no-cache-dir -r requirements.txt

# Detect hardware and install appropriate PyTorch
echo ""
echo -e "${YELLOW}Detecting hardware...${NC}"

if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118
elif command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✓ AMD GPU detected${NC}"
    echo -e "${YELLOW}Installing PyTorch with ROCm support...${NC}"
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/rocm6.0
else
    echo -e "${YELLOW}No GPU detected, using CPU version${NC}"
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# ============================================
# 3. Configure Environment
# ============================================
echo -e "${YELLOW}[3/5] Configuring environment...${NC}"

# Copy .env.example if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

echo ""

# ============================================
# 4. Get Connection Details
# ============================================
echo -e "${YELLOW}[4/5] Worker Configuration${NC}"
echo ""
echo -e "${BLUE}Please enter your Master Server connection details:${NC}"
echo ""
echo "You should have received these from the Master Server setup."
echo "Format: postgresql://optuna:PASSWORD@HOST:PORT/optuna_db"
echo ""

# Get OPTUNA_STORAGE
read -p "Enter OPTUNA_STORAGE URL: " OPTUNA_STORAGE_INPUT
if [ -z "$OPTUNA_STORAGE_INPUT" ]; then
    echo -e "${RED}Error: OPTUNA_STORAGE is required${NC}"
    exit 1
fi

# Get MLFLOW_TRACKING_URI
read -p "Enter MLFLOW_TRACKING_URI (or press Enter for default): " MLFLOW_INPUT
if [ -z "$MLFLOW_INPUT" ]; then
    # Try to extract from OPTUNA_STORAGE
    HOST=$(echo $OPTUNA_STORAGE_INPUT | sed -n 's/.*@\(.*\):.*/\1/p' | cut -d':' -f1)
    if [[ "$OPTUNA_STORAGE_INPUT" == *":443"* ]]; then
        MLFLOW_INPUT="http://$HOST:80"
    else
        MLFLOW_INPUT="http://$HOST:5000"
    fi
    echo -e "${YELLOW}Using default: $MLFLOW_INPUT${NC}"
fi

# Get study name
read -p "Enter study name (default: soccer_dynamic_v1): " STUDY_NAME_INPUT
STUDY_NAME=${STUDY_NAME_INPUT:-soccer_dynamic_v1}

# Get number of trials
read -p "Enter number of trials (default: infinite, press Enter for infinite): " N_TRIALS_INPUT
if [ -z "$N_TRIALS_INPUT" ]; then
    N_TRIALS_INPUT="1000000"
    INFINITE_MODE=true
else
    INFINITE_MODE=false
fi

# Update .env file
sed -i "s|OPTUNA_STORAGE=.*|OPTUNA_STORAGE=$OPTUNA_STORAGE_INPUT|" .env
sed -i "s|MLFLOW_TRACKING_URI=.*|MLFLOW_TRACKING_URI=$MLFLOW_INPUT|" .env
sed -i "s|OPTUNA_STUDY_NAME=.*|OPTUNA_STUDY_NAME=$STUDY_NAME|" .env

echo -e "${GREEN}✓ Configuration saved${NC}"
echo ""

# ============================================
# 5. Test Connection
# ============================================
echo -e "${YELLOW}[5/5] Testing connection to Master...${NC}"

# Extract host from OPTUNA_STORAGE
HOST=$(echo $OPTUNA_STORAGE_INPUT | sed -n 's/.*@\(.*\):.*/\1/p' | cut -d':' -f1)
PORT=$(echo $OPTUNA_STORAGE_INPUT | sed -n 's/.*:\(.*\)\/.*/\1/p')

echo -e "${YELLOW}Testing connection to $HOST:$PORT...${NC}"

if command -v nc &> /dev/null; then
    if nc -z -w5 $HOST $PORT &> /dev/null; then
        echo -e "${GREEN}✓ Connection to PostgreSQL successful${NC}"
    else
        echo -e "${RED}✗ Cannot connect to PostgreSQL${NC}"
        echo -e "${YELLOW}Possible issues:${NC}"
        echo "  - Cloudflare Tunnel not configured"
        echo "  - Firewall blocking connection"
        echo "  - Wrong hostname/password"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}netcat not found, skipping connection test${NC}"
fi

# Test MLflow
echo -e "${YELLOW}Testing connection to MLflow ($MLFLOW_INPUT)...${NC}"
if curl -s --connect-timeout 5 $MLFLOW_INPUT &> /dev/null; then
    echo -e "${GREEN}✓ Connection to MLflow successful${NC}"
else
    echo -e "${YELLOW}⚠️  Cannot connect to MLflow (will retry when starting)${NC}"
fi

echo ""

# ============================================
# 6. Start Worker
# ============================================
echo "============================================"
echo "🚀 Starting Worker"
echo "============================================"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Study: $STUDY_NAME"
echo "  Mode: $([ "$INFINITE_MODE" = true ] && echo "Infinite" || echo "$N_TRIALS_INPUT trials")"
echo "  Storage: $OPTUNA_STORAGE_INPUT"
echo "  MLflow: $MLFLOW_INPUT"
echo ""

# Set MuJoCo renderer (required!)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo -e "${YELLOW}Starting worker...${NC}"
echo -e "${GREEN}Logs will be shown below. Press Ctrl+C to stop.${NC}"
echo ""
echo "============================================"
echo ""

if [ "$INFINITE_MODE" = true ]; then
    python3 -u worker_entrypoint.py \
        --storage "$OPTUNA_STORAGE_INPUT" \
        --study-name "$STUDY_NAME" \
        --n-trials $N_TRIALS_INPUT \
        --infinite \
        --use-dynamic-rewards \
        --mlflow-tracking-uri "$MLFLOW_INPUT"
else
    python3 -u worker_entrypoint.py \
        --storage "$OPTUNA_STORAGE_INPUT" \
        --study-name "$STUDY_NAME" \
        --n-trials $N_TRIALS_INPUT \
        --use-dynamic-rewards \
        --mlflow-tracking-uri "$MLFLOW_INPUT"
fi
