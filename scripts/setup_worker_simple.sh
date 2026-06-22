#!/bin/bash
# ============================================
# Simple Worker Setup Script
# ============================================
# Usage: ./scripts/setup_worker_simple.sh
#
# This script:
# 1. Asks for master hostname (e.g., optuna.jasondietrich.de)
# 2. Installs dependencies
# 3. Configures connection automatically
# 4. Starts the worker with offline support
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
# 1. Get Master Hostname
# ============================================
echo -e "${YELLOW}[1/6] Master Server Configuration${NC}"
echo ""
echo -e "${BLUE}Enter your Master Server Cloudflare hostname:${NC}"
echo "Example: optuna.jasondietrich.de"
echo ""

read -p "Master Hostname: " MASTER_HOST

if [ -z "$MASTER_HOST" ]; then
    echo -e "${RED}Error: Hostname is required${NC}"
    exit 1
fi

# Auto-detect if it's a Cloudflare hostname
if [[ "$MASTER_HOST" == *".jasondietrich.de" ]]; then
    echo -e "${GREEN}✓ Cloudflare hostname detected${NC}"
    MLFLOW_HOST="mlflow${MASTER_HOST#optuna}"
    echo -e "${YELLOW}Auto-detected MLflow hostname: $MLFLOW_HOST${NC}"
else
    read -p "MLflow Hostname (default: mlflow.${MASTER_HOST#*.}): " MLFLOW_HOST
    if [ -z "$MLFLOW_HOST" ]; then
        MLFLOW_HOST="mlflow.${MASTER_HOST#*.}"
    fi
fi

echo ""

# ============================================
# 2. Test Connection
# ============================================
echo -e "${YELLOW}[2/6] Testing connection to Master...${NC}"

echo -e "${YELLOW}Testing PostgreSQL (${MASTER_HOST}:443)...${NC}"
if command -v nc &> /dev/null; then
    if nc -z -w5 $MASTER_HOST 443 &> /dev/null; then
        echo -e "${GREEN}✓ PostgreSQL endpoint reachable${NC}"
    else
        echo -e "${RED}✗ Cannot reach PostgreSQL endpoint${NC}"
        echo -e "${YELLOW}Possible issues:${NC}"
        echo "  - Cloudflare Tunnel not configured"
        echo "  - Master server not running"
        echo "  - Firewall blocking connection"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${YELLOW}netcat not found, skipping test${NC}"
fi

echo -e "${YELLOW}Testing MLflow (${MLFLOW_HOST}:80)...${NC}"
if curl -s --connect-timeout 5 http://$MLFLOW_HOST &> /dev/null; then
    echo -e "${GREEN}✓ MLflow server reachable${NC}"
else
    echo -e "${YELLOW}⚠️  MLflow not reachable (will retry when starting)${NC}"
fi

echo ""

# ============================================
# 3. Install Dependencies
# ============================================
echo -e "${YELLOW}[3/6] Installing dependencies...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Install system dependencies (quietly)
echo -e "${YELLOW}Installing system packages...${NC}"
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends \
        build-essential \
        libgl1-mesa-dev \
        libgl1 \
        libglew-dev \
        libosmesa6-dev \
        libglfw3 \
        libglib2.0-0 \
        libgomp1 2>/dev/null || true
    echo -e "${GREEN}✓ System packages installed${NC}"
else
    echo -e "${YELLOW}apt-get not found, skipping system packages${NC}"
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python packages (this may take 2-3 minutes)...${NC}"
pip3 install --no-cache-dir -q -r requirements.txt

# Detect hardware and install appropriate PyTorch
echo ""
echo -e "${YELLOW}Detecting hardware...${NC}"

if nvidia-smi &> /dev/null 2>&1; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    pip3 install --no-cache-dir -q torch --index-url https://download.pytorch.org/whl/cu118
elif command -v rocm-smi &> /dev/null 2>&1; then
    echo -e "${GREEN}✓ AMD GPU detected${NC}"
    pip3 install --no-cache-dir -q torch --index-url https://download.pytorch.org/whl/rocm6.0
else
    echo -e "${YELLOW}Using CPU version${NC}"
    pip3 install --no-cache-dir -q torch --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# ============================================
# 4. Configure Environment
# ============================================
echo -e "${YELLOW}[4/6] Configuring environment...${NC}"

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || touch .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Generate PostgreSQL password placeholder
echo -e "${YELLOW}Enter PostgreSQL password (from master setup):${NC}"
read -s -p "Password: " POSTGRES_PASSWORD
echo ""

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo -e "${RED}Error: Password is required${NC}"
    exit 1
fi

# Update .env file
sed -i "s|OPTUNA_STORAGE=.*|OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@$MASTER_HOST:443/optuna_db|" .env
sed -i "s|MLFLOW_TRACKING_URI=.*|MLFLOW_TRACKING_URI=http://$MLFLOW_HOST:80|" .env
sed -i "s|OPTUNA_STUDY_NAME=.*|OPTUNA_STUDY_NAME=soccer_dynamic_v1|" .env

echo -e "${GREEN}✓ Configuration saved to .env${NC}"
echo ""

# ============================================
# 5. Test Full Connection
# ============================================
echo -e "${YELLOW}[5/6] Testing full connection...${NC}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Quick Python test
python3 << EOF
import sys
try:
    import optuna
    storage_url = "postgresql://optuna:$POSTGRES_PASSWORD@$MASTER_HOST:443/optuna_db"
    study = optuna.create_study(
        study_name="soccer_dynamic_v1",
        storage=storage_url,
        load_if_exists=True,
    )
    print(f"✓ Connected to Optuna study: {len(study.trials)} trials so far")
    if len(study.trials) > 0:
        completed = [t for t in study.trials if t.state.name == 'COMPLETE']
        if len(completed) > 0:
            print(f"✓ Best value: {study.best_value:.2f}")
except Exception as e:
    print(f"⚠️  Connection test failed: {e}")
    print("  Will retry when starting worker...")
    sys.exit(0)  # Don't fail, just warn
EOF

echo ""

# ============================================
# 6. Start Worker
# ============================================
echo "============================================"
echo "🚀 Starting Worker"
echo "============================================"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Master Host: $MASTER_HOST"
echo "  MLflow Host: $MLFLOW_HOST"
echo "  Study: soccer_dynamic_v1"
echo "  Mode: Infinite"
echo ""

# Set MuJoCo renderer (required!)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo -e "${YELLOW}Starting worker with offline support...${NC}"
echo -e "${GREEN}Worker will continue training even if connection is lost!${NC}"
echo -e "${GREEN}Press Ctrl+C to stop.${NC}"
echo ""
echo "============================================"
echo ""

python3 -u worker_entrypoint.py \
    --storage "postgresql://optuna:$POSTGRES_PASSWORD@$MASTER_HOST:443/optuna_db" \
    --study-name "soccer_dynamic_v1" \
    --infinite \
    --use-dynamic-rewards \
    --mlflow-tracking-uri "http://$MLFLOW_HOST:80" \
    --log-to-file
