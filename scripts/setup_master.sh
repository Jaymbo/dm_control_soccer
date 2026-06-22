#!/bin/bash
# ============================================
# Master Server Setup Script (Tailscale)
# ============================================
# Usage: ./scripts/setup_master.sh
#
# This script:
# 1. Installs Docker & Docker Compose
# 2. Installs Tailscale
# 3. Clones the repository
# 4. Starts the Master Stack
# 5. Outputs Tailscale connection details
# ============================================

set -e

echo "============================================"
echo "🚀 Multi-Agent Soccer - Master Server Setup"
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
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}Error: Please do not run as root${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Installing...${NC}"
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}✓ Docker installed${NC}"
    echo -e "${YELLOW}Note: You may need to log out and back in for Docker permissions${NC}"
else
    echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Docker Compose not found. Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y docker-compose
    echo -e "${GREEN}✓ Docker Compose installed${NC}"
else
    echo -e "${GREEN}✓ Docker Compose found: $(docker-compose --version)${NC}"
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
# 2. Install Tailscale
# ============================================
echo -e "${YELLOW}[2/6] Installing Tailscale...${NC}"

if command -v tailscale &> /dev/null; then
    echo -e "${GREEN}✓ Tailscale already installed${NC}"
    TAILSCALE_ALREADY_INSTALLED=true
else
    echo -e "${YELLOW}Installing Tailscale...${NC}"
    curl -fsSL https://tailscale.com/install.sh | sh
    echo -e "${GREEN}✓ Tailscale installed${NC}"
fi

echo ""

# ============================================
# 3. Authenticate Tailscale
# ============================================
echo -e "${YELLOW}[3/6] Tailscale Authentication${NC}"
echo ""

if [ "$TAILSCALE_ALREADY_INSTALLED" = true ]; then
    TAILSCALE_IP=$(tailscale ip 2>/dev/null || echo "")
    if [ -n "$TAILSCALE_IP" ]; then
        echo -e "${GREEN}✓ Tailscale already authenticated${NC}"
        echo -e "${BLUE}Tailscale IP: $TAILSCALE_IP${NC}"
    else
        echo -e "${YELLOW}Tailscale installed but not authenticated${NC}"
        NEED_AUTH=true
    fi
else
    NEED_AUTH=true
fi

if [ "$NEED_AUTH" = true ]; then
    echo -e "${BLUE}Authenticating Tailscale...${NC}"
    echo ""
    echo "Tailscale will open a browser window for authentication."
    echo "If it doesn't open automatically, copy the URL that appears."
    echo ""
    
    # Try to authenticate
    if sudo tailscale up 2>/dev/null; then
        echo -e "${GREEN}✓ Tailscale authenticated successfully${NC}"
    else
        echo ""
        echo -e "${YELLOW}Automatic authentication failed.${NC}"
        echo -e "${YELLOW}Please authenticate manually:${NC}"
        echo ""
        echo "1. Run: sudo tailscale up"
        echo "2. Open the URL in your browser"
        echo "3. Login with Google, Microsoft, or GitHub"
        echo ""
        read -p "Press Enter after you've authenticated..."
    fi
    
    TAILSCALE_IP=$(tailscale ip)
    echo -e "${BLUE}Tailscale IP: $TAILSCALE_IP${NC}"
fi

echo ""

# ============================================
# 4. Clone Repository
# ============================================
echo -e "${YELLOW}[4/6] Cloning repository...${NC}"

REPO_URL="git@github.com:Jaymbo/dm_control_soccer.git"
REPO_DIR="dm_control_soccer"

if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Repository already exists. Updating...${NC}"
    cd $REPO_DIR
    git pull
    cd ..
else
    git clone $REPO_URL
    echo -e "${GREEN}✓ Repository cloned${NC}"
fi

cd $REPO_DIR
echo ""

# ============================================
# 5. Configure Environment
# ============================================
echo -e "${YELLOW}[5/6] Configuring environment...${NC}"

# Generate secure password
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)

# Copy .env.example to .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Update .env with secure password and Tailscale IP
sed -i "s/POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env
sed -i "s|OPTUNA_STORAGE=.*|OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@$TAILSCALE_IP:5432/optuna_db|" .env
sed -i "s|MLFLOW_TRACKING_URI=.*|MLFLOW_TRACKING_URI=http://$TAILSCALE_IP:5000|" .env

echo -e "${GREEN}✓ PostgreSQL password generated${NC}"
echo -e "${GREEN}✓ Tailscale IP configured: $TAILSCALE_IP${NC}"
echo ""

# ============================================
# 6. Start Master Stack
# ============================================
echo -e "${YELLOW}[6/6] Starting Master Stack...${NC}"

docker-compose -f docker-compose.master.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to start (this may take 30-60 seconds)...${NC}"
sleep 10

# Check status
docker-compose -f docker-compose.master.yml ps

echo -e "${GREEN}✓ Master Stack started${NC}"
echo ""

# ============================================
# Verify Services
# ============================================
echo -e "${YELLOW}Verifying services...${NC}"

# Check PostgreSQL
if docker-compose -f docker-compose.master.yml exec -T postgres pg_isready -U optuna -d optuna_db &> /dev/null; then
    echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
else
    echo -e "${YELLOW}PostgreSQL is starting... (wait a moment)${NC}"
fi

# Check MLflow
if curl -s http://localhost:5000 &> /dev/null; then
    echo -e "${GREEN}✓ MLflow Server is running${NC}"
else
    echo -e "${YELLOW}MLflow Server is starting... (wait a moment)${NC}"
fi

# Check Optuna Dashboard
if curl -s http://localhost:8080 &> /dev/null; then
    echo -e "${GREEN}✓ Optuna Dashboard is running${NC}"
else
    echo -e "${YELLOW}Optuna Dashboard is starting... (wait a moment)${NC}"
fi

echo ""

# ============================================
# Output Connection Details
# ============================================
echo "============================================"
echo -e "${GREEN}📊 MASTER SERVER SETUP COMPLETE!${NC}"
echo "============================================"
echo ""
echo -e "${BLUE}Tailscale IP:${NC} $TAILSCALE_IP"
echo ""
echo -e "${GREEN}Dashboard URLs (access via Tailscale):${NC}"
echo "  MLflow UI:      http://$TAILSCALE_IP:5000"
echo "  Optuna Dashboard: http://$TAILSCALE_IP:8080"
echo ""
echo -e "${GREEN}Worker Connection String:${NC}"
echo "  OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@$TAILSCALE_IP:5432/optuna_db"
echo "  MLFLOW_TRACKING_URI=http://$TAILSCALE_IP:5000"
echo ""
echo -e "${YELLOW}⚠️  Save this password!${NC} $POSTGRES_PASSWORD"
echo ""
echo "============================================"
echo "📝 NEXT STEPS FOR WORKERS:"
echo "============================================"
echo ""
echo "1. On each worker machine, run:"
echo "   git clone git@github.com:Jaymbo/dm_control_soccer.git"
echo "   cd dm_control_soccer"
echo "   ./scripts/setup_worker.sh"
echo ""
echo "2. When prompted, enter this Tailscale IP: $TAILSCALE_IP"
echo ""
echo "3. Workers will automatically connect and start training!"
echo ""
echo "============================================"
echo ""

# Save connection details to file
cat > worker_connection_info.txt << EOF
Master Server Connection Details (Tailscale)
============================================

Generated: $(date)

Tailscale IP: $TAILSCALE_IP
PostgreSQL Password: $POSTGRES_PASSWORD

Worker Connection String:
OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@$TAILSCALE_IP:5432/optuna_db
MLFLOW_TRACKING_URI=http://$TAILSCALE_IP:5000

Dashboard URLs:
MLflow UI: http://$TAILSCALE_IP:5000
Optuna Dashboard: http://$TAILSCALE_IP:8080

To access dashboards:
1. Install Tailscale on your laptop: https://tailscale.com/download
2. Login with the same account
3. Open the URLs above in your browser
EOF

echo -e "${GREEN}✓ Connection details saved to: worker_connection_info.txt${NC}"
echo ""
echo -e "${GREEN}🎉 Master Server Setup Complete!${NC}"
echo ""
echo -e "${BLUE}To view dashboards from your laptop:${NC}"
echo "1. Install Tailscale: https://tailscale.com/download"
echo "2. Login with the same account"
echo "3. Visit: http://$TAILSCALE_IP:5000"
