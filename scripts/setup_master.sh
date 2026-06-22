#!/bin/bash
# ============================================
# Master Server Setup Script
# ============================================
# Usage: ./scripts/setup_master.sh
#
# This script:
# 1. Installs Docker & Docker Compose
# 2. Clones the repository
# 3. Configures Cloudflare Tunnel for PostgreSQL
# 4. Starts the Master Stack (MLflow + Optuna + PostgreSQL)
# 5. Outputs connection details for workers
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
NC='\033[0m' # No Color

# ============================================
# 1. Check Prerequisites
# ============================================
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

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
# 2. Clone Repository
# ============================================
echo -e "${YELLOW}[2/7] Cloning repository...${NC}"

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
# 3. Configure Environment
# ============================================
echo -e "${YELLOW}[3/7] Configuring environment...${NC}"

# Generate secure password
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)

# Copy .env.example to .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Update .env with secure password
sed -i "s/POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env
echo -e "${GREEN}✓ PostgreSQL password generated${NC}"

echo ""

# ============================================
# 4. Setup Cloudflare Tunnel (Optional)
# ============================================
echo -e "${YELLOW}[4/7] Cloudflare Tunnel Setup${NC}"
echo ""
echo "Cloudflare Zero Trust provides secure access to PostgreSQL without exposing it to the internet."
echo ""
read -p "Do you want to setup Cloudflare Tunnel? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Installing cloudflared...${NC}"
    
    # Download and install cloudflared
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
    
    echo -e "${GREEN}✓ cloudflared installed${NC}"
    echo ""
    echo -e "${YELLOW}Next steps for Cloudflare Tunnel:${NC}"
    echo "1. Go to https://one.dash.cloudflare.com/"
    echo "2. Create a Zero Trust account (if not already done)"
    echo "3. Go to Access → Tunnels → Create Tunnel"
    echo "4. Name it 'soccer-master' and save"
    echo "5. Choose 'Linux' as environment and copy the install command"
    echo "6. Run the command to authenticate"
    echo "7. Add a Public Hostname:"
    echo "   - Subdomain: optuna"
    echo "   - Domain: your-domain.com"
    echo "   - Service: tcp://localhost:5432"
    echo ""
    echo -e "${YELLOW}After setup, your workers will connect to: optuna.your-domain.com:443${NC}"
    echo ""
    CLOUDFLARE_SETUP=true
else
    echo -e "${YELLOW}Skipping Cloudflare Tunnel setup${NC}"
    echo -e "${YELLOW}Alternative: Use Tailscale (documented in README)${NC}"
    CLOUDFLARE_SETUP=false
fi

echo ""

# ============================================
# 5. Start Master Stack
# ============================================
echo -e "${YELLOW}[5/7] Starting Master Stack...${NC}"

docker-compose -f docker-compose.master.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to start (this may take 30-60 seconds)...${NC}"
sleep 10

# Check status
docker-compose -f docker-compose.master.yml ps

echo -e "${GREEN}✓ Master Stack started${NC}"
echo ""

# ============================================
# 6. Verify Services
# ============================================
echo -e "${YELLOW}[6/7] Verifying services...${NC}"

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
# 7. Output Connection Details
# ============================================
echo -e "${YELLOW}[7/7] Setup Complete!${NC}"
echo ""
echo "============================================"
echo "📊 MASTER SERVER CONNECTION DETAILS"
echo "============================================"
echo ""
echo -e "${GREEN}Dashboard URLs:${NC}"
echo "  MLflow UI:      http://$(hostname -I | awk '{print $1}'):5000"
echo "  Optuna Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
echo ""

if [ "$CLOUDFLARE_SETUP" = true ]; then
    echo -e "${GREEN}Worker Connection (via Cloudflare):${NC}"
    echo "  OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@optuna.YOUR-DOMAIN.com:443/optuna_db"
    echo "  MLFLOW_TRACKING_URI=http://optuna.YOUR-DOMAIN.com:80"
    echo ""
else
    echo -e "${GREEN}Worker Connection (Direct/VPN):${NC}"
    echo "  OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@$(hostname -I | awk '{print $1}'):5432/optuna_db"
    echo "  MLFLOW_TRACKING_URI=http://$(hostname -I | awk '{print $1}'):5000"
    echo ""
    echo -e "${YELLOW}⚠️  Note: For external workers, setup Cloudflare Tunnel or Tailscale!${NC}"
    echo ""
fi

echo -e "${GREEN}PostgreSQL Password:${NC} $POSTGRES_PASSWORD"
echo ""
echo -e "${YELLOW}Save this password! You'll need it for worker configuration.${NC}"
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
echo "2. When prompted, enter the connection details above"
echo ""
echo "3. Workers will automatically start training!"
echo ""
echo "============================================"
echo ""

# Save connection details to file
cat > worker_connection_info.txt << EOF
Master Server Connection Details
================================

Generated: $(date)

PostgreSQL Password: $POSTGRES_PASSWORD

Worker Connection String:
EOF

if [ "$CLOUDFLARE_SETUP" = true ]; then
    echo "OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@optuna.YOUR-DOMAIN.com:443/optuna_db" >> worker_connection_info.txt
    echo "MLFLOW_TRACKING_URI=http://optuna.YOUR-DOMAIN.com:80" >> worker_connection_info.txt
else
    echo "OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@$(hostname -I | awk '{print $1}'):5432/optuna_db" >> worker_connection_info.txt
    echo "MLFLOW_TRACKING_URI=http://$(hostname -I | awk '{print $1}'):5000" >> worker_connection_info.txt
fi

echo ""
echo -e "${GREEN}✓ Connection details saved to: worker_connection_info.txt${NC}"
echo ""
echo -e "${GREEN}🎉 Master Server Setup Complete!${NC}"
