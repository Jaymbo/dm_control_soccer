#!/bin/bash
# ============================================
# Master Server Setup Script (Cloudflare Zero Trust)
# ============================================
# Usage: ./scripts/setup_master.sh
#
# This script:
# 1. Checks prerequisites (Docker, git)
# 2. Starts the Master Stack
# 3. Outputs Cloudflare connection details
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
echo -e "${YELLOW}[1/4] Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not found. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
else
    echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose not found. Please install it first.${NC}"
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
else
    echo -e "${GREEN}✓ Docker Compose found: $(docker-compose --version)${NC}"
fi

# Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: Git not found. Please install git first.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Git found${NC}"
fi

echo ""

# ============================================
# 2. Generate Password & Configure
# ============================================
echo -e "${YELLOW}[2/4] Configuring environment...${NC}"

# Generate secure password
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)

# Copy .env.example to .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
    # Backup existing .env
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    echo -e "${YELLOW}✓ Backup created: .env.backup.*${NC}"
fi

# Update .env with secure password (using | delimiter to avoid issues with /)
# First update the POSTGRES_PASSWORD
if grep -q "POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD" .env; then
    sed -i "s|POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" .env
    echo -e "${GREEN}✓ PostgreSQL password set${NC}"
else
    echo -e "${YELLOW}POSTGRES_PASSWORD already configured${NC}"
fi

echo -e "${GREEN}✓ PostgreSQL password generated: $POSTGRES_PASSWORD${NC}"
echo ""

# ============================================
# 3. Start Master Stack
# ============================================
echo -e "${YELLOW}[3/4] Starting Master Stack...${NC}"

# Check if port 5432 is already in use
if ss -tlnp 2>/dev/null | grep -q ":5432 " || netstat -tlnp 2>/dev/null | grep -q ":5432 "; then
    echo -e "${YELLOW}Warning: Port 5432 is already in use${NC}"
    echo -e "${YELLOW}Using alternative port 5433 for this PostgreSQL instance...${NC}"
    
    # Create temporary docker-compose override
    cat > docker-compose.master.override.yml << EOF
version: "3.8"
services:
  postgres:
    ports:
      - "5433:5432"
EOF
    
    docker-compose -f docker-compose.master.yml -f docker-compose.master.override.yml up -d
else
    docker-compose -f docker-compose.master.yml up -d
fi

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to start (this may take 30-60 seconds)...${NC}"
sleep 15

# Check status
docker-compose -f docker-compose.master.yml ps

echo -e "${GREEN}✓ Master Stack started${NC}"
echo ""

# ============================================
# 4. Verify Services & Output Details
# ============================================
echo -e "${YELLOW}[4/4] Verifying services...${NC}"

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
echo -e "${BLUE}PostgreSQL Password:${NC} $POSTGRES_PASSWORD"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Save this password!${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo ""
echo "1. Configure Cloudflare Zero Trust Tunnel:"
echo "   - Go to: https://one.dash.cloudflare.com/"
echo "   - Access → Tunnels → Create a tunnel"
echo "   - Name: soccer-master"
echo "   - Install cloudflared on this server"
echo "   - Add public hostnames:"
echo "     * optuna.jasondietrich.de → tcp://localhost:5432"
echo "     * mlflow.jasondietrich.de → http://localhost:5000"
echo ""
echo "2. After Cloudflare is configured, workers can connect with:"
echo "   OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@optuna.jasondietrich.de:443/optuna_db"
echo "   MLFLOW_TRACKING_URI=http://mlflow.jasondietrich.de:80"
echo ""
echo "3. Or use the simple worker setup:"
echo "   ./scripts/setup_worker_simple.sh"
echo ""
echo "============================================"

# Save connection details to file
cat > worker_connection_info.txt << EOF
Master Server Connection Details (Cloudflare Zero Trust)
=========================================================

Generated: $(date)

PostgreSQL Password: $POSTGRES_PASSWORD

After Cloudflare Tunnel is configured:
Worker Connection String:
OPTUNA_STORAGE=postgresql://optuna:$POSTGRES_PASSWORD@optuna.jasondietrich.de:443/optuna_db
MLFLOW_TRACKING_URI=http://mlflow.jasondietrich.de:80

Dashboard URLs (after Cloudflare setup):
MLflow UI: http://mlflow.jasondietrich.de
Optuna Dashboard: http://optuna.jasondietrich.de

Local URLs (on server only):
MLflow UI: http://localhost:5000
Optuna Dashboard: http://localhost:8080
PostgreSQL: localhost:5432

Cloudflare Setup Instructions:
1. Go to https://one.dash.cloudflare.com/
2. Access → Tunnels → Create a tunnel
3. Name: soccer-master
4. Install cloudflared on this server
5. Add public hostnames:
   - optuna.jasondietrich.de → tcp://localhost:5432
   - mlflow.jasondietrich.de → http://localhost:5000
EOF

echo -e "${GREEN}✓ Connection details saved to: worker_connection_info.txt${NC}"
echo ""
echo -e "${GREEN}🎉 Master Server Setup Complete!${NC}"
echo ""
echo -e "${BLUE}Local Dashboard URLs (on server only):${NC}"
echo "  MLflow UI:      http://localhost:5000"
echo "  Optuna Dashboard: http://localhost:8080"
echo ""
