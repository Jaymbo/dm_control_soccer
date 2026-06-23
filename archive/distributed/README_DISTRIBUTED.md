# 🚀 Distributed Hyperparameter Optimization with Docker

This guide explains how to set up a distributed Optuna cluster for hyperparameter optimization of the Curriculum MAPPO Soccer agent.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR SERVER                              │
│  ┌─────────────────┐     ┌─────────────────────────────────┐    │
│  │   PostgreSQL    │────▶│     Optuna Dashboard            │    │
│  │   (Storage)     │     │     http://localhost:8080       │    │
│  └─────────────────┘     └─────────────────────────────────┘    │
│            ▲                                                     │
│            │ (via Cloudflare Zero Trust / VPN)                  │
└────────────┼─────────────────────────────────────────────────────┘
             │
    ┌────────┼────────┬────────────────┬────────────────┐
    │        │        │                │                │
    ▼        ▼        ▼                ▼                ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Slave  │ │ Slave  │ │ Slave  │ │ Slave  │ │ Slave  │
│ Docker │ │ Docker │ │ Docker │ │ Docker │ │ Docker │
│ Laptop │ │ Desktop│ │  VM    │ │ Server │ │  GPU   │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

**Components:**
- **Master (Server):** PostgreSQL database + Optuna Dashboard
- **Slaves (Workers):** Docker containers that pull trials and run training
- **Storage:** PostgreSQL (recommended) or SQLite for local testing

---

## Quick Start (Local Testing)

### 1. Start Master Stack (SQLite for testing)

```bash
# For local testing without PostgreSQL:
python worker_entrypoint.py --storage sqlite:///optuna.db --n-trials 50
```

### 2. Start Multiple Workers (same machine)

```bash
# Terminal 1
python worker_entrypoint.py --storage sqlite:///optuna.db --n-trials 20 &

# Terminal 2
python worker_entrypoint.py --storage sqlite:///optuna.db --n-trials 20 &

# Terminal 3
python worker_entrypoint.py --storage sqlite:///optuna.db --n-trials 20 &
```

---

## Production Setup (Multi-Machine)

### Step 1: Set Up Master Server

#### 1.1 Install Docker and Docker Compose

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 1.2 Configure Environment

Create `.env` file on the server:

```bash
# .env (on master server)
POSTGRES_USER=optuna
POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD
POSTGRES_DB=optuna_db
POSTGRES_PORT=5433
DASHBOARD_PORT=8080
```

#### 1.3 Start Master Stack

```bash
docker-compose -f docker-compose.master.yml up -d
```

Verify:
```bash
docker-compose -f docker-compose.master.yml ps
# Should show postgres and dashboard as "Up"
```

Access Dashboard: `http://your-server-ip:8080`

---

### Step 2: Secure PostgreSQL Access (Critical!)

**⚠️ Never expose PostgreSQL directly to the internet!**

#### Option A: Cloudflare Zero Trust (Recommended)

1. Install `cloudflared` on your server:
   ```bash
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. Create a tunnel (follow Cloudflare Zero Trust dashboard):
   ```bash
   cloudflared tunnel create my-tunnel
   cloudflared tunnel route dns my-tunnel optuna.your-domain.com
   ```

3. Configure `cloudflared` to proxy PostgreSQL:
   ```yaml
   # /etc/cloudflared/config.yml
   tunnel: my-tunnel
   ingress:
     - hostname: optuna.your-domain.com
       service: tcp://localhost:5433
     - service: http_status:404
   ```

4. Start cloudflared:
   ```bash
   sudo systemctl enable cloudflared
   sudo systemctl start cloudflared
   ```

#### Option B: SSH Tunnel (Simple)

Workers connect via SSH tunnel:
```bash
ssh -L 5433:localhost:5433 user@your-server
```

Connection string for workers:
```
postgresql://optuna:password@localhost:5433/optuna_db
```

#### Option C: WireGuard VPN

Set up WireGuard on server and workers for private network access.

---

### Step 3: Configure Workers

#### 3.1 On Each Worker Machine

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/soccer-project.git
   cd soccer-project
   ```

2. **Create `.env` file:**
   ```bash
   # .env (on worker)
   
   # PostgreSQL connection (via Cloudflare tunnel or VPN)
   OPTUNA_STORAGE=postgresql://optuna:CHANGE_ME_TO_SECURE_PASSWORD@optuna.your-domain.com:443/optuna_db
   
   # Study name (must match master)
   OPTUNA_STUDY_NAME=soccer_curriculum_v1
   
   # Number of trials per worker run
   OPTUNA_N_TRIALS=100
   
   # Optional: Worker ID for logging
   OPTUNA_WORKER_ID=laptop-01
   
   # Enable file logging
   OPTUNA_LOG_TO_FILE=true
   ```

3. **Start Worker (Docker Compose):**
   ```bash
   docker-compose -f docker-compose.worker.yml up -d
   ```

4. **View Logs:**
   ```bash
   docker-compose -f docker-compose.worker.yml logs -f
   ```

#### 3.2 Alternative: Run Worker Directly (no Docker)

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or cuda/rocm

# Run worker
python worker_entrypoint.py \
  --storage postgresql://optuna:password@optuna.your-domain.com:443/optuna_db \
  --n-trials 100 \
  --worker-id laptop-01 \
  --log-to-file
```

---

## Monitoring

### Optuna Dashboard

Access at `http://your-server-ip:8080` to see:
- Trial history
- Hyperparameter importance
- Parallel coordinate plots
- Best trials

### TensorBoard

```bash
# On master server
tensorboard --logdir logs/optuna/tensorboard --port 6006
```

### Worker Logs

```bash
# Docker logs
docker logs optuna-worker -f

# Local log files (if enabled)
tail -f logs/workers/worker_*.log
```

---

## Scaling

### Add More Workers

Simply start more worker containers on any machine:

```bash
# On machine 1
docker-compose -f docker-compose.worker.yml up -d

# On machine 2
docker-compose -f docker-compose.worker.yml up -d

# On machine 3
python worker_entrypoint.py --storage postgresql://... --infinite
```

Workers automatically pull new trials from the central storage.

### Resource Limits

In `docker-compose.worker.yml`, set CPU/memory limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

---

## Advanced Configuration

### Custom Study Configuration

Modify `worker_entrypoint.py` to change:
- Search space (hyperparameters)
- Training epochs per trial
- Pruning strategy

### Multiple Studies

Run different studies in parallel:

```bash
# Study 1: Curriculum learning
python worker_entrypoint.py --study-name soccer_curriculum_v1 --storage ...

# Study 2: Architecture search
python worker_entrypoint.py --study-name soccer_architecture_v1 --storage ...
```

### Persistent Workers (Infinite Mode)

```bash
python worker_entrypoint.py --storage postgresql://... --infinite
```

Worker will continuously pull new trials as they become available.

---

## Troubleshooting

### Worker Cannot Connect to Database

1. Check network connectivity:
   ```bash
   telnet optuna.your-domain.com 443
   ```

2. Verify credentials:
   ```bash
   psql postgresql://optuna:password@optuna.your-domain.com:443/optuna_db
   ```

3. Check Cloudflare tunnel status:
   ```bash
   sudo systemctl status cloudflared
   ```

### Trials Not Being Distributed

1. Ensure all workers use the same `--study-name`
2. Check that storage URL is identical
3. Verify no firewall is blocking connections

### Out of Memory

Reduce batch size or episodes per trial:
- Edit `create_training_args()` in `worker_entrypoint.py`
- Set `num_episodes=200` instead of 400 for faster trials

---

## Security Checklist

- [ ] PostgreSQL password changed from default
- [ ] PostgreSQL not exposed to public internet
- [ ] Cloudflare Zero Trust or VPN configured
- [ ] Worker machines have firewall enabled
- [ ] Docker containers run as non-root user
- [ ] Regular backups of PostgreSQL database

---

## Backup and Restore

### Backup Database

```bash
docker-compose -f docker-compose.master.yml exec postgres \
  pg_dump -U optuna optuna_db > backup_$(date +%Y%m%d).sql
```

### Restore Database

```bash
docker-compose -f docker-compose.master.yml exec -T postgres \
  psql -U optuna optuna_db < backup_20260622.sql
```

---

## Cost Estimation

**Example: 5 workers running for 24 hours**

- Each worker completes ~10 trials/day (400 episodes each)
- Total: 50 trials/day
- Time to complete 500 trials: ~10 days
- Cost: Only electricity/cloud compute for 5 machines

---

## Next Steps

1. Set up master server with PostgreSQL
2. Configure Cloudflare Zero Trust for secure access
3. Test with one local worker
4. Deploy workers to additional machines
5. Monitor progress via Optuna Dashboard
6. Analyze best hyperparameters and retrain final model

For questions or issues, check the main `AGENTS.md` or open an issue.
