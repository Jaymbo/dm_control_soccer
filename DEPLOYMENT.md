# 🚀 Deployment Guide - Master + Worker Setup

Komplette Anleitung für das Deployment auf deinem Server mit externen Workern.

---

## 🏗️ Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────┐
│                  MASTER SERVER (24/7)                        │
│  Your Server IP: xxx.xxx.xxx.xxx                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  MLflow      │  │  PostgreSQL  │  │  Cloudflare      │   │
│  │  :5000       │  │  :5432       │  │  Tunnel          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                              │
│  Dashboards:                                                 │
│  - MLflow UI: http://xxx.xxx.xxx.xxx:5000                   │
│  - Optuna: http://xxx.xxx.xxx.xxx:8080                      │
│  - Cloudflare: optuna.your-domain.com:443                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Secure Tunnel (Cloudflare/Tailscale)
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌─────────┐         ┌─────────┐           ┌─────────┐
│ Worker 1│         │ Worker 2│           │ Worker 3│
│ Laptop  │         │ Desktop │           │ Server  │
│ (CPU)   │         │ (GPU)   │           │ (CPU)   │
│ Anywhere│         │ Anywhere│           │ Anywhere│
└─────────┘         └─────────┘           └─────────┘
```

---

## 📋 Voraussetzungen

### Master Server
- Ubuntu/Debian Server (24/7 erreichbar)
- 4+ CPU Kerne, 8+ GB RAM
- Docker & Docker Compose
- Öffentliche IP oder Domain
- Cloudflare Zero Trust Account (kostenlos)

### Worker Machines
- Beliebiger Computer (Laptop, Desktop, Server)
- Python 3.8+
- Internet-Verbindung zum Master
- Optional: GPU für schnelleres Training

---

## 🔧 Schritt-für-Schritt Setup

### TEIL 1: Master Server einrichten

#### 1.1 SSH auf Server verbinden
```bash
ssh user@your-server-ip
```

#### 1.2 Setup-Script ausführen
```bash
# Repository klonen
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer

# Master Setup starten
./scripts/setup_master.sh
```

Das Script macht automatisch:
- ✅ Docker Installation (falls nicht vorhanden)
- ✅ Repository Klonen
- ✅ Sicheres Passwort generieren
- ✅ Cloudflare Tunnel einrichten (optional)
- ✅ Master Stack starten (MLflow + Optuna + PostgreSQL)
- ✅ Connection Details anzeigen

#### 1.3 Cloudflare Zero Trust einrichten (wenn im Script gewählt)

**Schritt A: Tunnel erstellen**
1. Gehe zu https://one.dash.cloudflare.com/
2. Melde dich an (oder erstelle kostenlosen Account)
3. Gehe zu **Access → Tunnels**
4. Klicke **Create Tunnel**
5. Name: `soccer-master`
6. Wähle **Linux** als Environment
7. Kopiere den Install-Befehl und führe ihn auf dem Server aus

**Schritt B: Tunnel authentifizieren**
```bash
# Befehl aus Cloudflare Dashboard einfügen
sudo cloudflared service install <TOKEN>
```

**Schritt C: Public Hostname hinzufügen**
1. Im Tunnel-Dashboard auf **Next** klicken
2. **Add Public Hostname** wählen:
   - **Subdomain:** `optuna`
   - **Domain:** deine-domain.com (oder .workers.dev)
   - **Service:** `tcp://localhost:5433`
3. Speichern

**Schritt D: Zweiten Hostname für MLflow**
1. Erneut **Add Public Hostname**:
   - **Subdomain:** `mlflow`
   - **Domain:** deine-domain.com
   - **Service:** `http://localhost:5000`
2. Speichern

#### 1.4 Connection Details speichern

Nach dem Setup zeigt das Script:
```
Worker Connection String:
postgresql://optuna:PASSWORT@optuna.deine-domain.com:443/optuna_db
MLFLOW_TRACKING_URI=http://mlflow.deine-domain.com:80
```

**⚠️ WICHTIG:** Speichere diese Daten! Du brauchst sie für die Worker.

---

### TEIL 2: Worker einrichten

#### 2.1 Auf jedem Worker-Gerät

```bash
# 1. Repository klonen
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer

# 2. Worker Setup starten
./scripts/setup_worker.sh
```

#### 2.2 Interaktive Konfiguration

Das Script fragt dich:

```
Please enter your Master Server connection details:

Enter OPTUNA_STORAGE URL: 
  → Hier einfügen: postgresql://optuna:PASSWORT@optuna.deine-domain.com:443/optuna_db

Enter MLFLOW_TRACKING_URI: 
  → Hier einfügen: http://mlflow.deine-domain.com:80
  (oder Enter für Auto-Detection)

Enter study name: 
  → soccer_dynamic_v1 (oder Enter für Default)

Enter number of trials: 
  → Enter für infinite (dauerhaftes Training)
```

#### 2.3 Worker startet automatisch!

Nach der Konfiguration:
- ✅ Dependencies werden installiert
- ✅ PyTorch wird für deine Hardware installiert (CPU/GPU)
- ✅ Verbindung zum Master wird getestet
- ✅ Worker startet das Training

**Logs werden live angezeigt!**

---

## 🔒 Sicherheitsoptionen

### Option A: Cloudflare Zero Trust (Empfohlen) ⭐

**Vorteile:**
- ✅ Keine Client-Software auf Workern nötig
- ✅ Automatische HTTPS-Verschlüsselung
- ✅ Zugriffskontrolle möglich
- ✅ Kostenlos bis 50 User
- ✅ Funktioniert hinter Firewalls/NAT

**Setup:** Siehe oben im Master-Setup

### Option B: Tailscale (Alternative)

**Vorteile:**
- ✅ Einfaches Setup
- ✅ Mesh-VPN (direkte Verbindungen)
- ✅ Kostenlos für persönliche Nutzung

**Setup:**

**Auf Master:**
```bash
# Tailscale installieren
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Notiere die Tailscale IP
tailscale ip
```

**Auf jedem Worker:**
```bash
# Tailscale installieren
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Mit Tailscale Account verbinden
```

**Worker Connection String:**
```
postgresql://optuna:PASSWORT@TAILSCALE_IP:5433/optuna_db
MLFLOW_TRACKING_URI=http://TAILSCALE_IP:5000
```

### Option C: SSH Tunnel (Einfach, aber manuell)

**Auf jedem Worker:**
```bash
# SSH Tunnel öffnen (im Hintergrund)
ssh -N -L 5432:localhost:5433 -L 5000:localhost:5000 user@server-ip &

# Worker Connection:
OPTUNA_STORAGE=postgresql://optuna:PASSWORT@localhost:5433/optuna_db
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 📊 Monitoring

### Dashboards öffnen

**MLflow UI:**
```
http://MASTER_IP:5000
oder
http://mlflow.deine-domain.com
```

**Optuna Dashboard:**
```
http://MASTER_IP:8080
oder
http://optuna.deine-domain.com
```

### Worker Status prüfen

**Auf Master:**
```bash
# Docker Container Status
docker-compose -f docker-compose.master.yml ps

# Logs ansehen
docker-compose -f docker-compose.master.yml logs -f

# Optuna Studie prüfen
optuna-dashboard postgresql://optuna:PASSWORT@localhost:5433/optuna_db
```

**Auf Worker:**
```bash
# Worker Prozess prüfen
ps aux | grep worker_entrypoint

# Logs ansehen
tail -f logs/optuna/workers/worker_*.log
```

---

## 🎯 Typisches Setup-Beispiel

### Master Server (Hetzner, AWS, etc.)

```bash
# Server: Ubuntu 22.04, 4 CPU, 8GB RAM
ssh root@my-server.com

# Setup
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_master.sh

# Cloudflare Tunnel einrichten (im Script "y" wählen)
# → Tunnel Name: soccer-master
# → Hostname: optuna.myapp.workers.dev → tcp://localhost:5433
# → Hostname: mlflow.myapp.workers.dev → http://localhost:5000
```

### Worker 1 (Laptop zuhause, CPU)

```bash
# Laptop: Ubuntu, 8 CPU, 16GB RAM
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_worker.sh

# Eingabe:
# OPTUNA_STORAGE=postgresql://optuna:PASSWORT@optuna.myapp.workers.dev:443/optuna_db
# MLFLOW_TRACKING_URI=http://mlflow.myapp.workers.dev:80
# Trials: Enter (infinite)
```

### Worker 2 (Desktop mit GPU)

```bash
# Desktop: Ubuntu, NVIDIA GPU, 32GB RAM
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_worker.sh

# Gleiche Connection Details
# → PyTorch mit CUDA wird automatisch installiert!
```

---

## 🛠️ Troubleshooting

### Worker kann sich nicht verbinden

**Problem:** "Connection refused" oder Timeout

**Lösung:**
1. Cloudflare Tunnel Status prüfen:
   ```bash
   sudo systemctl status cloudflared
   ```

2. Tunnel im Dashboard prüfen: https://one.dash.cloudflare.com/

3. Verbindung testen:
   ```bash
   # Auf Worker
   nc -zv optuna.myapp.workers.dev 443
   curl http://mlflow.myapp.workers.dev
   ```

### PostgreSQL Passwort falsch

**Problem:** "password authentication failed"

**Lösung:**
1. Passwort aus `worker_connection_info.txt` auf Master kopieren
2. In Worker `.env` Datei einfügen
3. Worker neustarten

### Worker bricht nach einiger Zeit ab

**Problem:** OOM (Out of Memory) oder Timeout

**Lösung:**
1. Batch-Größe reduzieren in `.env`:
   ```
   OPTUNA_N_TRIALS=5
   ```

2. Oder in `worker_entrypoint.py` anpassen:
   ```python
   num_episodes=200  # statt 400
   ```

### Training zu langsam

**Problem:** < 1000 steps/sec

**Lösung:**
1. GPU-Worker hinzufügen (5-10x schneller)
2. Batch-Größe erhöhen:
   ```bash
   python worker_entrypoint.py --storage ... --episodes-per-batch 40
   ```

---

## 📈 Skalierung

### Mehr Worker hinzufügen

Einfach das Worker-Setup auf weiteren Geräten wiederholen:

```bash
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_worker.sh
```

Alle Worker teilen sich automatisch die Trials!

### Ressourcen-Limits pro Worker

In `docker-compose.worker.yml` (falls Docker verwendet):

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 💾 Backup

### PostgreSQL Backup

```bash
# Auf Master
docker-compose -f docker-compose.master.yml exec postgres \
  pg_dump -U optuna optuna_db > backup_$(date +%Y%m%d).sql

# Backup herunterladen
scp user@server:dm_control_soccer/backup_*.sql ./
```

### MLflow Artifacts Backup

```bash
# Auf Master
docker cp optuna-postgres:/var/lib/postgresql/data ./backup_pg_data
docker cp mlflow-server:/mlflow-artifacts ./backup_mlflow
```

---

## 🎓 Best Practices

1. **Cloudflare immer verwenden** - Niemals PostgreSQL direkt exponieren!
2. **Starke Passwörter** - Das Script generiert sie automatisch
3. **Monitoring einrichten** - Dashboards regelmäßig prüfen
4. **Worker überwachen** - Logs auf Fehlern prüfen
5. **Regelmäßige Backups** - Wöchentlich PostgreSQL dumpen
6. **Updates einspielen** - `git pull` auf Master und Workern

---

## 📚 Weiterführende Links

- [Cloudflare Zero Trust Docs](https://developers.cloudflare.com/cloudflare-one/)
- [Tailscale Docs](https://tailscale.com/kb/)
- [MLflow Docs](https://mlflow.org/docs/)
- [Optuna Docs](https://optuna.readthedocs.io/)

---

**Viel Erfolg beim Deployment! 🚀**

Bei Fragen: Siehe `QUICKSTART.md` oder `README_MLFLOW_MAPO.md`
