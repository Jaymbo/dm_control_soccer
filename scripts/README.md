# 🚀 Deployment Scripts

Automatisierte Scripts für Master- und Worker-Setup.

---

## 📋 Überblick

| Script | Zweck | Dauer |
|--------|-------|-------|
| `setup_master.sh` | Master Server einrichten | 5-10 Min |
| `setup_worker.sh` | Worker einrichten | 3-5 Min |

---

## 🖥️ Master Server Setup

### Usage

```bash
# Auf deinem Server (Ubuntu/Debian)
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_master.sh
```

### Was das Script macht

1. ✅ **Prerequisites prüfen**
   - Docker installieren (falls nicht vorhanden)
   - Docker Compose installieren
   - Git installieren

2. ✅ **Repository klonen**
   - Pullt neueste Version von GitHub

3. ✅ **Environment konfigurieren**
   - Erstellt `.env` Datei
   - Generiert sicheres PostgreSQL-Passwort

4. ✅ **Cloudflare Tunnel (optional)**
   - Installiert `cloudflared`
   - Guidet dich durch die Einrichtung
   - Erstellt secure Tunnel für PostgreSQL

5. ✅ **Master Stack starten**
   - PostgreSQL (Optuna Storage)
   - MLflow Server (Experiment Tracking)
   - Optuna Dashboard (HPO UI)

6. ✅ **Connection Details anzeigen**
   - Zeigt Worker-Connection-Strings
   - Speichert Details in `worker_connection_info.txt`

### Interaktive Schritte

Das Script fragt:
```
Do you want to setup Cloudflare Tunnel? (y/n)
```

**Empfehlung:** `y` für externe Worker, `n` für lokales Testing

### Output

Nach erfolgreichem Setup:
```
============================================
📊 MASTER SERVER CONNECTION DETAILS
============================================

Dashboard URLs:
  MLflow UI:      http://xxx.xxx.xxx.xxx:5000
  Optuna Dashboard: http://xxx.xxx.xxx.xxx:8080

Worker Connection (via Cloudflare):
  OPTUNA_STORAGE=postgresql://optuna:PASSWORT@optuna.domain.com:443/optuna_db
  MLFLOW_TRACKING_URI=http://mlflow.domain.com:80

PostgreSQL Password: PASSWORT

Save this password! You'll need it for worker configuration.
```

---

## 💻 Worker Setup

### Usage

```bash
# Auf jedem Worker-Gerät (Laptop, Desktop, Server)
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_worker.sh
```

### Was das Script macht

1. ✅ **Prerequisites prüfen**
   - Python 3.8+ vorhanden?
   - pip3 installieren (falls nötig)
   - Git installieren (falls nötig)

2. ✅ **Dependencies installieren**
   - System-Pakete für MuJoCo
   - Python-Pakete aus `requirements.txt`
   - PyTorch (auto-detect: CPU, NVIDIA GPU, AMD GPU)

3. ✅ **Environment konfigurieren**
   - Erstellt `.env` Datei
   - Bereitet Konfiguration vor

4. ✅ **Interaktive Konfiguration**
   - Fragt Master-Connection-Details ab
   - Study Name konfigurieren
   - Anzahl Trials (oder infinite)

5. ✅ **Verbindung testen**
   - Testet PostgreSQL-Connection
   - Testet MLflow-Connection
   - Gibt hilfreiche Fehlermeldungen

6. ✅ **Worker starten**
   - Startet `worker_entrypoint.py`
   - Zeigt Live-Logs
   - Beginnt mit Training

### Interaktive Eingaben

Das Script fragt:
```
Please enter your Master Server connection details:

Enter OPTUNA_STORAGE URL: 
  → postgresql://optuna:PASSWORT@optuna.domain.com:443/optuna_db

Enter MLFLOW_TRACKING_URI: 
  → http://mlflow.domain.com:80 (oder Enter für Auto)

Enter study name: 
  → soccer_dynamic_v1 (oder Enter)

Enter number of trials: 
  → Enter für infinite (dauerhaft)
```

### Hardware-Auto-Detection

Das Script erkennt automatisch:
- **NVIDIA GPU** → PyTorch mit CUDA
- **AMD GPU** → PyTorch mit ROCm
- **Keine GPU** → PyTorch CPU

### Output

```
============================================
🚀 Starting Worker
============================================

Configuration:
  Study: soccer_dynamic_v1
  Mode: Infinite
  Storage: postgresql://...
  MLflow: http://...

============================================

[Worker-xxx] INFO: Starting Optuna Worker
[Worker-xxx] INFO: Connected to study 'soccer_dynamic_v1'
...
```

---

## 🔧 Troubleshooting

### Master Script schlägt fehl

**Fehler:** "Docker permission denied"

**Lösung:**
```bash
# User zu Docker-Gruppe hinzufügen
sudo usermod -aG docker $USER

# Aus- und einloggen
exit
# Dann erneut einloggen
```

**Fehler:** "Cloudflare Tunnel failed"

**Lösung:**
1. Script mit `n` bei Cloudflare-Frage durchlaufen
2. Tunnel manuell einrichten (siehe DEPLOYMENT.md)
3. Worker mit manueller Connection starten

### Worker Script schlägt fehl

**Fehler:** "Connection refused"

**Lösung:**
1. Cloudflare Tunnel auf Master prüfen:
   ```bash
   sudo systemctl status cloudflared
   ```

2. Verbindung testen:
   ```bash
   nc -zv optuna.domain.com 443
   ```

3. Firewall auf Master prüfen

**Fehler:** "PyTorch installation failed"

**Lösung:**
```bash
# Manuell installieren
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
# Oder für GPU:
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

**Fehler:** "Out of memory"

**Lösung:**
- Weniger Trials pro Run: `5` statt `infinite`
- Oder in `.env`: `OPTUNA_N_TRIALS=5`

---

## 📊 Monitoring

### Master Status prüfen

```bash
# Docker Container
docker-compose -f docker-compose.master.yml ps

# Logs
docker-compose -f docker-compose.master.yml logs -f

# PostgreSQL
docker-compose -f docker-compose.master.yml exec postgres pg_isready
```

### Worker Status prüfen

```bash
# Prozess
ps aux | grep worker_entrypoint

# Logs
tail -f logs/optuna/workers/worker_*.log

# TensorBoard
tensorboard --logdir logs/optuna/tensorboard
```

---

## 🔄 Updates

### Master updaten

```bash
cd dm_control_soccer
git pull
docker-compose -f docker-compose.master.yml restart
```

### Worker updaten

```bash
cd dm_control_soccer
git pull
# Worker läuft weiter, keine Neustart nötig
```

---

## 📚 Weitere Dokumentation

- [DEPLOYMENT.md](../DEPLOYMENT.md) - Komplette Deploy-Anleitung
- [QUICKSTART.md](../QUICKSTART.md) - Schnellstart
- [README_MLFLOW_MAPO.md](../README_MLFLOW_MAPO.md) - MLflow + Optuna Details

---

**Viel Erfolg! 🚀**
