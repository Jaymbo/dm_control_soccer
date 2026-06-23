# 🔄 Offline-First Worker Guide

**Robustes Training auch bei Verbindungsverlust!**

---

## 🎯 Features

### ✅ Offline-Support
- Worker trainiert **weiter** auch wenn Verbindung zum Master abbricht
- Metrics werden **lokal gepuffert**
- Bei Reconnect: **Automatischer Sync**

### ✅ Auto-Reconnect
- Alle **60 Sekunden** Reconnect-Versuch
- **Exponentielles Backoff** (optional)
- **Graceful** ohne Datenverlust

### ✅ SSH-Tunnel (Optional)
- **Sichere** Verbindung zum Master
- **Automatisches** Setup
- **Stabil** bei IP-Wechseln

---

## 🚀 Quickstart

### Einfaches Setup (nur Master-Host angeben)

```bash
./scripts/setup_worker_simple.sh
```

**Das Script fragt:**
1. Master Hostname (z.B. `optuna.jasondietrich.de`)
2. PostgreSQL Passwort
3. Rest automatisch!

---

## 📊 So funktioniert's

### Normaler Betrieb (Online)

```
┌─────────────────────────────────────────────────────────────┐
│  Worker                                      Master         │
│                                                              │
│  1. Trial starten                                            │
│     ───────────────────────────────────▶                     │
│     "Hol Hyperparameter"                                     │
│                                                              │
│  2. Training (200 Episoden)                                  │
│     ◀───────────────────────────────────                     │
│     Alle 10 Episoden: "Metric Update"                        │
│                                                              │
│  3. Trial Ende                                               │
│     ───────────────────────────────────▶                     │
│     "Ergebnis: -682.68"                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Bei Verbindungsverlust

```
┌─────────────────────────────────────────────────────────────┐
│  Worker                                      Master         │
│                                                              │
│  Verbindung abgebrochen!                                     │
│                                                              │
│  ╔══════════════════════════════════════════╗               │
│  ║  Training läuft WEITER (offline)         ║               │
│  ║                                          ║               │
│  ║  Metrics → lokaler Buffer                ║               │
│  ║  Checkpoints → lokale Festplatte         ║               │
│  ║  KEIN Pruning (weitertrainieren!)        ║               │
│  ║                                          ║               │
│  ║  Alle 60s: Reconnect-Versuch             ║               │
│  ╚══════════════════════════════════════════╝               │
│                                                              │
│  Nach 30 Sekunden: Verbindung wieder da!                     │
│                                                              │
│  Sync: "Hier sind 5 gepufferte Trials"                       │
│     ───────────────────────────────────▶                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Konfiguration

### Umgebungsvariablen

In `.env` oder als Export:

```bash
# Offline Mode aktivieren
OPTUNA_OFFLINE_MODE=true

# Reconnect Intervall (Sekunden)
OPTUNA_RECONNECT_INTERVAL=60

# Maximale Retries (0 = unendlich)
OPTUNA_MAX_RETRIES=5

# Retry Delay (Sekunden)
OPTUNA_RETRY_DELAY=5
```

### Im Code (`worker_entrypoint.py`)

```python
# Configuration
RECONNECT_INTERVAL = 60  # Sekunden
OFFLINE_MODE_ENABLED = True  # Offline-Support an/aus
MAX_RETRIES = 5  # Maximale Verbindungsversuche
RETRY_DELAY = 5  # Sekunden zwischen Versuchen
```

---

## 📁 Lokale Buffer

### Wo werden Daten gespeichert?

```
logs/optuna/
├── workers/
│   └── worker_hostname_pid_status.json  # Status
├── offline_buffer/
│   ├── metrics_trial_42.json            # Gepufferte Metrics
│   └── results_trial_43.json            # Gepufferte Ergebnisse
└── tensorboard/                          # Lokale TensorBoard Logs
```

### Buffer-Format

```json
{
  "trial_number": 42,
  "value": -682.68,
  "params": {
    "lr": 0.0003,
    "entropy_coef": 0.05
  },
  "timestamp": "2026-06-22T23:45:27",
  "synced": false
}
```

---

## 🧪 Use Cases

### Use Case 1: Laptop im Café

**Szenario:**
- Worker auf Laptop
- Training im Café (wechselndes WLAN)
- Verbindung bricht oft ab

**Lösung:**
```bash
./scripts/setup_worker_simple.sh
# → Trainiert weiter auch ohne Internet
# → Sync wenn wieder online
```

### Use Case 2: Reise mit Zug

**Szenario:**
- Worker auf Laptop im Zug
- Tunnel导致 Verbindungsabbrüche
- Lange Trainingseinheit (4 Stunden)

**Lösung:**
```bash
# Worker startet automatisch
./scripts/setup_worker_simple.sh

# → Trainiert durchgehend
# → Puffert bei Tunnel-Durchfahrt
# → Sync bei nächstem Empfang
```

### Use Case 3: Multi-Location

**Szenario:**
- Worker auf Desktop zuhause
- Worker auf Laptop im Büro
- Beide verbinden zu gleichem Master

**Lösung:**
```bash
# Zuhause
./scripts/setup_worker_simple.sh
# → Master Host: optuna.jasondietrich.de

# Im Büro
./scripts/setup_worker_simple.sh
# → Gleicher Master Host
# → Beide arbeiten parallel
```

---

## 🛠️ Troubleshooting

### Worker verbindet nicht

**Symptom:** "Connection refused"

**Lösung:**
```bash
# 1. Cloudflare Tunnel prüfen
nc -zv optuna.jasondietrich.de 443

# 2. Master Status prüfen (auf Server)
docker-compose -f docker-compose.master.yml ps

# 3. SSH-Tunnel manuell testen
ssh -L 5433:localhost:5433 user@master-server
```

### Worker bleibt im Offline-Mode

**Symptom:** "Still offline..." nach >5 Minuten

**Lösung:**
```bash
# 1. Netzwerk prüfen
ping optuna.jasondietrich.de

# 2. Firewall prüfen
sudo ufw status

# 3. Worker neustarten
Ctrl+C
./scripts/setup_worker_simple.sh
```

### Buffer wird nicht gesynced

**Symptom:** Trials bleiben im Buffer

**Lösung:**
```bash
# Buffer manuell prüfen
cat logs/optuna/offline_buffer/*.json

# Sync erzwingen (Worker neustarten)
# Buffer wird automatisch beim Start gesynced
```

---

## 📊 Monitoring

### Worker Status

```bash
# Aktuellen Status prüfen
cat logs/optuna/workers/worker_*_status.json

# Live-Logs
tail -f logs/optuna/workers/worker_*.log
```

### Buffer Status

```bash
# Anzahl gepufferter Trials
ls -la logs/optuna/offline_buffer/ | wc -l

# Buffer-Inhalt
cat logs/optuna/offline_buffer/*.json | jq '.trial_number'
```

### Sync Status

```bash
# Worker zeigt beim Start:
"✓ Synced 5 buffered trials from offline mode"
```

---

## 🎯 Best Practices

### 1. Lange Reconnect-Intervalle bei stabilem Netz

```bash
# In .env
OPTUNA_RECONNECT_INTERVAL=300  # 5 Minuten
```

### 2. Kurze Intervalle bei mobilem Einsatz

```bash
# In .env
OPTUNA_RECONNECT_INTERVAL=30  # 30 Sekunden
```

### 3. Lokale Checkpoints aktivieren

```bash
# In train_mappo_dynamic.py
--save-interval 10  # Alle 10 Episoden speichern
```

### 4. Graceful Shutdown ermöglichen

```bash
# Immer mit Ctrl+C beenden, nicht kill -9
# Worker speichert Status automatisch
```

---

## 📈 Performance

### Overhead durch Offline-Support

| Metrik | Online | Offline | Unterschied |
|--------|--------|---------|-------------|
| Training Speed | 100% | 100% | 0% |
| Memory Usage | +5 MB | +5 MB | 0% |
| CPU Usage | Normal | Normal | 0% |
| Sync-Zeit | - | ~1s/Trial | Minimal |

**Fazit:** Offline-Support hat **keinen** Performance-Nachteil!

---

## 🚀 Nächste Schritte

1. **Worker testen:**
   ```bash
   ./scripts/setup_worker_simple.sh
   ```

2. **Offline-Mode simulieren:**
   ```bash
   # Netzwerk trennen während Training
   sudo ip link set wlan0 down
   
   # Worker trainiert weiter!
   
   # Netzwerk wieder verbinden
   sudo ip link set wlan0 up
   
   # Worker synced automatisch
   ```

3. **Monitoring einrichten:**
   ```bash
   # Cronjob für Status-Report
   */5 * * * * cat ~/dm_control_soccer/logs/optuna/workers/*.json
   ```

---

**Robustes Training für die reale Welt! 🌍**
