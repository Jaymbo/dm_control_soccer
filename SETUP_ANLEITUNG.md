# 📝 Schritt-für-Schritt Setup Anleitung

**Für:** Master Server + Externe Worker  
**Dauer:** ~15 Minuten  
**Schwierigkeit:** Einfach ⭐⭐⭐☆☆

---

## TEIL 1: Master Server einrichten (5-10 Min)

### Schritt 1: Auf Server einloggen

```bash
ssh dein-user@dein-server.com
```

---

### Schritt 2: Repository klonen

```bash
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
```

---

### Schritt 3: Master-Setup starten

```bash
./scripts/setup_master.sh
```

Das Script läuft automatisch durch. Es fragt nur EINE Sache:

```
Do you want to setup Cloudflare Tunnel? (y/n)
```

👉 **Antwort:** `y` (für externe Worker)

---

### Schritt 4: Cloudflare Tunnel einrichten

Das Script zeigt dir:

```
Next steps for Cloudflare Tunnel:
1. Go to https://one.dash.cloudflare.com/
2. Create a Zero Trust account (if not already done)
3. Go to Access → Tunnels → Create Tunnel
4. Name it 'soccer-master' and save
5. Choose 'Linux' as environment and copy the install command
6. Run the command to authenticate
```

**Mache das jetzt im Browser:**

1. Gehe zu https://one.dash.cloudflare.com/
2. Account erstellen (kostenlos)
3. **Access** → **Tunnels** → **Create Tunnel**
4. Name: `soccer-master`
5. Linux wählen, Befehl kopieren
6. Befehl auf dem Server einfügen (im SSH-Terminal)

**Beispiel-Befehl:**
```bash
sudo cloudflared service install eyJhIjoi...
```

---

### Schritt 5: Public Hostname hinzufügen

Im Cloudflare Dashboard:

1. Bei Tunnel auf **Next** klicken
2. **Add Public Hostname**:
   ```
   Subdomain: optuna
   Domain: deine-domain.com (oder .workers.dev)
   Service: tcp://localhost:5432
   ```
3. **Save**

4. **Noch einen Hostname hinzufügen**:
   ```
   Subdomain: mlflow
   Domain: deine-domain.com
   Service: http://localhost:5000
   ```
5. **Save**

---

### Schritt 6: Connection Details speichern

Das Script zeigt am Ende:

```
============================================
📊 MASTER SERVER CONNECTION DETAILS
============================================

Worker Connection (via Cloudflare):
  OPTUNA_STORAGE=postgresql://optuna:DEIN_PASSWORT@optuna.deine-domain.com:443/optuna_db
  MLFLOW_TRACKING_URI=http://mlflow.deine-domain.com:80

PostgreSQL Password: DEIN_PASSWORT
```

**📸 Mache ein Foto oder kopiere das in eine Datei!**

Du brauchst diese Daten für die Worker!

---

### ✅ Master fertig!

Der Master läuft jetzt 24/7 im Hintergrund.

**Testen:**
```bash
# Dashboard öffnen (im Browser)
http://dein-server-ip:5000  → MLflow
http://dein-server-ip:8080  → Optuna
```

---

## TEIL 2: Worker einrichten (3-5 Min pro Worker)

### Schritt 1: Repository klonen (auf jedem Worker)

```bash
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
```

---

### Schritt 2: Worker-Setup starten

```bash
./scripts/setup_worker.sh
```

---

### Schritt 3: Connection Details eingeben

Das Script fragt:

```
Enter OPTUNA_STORAGE URL:
```

👉 **Eingabe:** Den PostgreSQL-String vom Master einfügen

```
postgresql://optuna:DEIN_PASSWORT@optuna.deine-domain.com:443/optuna_db
```

---

```
Enter MLFLOW_TRACKING_URI:
```

👉 **Eingabe:** Den MLflow-String vom Master einfügen (oder Enter für Auto)

```
http://mlflow.deine-domain.com:80
```

---

```
Enter study name (default: soccer_dynamic_v1):
```

👉 **Eingabe:** Einfach `Enter` drücken (Default ist perfekt)

---

```
Enter number of trials (default: infinite):
```

👉 **Eingabe:** `Enter` für dauerhaftes Training

---

### Schritt 4: Warten bis Installation fertig ist

Das Script installiert automatisch:
- System-Pakete (~2 Min)
- Python-Pakete (~2 Min)
- PyTorch (~1 Min)

**Es erkennt automatisch:**
- ✅ NVIDIA GPU → Installiert CUDA-Version
- ✅ AMD GPU → Installiert ROCm-Version
- ✅ Keine GPU → Installiert CPU-Version

---

### Schritt 5: Worker startet automatisch!

Nach der Installation:

```
============================================
🚀 Starting Worker
============================================

Configuration:
  Study: soccer_dynamic_v1
  Mode: Infinite
  Storage: postgresql://...
  MLflow: http://...

[Worker-xxx] INFO: Starting Optuna Worker
[Worker-xxx] INFO: Connected to study 'soccer_dynamic_v1'
[Worker-xxx] INFO: Trial 0 started
...
```

**🎉 Fertig! Der Worker trainiert jetzt!**

---

## 📊 Fortschritt überwachen

### Auf Master (im Browser)

1. **MLflow UI:** http://mlflow.deine-domain.com
   - Alle Trials im Überblick
   - Hyperparameter-Vergleich
   - Reward-Historie

2. **Optuna Dashboard:** http://optuna.deine-domain.com
   - Trial-Status
   - Beste Hyperparameter
   - Pruning-Statistiken

### Auf Worker (Terminal)

Live-Logs sehen:
```bash
tail -f logs/optuna/workers/worker_*.log
```

---

## 🎯 Zusammenfassung

### Master (1x einrichten)
```bash
ssh server
git clone ...
./scripts/setup_master.sh
# Cloudflare im Browser einrichten
# Connection Details speichern
```

### Worker (pro Gerät)
```bash
git clone ...
./scripts/setup_worker.sh
# Connection Details eingeben
# Fertig!
```

---

## 🆘 Hilfe bei Problemen

### Worker verbindet nicht

1. Cloudflare Tunnel prüfen:
   - Im Dashboard: https://one.dash.cloudflare.com/
   - Tunnel muss "Healthy" sein

2. Verbindung testen:
   ```bash
   nc -zv optuna.deine-domain.com 443
   ```

3. Passwort prüfen:
   - Stimmt das Passwort im Connection-String?

### Training zu langsam

- GPU-Worker hinzufügen (5-10x schneller)
- Mehr Worker hinzufügen (parallel)

### Mehr Details

- Siehe `DEPLOYMENT.md` für alle Optionen
- Siehe `QUICKSTART.md` für Training-Commands

---

**Viel Erfolg! ⚽🤖**

Bei Fragen: Einfach die Logs prüfen (`tail -f logs/...`)
