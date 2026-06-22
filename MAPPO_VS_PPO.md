# 🤖 MAPPO vs. Zentralisierter PPO - Vergleich

## 📊 Architektur-Unterschiede

### Zentralisierter PPO (`train.py`)
```
Alle Observations (476) ──→ MLP ──→ Alle Actions (12)
                              │
                              └──→ Global Value
```
- **Ein** zentrales Netzwerk steuert alle 4 Spieler
- Beobachtet den **globalen Zustand**
- Gibt **alle Actions** auf einmal aus

### MAPPO (`train_mappo.py`)
```
Spieler 1 Obs (119) ──┐
Spieler 2 Obs (119) ──┼──→ Shared Actor ──→ Action 1
Spieler 3 Obs (119) ──┤                     Action 2
Spieler 4 Obs (119) ──┘                     Action 3
                                              Action 4
                      
Alle Observations (476) ──→ Centralized Critic ──→ Global Value
```
- **Ein Actor** pro Spieler (shared weights)
- **Ein Critic** mit globalem Überblick (CTDE)
- Jeder Spieler entscheidet **lokal**

## 🎯 Vor- und Nachteile

### Zentralisierter PPO

**Vorteile:**
- ✅ Einfacher zu implementieren
- ✅ Weniger Parameter (ein Netzwerk)
- ✅ Schnelleres Training (weniger Forward-Passes)
- ✅ Optimal bei vollständiger Information

**Nachteile:**
- ❌ Skaliert schlecht bei mehr Spielern
- ❌ Ein Fehler im Netzwerk betrifft alle Spieler
- ❌ Kann keine individuellen Strategien lernen

### MAPPO (CTDE)

**Vorteile:**
- ✅ Bessere Skalierbarkeit (n Spieler)
- ✅ Robuster bei partiellen Observations
- ✅ Spieler können spezialisierte Rollen lernen
- ✅ State-of-the-Art für Multi-Agenten-Probleme
- ✅ Transfer auf reale Robotik (dezentrale Execution)

**Nachteile:**
- ❌ Mehr Forward-Passes nötig
- ❌ Komplexere Implementierung
- ❌ Längere Trainingszeit pro Episode

## 📈 Wann welches Verfahren?

| Szenario | Empfehlung |
|----------|------------|
| 2vs2 Soccer (dieses Projekt) | Beide möglich, PPO einfacher |
| 5vs5 Soccer | **MAPPO** |
| Partielle Observations | **MAPPO** |
| Schnelles Prototyping | **PPO** |
| Individuelle Rollen | **MAPPO** |
| Forschung/SOTA | **MAPPO** |

## 🚀 Training Commands

### Zentralisierter PPO
```bash
# Standard
python train.py --num-episodes 1000

# Mit Viewer
python train.py --viewer --viewer-interval 50
```

### MAPPO
```bash
# Standard (Centralized Critic)
python train_mappo.py --num-episodes 1000

# Mit Decentralized Critic
python train_mappo.py --decentralized-critic --num-episodes 1000

# Mit Viewer
python train_mappo.py --viewer --viewer-interval 50
```

## 📁 Dateien

| Datei | Zweck |
|-------|-------|
| `agent.py` | Zentralisierter Actor-Critic |
| `train.py` | Zentralisiertes PPO Training |
| `test.py` | Zentralisierten Agent testen |
| `agent_mappo.py` | MAPPO Agent (CTDE) |
| `train_mappo.py` | MAPPO Training |
| `test_mappo.py` | MAPPO Agent testen |

## 🔬 Forschungshintergrund

**MAPPO Paper:**  
*"Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"*  
Yu et al., NeurIPS 2021

**Kernidee (CTDE):**
- **Training:** Critic sieht alles → stabileres Learning
- **Execution:** Jeder Actor sieht nur sich → robust, skalierbar

**Warum CTDE?**
1. **Credit Assignment:** Critic kann beobachten, welcher Spieler zum Erfolg beiträgt
2. **Non-Stationarity:** Andere Spieler ändern sich → globaler Critic hilft
3. **Scalability:** Neue Spieler können hinzugefügt werden ohne Retraining

## 🎯 Erwartete Performance

### Training Speed
- **PPO:** ~100-200 Episoden bis erste Tore
- **MAPPO:** ~150-250 Episoden bis erste Tore (etwas langsamer)

### Final Performance
- **PPO:** Stabil nach ~500-800 Episoden
- **MAPPO:** Stabil nach ~600-1000 Episoden, aber **besser generalisierend**

### Generalization
- **PPO:** Gut auf bekannte Situationen
- **MAPPO:** Besser auf neue Situationen (robuster)

## 💡 Tipps für MAPPO

1. **Learning Rate:** Etwas niedriger (1e-4 bis 3e-4)
2. **Hidden Dim:** 256-512 empfohlen
3. **Centralized Critic:** Fast immer besser als decentralized
4. **Reward Shaping:** Wichtig für beide, aber besonders für MAPPO
5. **Training Length:** MAPPO braucht oft länger, wird aber besser

## 📊 Monitoring

Beide Methoden nutzen Tensorboard:
```bash
# PPO Logs
tensorboard --logdir logs/soccer_ppo

# MAPPO Logs
tensorboard --logdir logs/soccer_mappo
```

Vergleiche:
- `Reward/avg_100` - Durchschnittliche Performance
- `Loss/policy` - Policy Learning
- `Loss/value` - Value Estimation Quality
