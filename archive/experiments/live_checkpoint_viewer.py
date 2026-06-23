"""
Live Checkpoint Watcher für MAPPO Dynamic V2 Soccer.

Läuft parallel zum Training und öffnet automatisch den neuesten Checkpoint
im dm_control Viewer. Sobald eine neue/bessere Checkpoint-Datei erkannt wird,
startet der Viewer neu mit der aktuellsten Policy.

Nutzung:
    python live_checkpoint_viewer.py --checkpoint-dir logs/soccer_mappo_dynamic_v2_online
    
    Oder direkt auf best_agent.pt:
    python live_checkpoint_viewer.py --checkpoint logs/soccer_mappo_dynamic_v2_online/best_agent.pt

Steuerung im Viewer:
    SPACE      - Pause / Resume
    TAB        - Nächste Kamera (startet mit top_down)
    R          - Reset
    +/-        - Geschwindigkeit
    ESC/Q      - Beenden
"""
import os
import sys
import time
import argparse
import subprocess
import glob
import torch
import numpy as np
import signal
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent


def get_latest_checkpoint(checkpoint_dir, pattern="*.pt"):
    """Finde die neueste .pt Datei im Verzeichnis (nach Änderungszeit)."""
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def flatten_obs(obs, num_agents, obs_dim_per_agent):
    """Flachte Observations aller Spieler."""
    flat = []
    for player_obs in obs:
        for key in sorted(player_obs.keys()):
            flat.append(np.asarray(player_obs[key]).flatten())
    flat_arr = np.concatenate(flat).astype(np.float32)
    return np.stack(split_obs_by_agent(flat_arr, num_agents, obs_dim_per_agent), axis=0)


def build_policy(agent, num_agents, obs_dim_per_agent, device):
    """Erzeugt eine Policy-Funktion für den dm_control viewer."""
    def policy(timestep):
        obs_per_agent = flatten_obs(timestep.observation, num_agents, obs_dim_per_agent)
        obs_t = torch.FloatTensor(obs_per_agent).unsqueeze(0).to(device)
        with torch.no_grad():
            actions, _, _ = agent.get_actions(obs_t, deterministic=True)
        return actions.squeeze(0).cpu().numpy().flatten()
    return policy


def load_agent(checkpoint_path, device, team_size, actor_layers=3, critic_layers=3, hidden_dim=256, use_layer_norm=False):
    """Lädt Agent aus Checkpoint."""
    num_agents = team_size * 2
    obs_dim_per_agent = 119

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=3,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        centralized_critic=True,
        actor_layers=actor_layers,
        critic_layers=critic_layers,
        use_layer_norm=use_layer_norm,
    )
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.to(device)
    agent.eval()
    return agent


def launch_viewer(env, agent, num_agents, obs_dim_per_agent, device, checkpoint_path=None,
                  team_size=2, time_limit=10.0, hidden_dim=256, actor_layers=3, critic_layers=3):
    """Startet den Viewer mit der gegebenen Policy als separates Skript (non-blocking)."""
    # Speichere Agent temporär
    if checkpoint_path is None:
        temp_checkpoint = "/tmp/viewer_temp_checkpoint.pt"
        torch.save({'agent_state_dict': agent.state_dict()}, temp_checkpoint)
        checkpoint_path = temp_checkpoint
    
    viewer_script = f'''
import sys
sys.path.insert(0, "{os.getcwd()}")

import torch
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent

def flatten_obs(obs):
    flat = []
    for player_obs in obs:
        for key in sorted(player_obs.keys()):
            flat.append(np.asarray(player_obs[key]).flatten())
    return np.concatenate(flat).astype(np.float32)

checkpoint = torch.load("{checkpoint_path}", map_location="{device}", weights_only=False)
agent = MAPPOAgent(
    obs_dim_per_agent={obs_dim_per_agent},
    action_dim_per_agent=3,
    num_agents={num_agents},
    hidden_dim={hidden_dim},
    centralized_critic=True,
    actor_layers={actor_layers},
    critic_layers={critic_layers},
)
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

env = dm_soccer.load(
    team_size={team_size},
    time_limit={time_limit},
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False,
    walker_type=dm_soccer.WalkerType.BOXHEAD,
)

def policy(timestep):
    obs_flat = flatten_obs(timestep.observation)
    obs_per_agent = split_obs_by_agent(obs_flat, num_agents={num_agents}, obs_dim_per_agent={obs_dim_per_agent})
    with torch.no_grad():
        actions, _, _ = agent.get_actions(torch.FloatTensor(obs_per_agent).unsqueeze(0).to("{device}"), deterministic=True)
    return np.concatenate([a.cpu().numpy() for a in actions])

print("\\nViewer gestartet. Steuerung:")
print("  TAB - Kamera wechseln")
print("  LEERTASTE - Start/Pause")
print("  R - Reset")
print("  +/- - Geschwindigkeit")
print("  ESC/Q - Beenden")
print("\\nHinweis: Drücke LEERTASTE zum Starten!")

viewer.launch(env, policy=policy, title="MAPPO Soccer Viewer - {checkpoint_path}")
'''
    
    temp_script = "/tmp/soccer_viewer_temp.py"
    with open(temp_script, 'w') as f:
        f.write(viewer_script)
    
    # Non-blocking starten (ohne timeout)
    try:
        proc = subprocess.Popen(['python', temp_script], cwd=os.getcwd())
        return proc, temp_script
    except Exception as e:
        print(f"[Viewer] Error: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return None, None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_agents = args.team_size * 2
    obs_dim_per_agent = 119

    # Basis-Environment für den Viewer (wird mehrfach neu gestartet)
    def make_viewer_env():
        return dm_soccer.load(
            team_size=args.team_size,
            time_limit=args.time_limit,
            disable_walker_contacts=False,
            enable_field_box=True,
            terminate_on_goal=False,
            walker_type=dm_soccer.WalkerType.BOXHEAD,
        )

    if args.checkpoint:
        # Einzelner Checkpoint-Modus
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint nicht gefunden: {checkpoint_path}")
            sys.exit(1)
        print(f"Lade Checkpoint: {checkpoint_path}")
        agent = load_agent(
            checkpoint_path, device, args.team_size,
            actor_layers=args.actor_layers,
            critic_layers=args.critic_layers,
            hidden_dim=args.hidden_dim,
            use_layer_norm=args.use_layer_norm,
        )
        print(f"Starte Viewer für: {checkpoint_path}")
        launch_viewer(
            None, agent, num_agents, obs_dim_per_agent, device, checkpoint_path=checkpoint_path,
            team_size=args.team_size, time_limit=args.time_limit,
            hidden_dim=args.hidden_dim, actor_layers=args.actor_layers, critic_layers=args.critic_layers,
        )
        return

    # Watcher-Modus
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        print(f"[ERROR] Verzeichnis nicht gefunden: {checkpoint_dir}")
        sys.exit(1)

    print(f"Watcher gestartet für: {checkpoint_dir}")
    print(f"Prüfe alle {args.poll_interval}s auf neue Checkpoints...")
    print("Drücke STRG+C zum Beenden.\n")

    last_checkpoint = None
    last_mtime = 0
    current_viewer_proc = None
    current_temp_script = None

    try:
        while True:
            checkpoint_path = get_latest_checkpoint(checkpoint_dir, args.pattern)

            if checkpoint_path is None:
                print(f"[Watcher] Kein Checkpoint gefunden. Warte...")
                time.sleep(args.poll_interval)
                continue

            mtime = os.path.getmtime(checkpoint_path)

            if checkpoint_path != last_checkpoint or mtime > last_mtime:
                print(f"[Watcher] Neuer Checkpoint erkannt: {checkpoint_path}")

                # Alten Viewer schließen falls vorhanden
                if current_viewer_proc is not None:
                    print(f"[Watcher] Schließe alten Viewer...")
                    current_viewer_proc.terminate()
                    try:
                        current_viewer_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        current_viewer_proc.kill()
                    if current_temp_script and os.path.exists(current_temp_script):
                        os.remove(current_temp_script)

                try:
                    agent = load_agent(
                        checkpoint_path, device, args.team_size,
                        actor_layers=args.actor_layers,
                        critic_layers=args.critic_layers,
                        hidden_dim=args.hidden_dim,
                        use_layer_norm=args.use_layer_norm,
                    )
                    last_checkpoint = checkpoint_path
                    last_mtime = mtime

                    print(f"[Watcher] Starte Viewer für: {checkpoint_path}")
                    current_viewer_proc, current_temp_script = launch_viewer(
                        None, agent, num_agents, obs_dim_per_agent, device, checkpoint_path=checkpoint_path,
                        team_size=args.team_size, time_limit=args.time_limit,
                        hidden_dim=args.hidden_dim, actor_layers=args.actor_layers, critic_layers=args.critic_layers,
                    )
                    
                    if current_viewer_proc:
                        print(f"[Watcher] Viewer läuft (PID: {current_viewer_proc.pid}). Watcher prüft weiter...")

                except Exception as e:
                    print(f"[Watcher] Fehler beim Laden/Anzeigen: {e}")
                    time.sleep(args.poll_interval)
                    continue
            else:
                time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[Watcher] Beendet durch Benutzer.")
        if current_viewer_proc is not None:
            print("[Watcher] Schließe Viewer...")
            current_viewer_proc.terminate()
            try:
                current_viewer_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                current_viewer_proc.kill()
        if current_temp_script and os.path.exists(current_temp_script):
            os.remove(current_temp_script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live Checkpoint Watcher für MAPPO Soccer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint-dir", type=str,
                        default="logs/soccer_mappo_dynamic_v2_online",
                        help="Verzeichnis, das auf neue Checkpoints überwacht wird")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional: Einzelner Checkpoint zum Anzeigen (deaktiviert Watcher)")
    parser.add_argument("--pattern", type=str, default="*.pt",
                        help="Dateipattern für Checkpoints (z.B. 'best_agent.pt' oder 'checkpoint_*.pt')")
    parser.add_argument("--team-size", type=int, default=2,
                        help="Teamgröße (muss zum Training passen)")
    parser.add_argument("--time-limit", type=float, default=100.0,
                        help="Zeitlimit pro Episode im Viewer")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Wartezeit zwischen Checks in Sekunden")
    
    # Architektur-Parameter (müssen mit Training übereinstimmen)
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden Dimension (muss mit Training übereinstimmen)")
    parser.add_argument("--actor-layers", type=int, default=3,
                        help="Anzahl Actor Layers (muss mit Training übereinstimmen)")
    parser.add_argument("--critic-layers", type=int, default=3,
                        help="Anzahl Critic Layers (muss mit Training übereinstimmen)")
    parser.add_argument("--use-layer-norm", action="store_true", default=False,
                        help="LayerNorm verwenden (muss mit Training übereinstimmen)")

    args = parser.parse_args()
    main(args)
