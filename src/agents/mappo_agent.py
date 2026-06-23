"""
Optimierte MAPPO-Implementierung für DM Control Soccer.

Zentrale Verbesserungen gegenüber agent_mappo.py:
- Vollständig vektorisierte Forward/Evaluation für ganze Batches
- Parameter Sharing über alle Agenten via Batch-Dimension
- Korrekte tanh-Korrektur für Log-Probabilities
- Zentrale Critic-Option mit globaler Observation
- Orthogonal Initialization + LayerNorm-Option für stabilere Gradienten
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MAPPOActor(nn.Module):
    """
    Shared Actor für alle Agenten.
    Input:  (batch, obs_dim_per_agent)
    Output: (batch, action_dim_per_agent) Means + Stds
    """

    def __init__(self, obs_dim_per_agent=119, action_dim_per_agent=3,
                 hidden_dim=256, num_layers=5, use_layer_norm=False):
        super().__init__()
        self.obs_dim = obs_dim_per_agent
        self.action_dim = action_dim_per_agent

        layers = []
        in_dim = obs_dim_per_agent
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(hidden_dim, action_dim_per_agent)
        # Pro-Dimension learnable log-std -> feinere Kontrolle pro Gelenk
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim_per_agent))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, obs):
        features = self.network(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(mean)
            return action, log_prob

        dist = Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        # Korrektur wegen tanh (nach SAC-Formel)
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
        actions: bereits im tanh-Raum [-1, 1].
        Wir berechnen Log-Prob der erzeugenden Normalverteilung.
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        
        # Inverses tanh mit Clipping für numerische Stabilität
        # Verhindert NaN wenn actions knapp außerhalb [-1, 1] durch Floating-Point-Fehler
        actions_clipped = torch.clamp(actions, -0.9999, 0.9999)
        raw = 0.5 * (torch.log1p(actions_clipped) - torch.log1p(-actions_clipped))
        
        log_prob = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (math.log(2) - raw - F.softplus(-2 * raw))).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    Critic für MAPPO.
    - central: globale Beobachtung aller Agenten
    - dezentral: lokale Beobachtung eines Agenten
    """

    def __init__(self, obs_dim_per_agent=119, num_agents=4, hidden_dim=256,
                 num_layers=2, centralized=True, use_layer_norm=False):
        super().__init__()
        self.centralized = centralized
        self.num_agents = num_agents
        input_dim = obs_dim_per_agent * num_agents if centralized else obs_dim_per_agent

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, obs):
        # obs: (batch, num_agents, obs_dim) -> flattened
        if self.centralized and obs.dim() == 3:
            obs = obs.view(obs.size(0), -1)
        return self.network(obs)


class MAPPOAgent(nn.Module):
    """
    Vollständiger MAPPO-Agent mit Parameter Sharing.

    get_actions/evaluate_actions unterstützen sowohl
    (num_agents, obs_dim) als auch (batch, num_agents, obs_dim).
    """

    def __init__(self, obs_dim_per_agent=119, action_dim_per_agent=3,
                 num_agents=4, hidden_dim=256, centralized_critic=True,
                 actor_layers=2, critic_layers=2, use_layer_norm=False):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim_per_agent = obs_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent

        self.actor = MAPPOActor(
            obs_dim_per_agent=obs_dim_per_agent,
            action_dim_per_agent=action_dim_per_agent,
            hidden_dim=hidden_dim,
            num_layers=actor_layers,
            use_layer_norm=use_layer_norm,
        )
        self.critic = MAPPOCritic(
            obs_dim_per_agent=obs_dim_per_agent,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            num_layers=critic_layers,
            centralized=centralized_critic,
            use_layer_norm=use_layer_norm,
        )

    def _prepare_obs(self, all_observations):
        if isinstance(all_observations, list):
            all_observations = np.stack(all_observations, axis=0)
        obs = torch.as_tensor(all_observations, dtype=torch.float32)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # (1, num_agents, obs_dim)
        return obs

    def get_actions(self, all_observations, deterministic=False):
        obs = self._prepare_obs(all_observations).to(next(self.parameters()).device)
        batch_size, num_agents, obs_dim = obs.shape
        # Flache Batch-Dimension für shared Actor
        obs_flat = obs.view(batch_size * num_agents, obs_dim)
        actions, log_probs = self.actor.get_action(obs_flat, deterministic=deterministic)
        actions = actions.view(batch_size, num_agents, -1)
        log_probs = log_probs.view(batch_size, num_agents, -1)
        value = self.critic(obs)
        return actions, log_probs, value

    def evaluate_actions(self, all_observations, all_actions):
        obs = self._prepare_obs(all_observations).to(next(self.parameters()).device)
        actions = torch.as_tensor(all_actions, dtype=torch.float32, device=obs.device)
        batch_size, num_agents, obs_dim = obs.shape
        obs_flat = obs.view(batch_size * num_agents, obs_dim)
        actions_flat = actions.view(batch_size * num_agents, -1)
        log_probs, entropy = self.actor.evaluate_actions(obs_flat, actions_flat)
        log_probs = log_probs.view(batch_size, num_agents, -1)
        entropy = entropy.view(batch_size, num_agents, -1)
        value = self.critic(obs)
        return log_probs, value, entropy


class MAPPOReplayBuffer:
    """On-Policy Buffer für MAPPO."""

    def __init__(self, max_size=200000, num_agents=4):
        self.max_size = max_size
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, observations, actions, rewards, dones, log_probs, value):
        self.observations.append(np.asarray(observations, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.values.append(np.asarray(value, dtype=np.float32).flatten())

    def get_batch(self):
        return (
            np.array(self.observations, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
        )

    def __len__(self):
        return len(self.observations)


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    GAE für 1D-Zeitreihen (summierte Rewards, globale Values).
    rewards/values/dones: (T,)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    advantages = np.empty_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


def split_obs_by_agent(flat_obs, num_agents=4, obs_dim_per_agent=119):
    """Teilt flache Beobachtung in Agenten auf."""
    flat_obs = np.asarray(flat_obs)
    if flat_obs.ndim == 1:
        return [flat_obs[i * obs_dim_per_agent:(i + 1) * obs_dim_per_agent]
                for i in range(num_agents)]
    # Batch-Dimension
    return [flat_obs[:, i * obs_dim_per_agent:(i + 1) * obs_dim_per_agent]
            for i in range(num_agents)]
