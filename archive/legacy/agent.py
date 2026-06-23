"""
Actor-Critic Agent für DM Control Soccer.
Zentralisierte Policy für alle 4 Spieler (2-vs-2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    """
    Einfaches MLP-basiertes Actor-Critic Netzwerk.
    
    - Actor: Gibt Mean und Std für jede Action-Dimension aus
    - Critic: Schätzt den Value des aktuellen States
    """
    
    def __init__(self, obs_dim=476, action_dim=12, hidden_dim=256):
        super().__init__()
        
        # Gemeinsame Feature-Extraction Layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor Head - gibt Mean und Log-Std aus
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # learnable std
        
        # Critic Head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialisierung
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Actor mean letzte Layer kleiner initialisieren
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, obs):
        """Forward pass für beide Heads."""
        features = self.shared(obs)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """
        Sample eine Action aus der Policy.
        
        Args:
            obs: Tensor der Observation (batch_size, obs_dim)
            deterministic: Wenn True, verwende nur den Mean (für Evaluation)
        
        Returns:
            action: Tensor der Action
            log_prob: Log-Wahrscheinlichkeit der Action
            value: Geschätzter Value
        """
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.rsample()  # reparameterization trick
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            # Clip actions to [-1, 1]
            action = torch.tanh(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate gegebene Actions unter der aktuellen Policy.
        Wird für PPO-Updates benötigt.
        
        Returns:
            log_prob: Log-Wahrscheinlichkeiten
            value: Values
            entropy: Entropie der Policy (für Regularization)
        """
        action_mean, action_std, value = self.forward(obs)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class ReplayBuffer:
    """
    Einfacher Replay Buffer für Trajektorien.
    Speichert komplette Episoden für On-Policy Learning (PPO).
    """
    
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_batch(self, batch_size=None):
        """
        Returns alle gesammelten Daten als Tensors.
        Wenn batch_size gegeben, sample zufällige Mini-Batches.
        """
        obs = torch.FloatTensor(np.array(self.observations))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(-1)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(-1)
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        values = torch.FloatTensor(np.array(self.values))
        
        if batch_size is not None and batch_size < len(obs):
            indices = np.random.choice(len(obs), batch_size, replace=False)
            return (
                obs[indices], actions[indices], rewards[indices],
                dones[indices], log_probs[indices], values[indices]
            )
        
        return obs, actions, rewards, dones, log_probs, values
    
    def __len__(self):
        return len(self.observations)


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0
    
    # Reverse durch die Trajektorie
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.FloatTensor(advantages).unsqueeze(-1)
    returns = advantages + torch.FloatTensor(values).unsqueeze(-1)
    
    return advantages, returns
