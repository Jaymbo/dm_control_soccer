"""
MAPPO (Multi-Agent Proximal Policy Optimization) Agent für Soccer.

Centralized Training with Decentralized Execution (CTDE):
- Jeder Spieler hat seine eigene Policy (Actor)
- Critic kann globale Informationen nutzen (optional)
- Shared Weights für alle Spieler (Parameter Sharing)

Vorteile gegenüber zentralisiertem PPO:
- Bessere Skalierbarkeit auf mehr Spieler
- Jeder Agent lernt individuelle Verhaltensweisen
- Robuster bei partiellen Observations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class MAPPOActor(nn.Module):
    """
    Actor für einen einzelnen Spieler.
    Verarbeitet lokale Observation und gibt Action für diesen Spieler.
    """
    
    def __init__(self, obs_dim_per_agent=119, action_dim_per_agent=3, hidden_dim=256):
        """
        Args:
            obs_dim_per_agent: Observation Dimension pro Spieler (119 für DM Control Soccer)
            action_dim_per_agent: Action Dimension pro Spieler (3 für BoxHead Walker)
            hidden_dim: Hidden Layer Dimension
        """
        super().__init__()
        
        self.obs_dim = obs_dim_per_agent
        self.action_dim = action_dim_per_agent
        
        # Actor Network
        self.network = nn.Sequential(
            nn.Linear(obs_dim_per_agent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Action Output (Mean)
        self.actor_mean = nn.Linear(hidden_dim, action_dim_per_agent)
        
        # Learnable Log-Std (shared across all action dimensions)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim_per_agent))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Last layer smaller initialization
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, obs):
        """
        Args:
            obs: (batch_size, obs_dim_per_agent)
        
        Returns:
            action_mean, action_std
        """
        features = self.network(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        
        return action_mean, action_std
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: (batch_size, obs_dim_per_agent)
            deterministic: Use mean only (for evaluation)
        
        Returns:
            action, log_prob
        """
        action_mean, action_std = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            action = torch.tanh(action)  # Clip to [-1, 1]
        
        return action, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions under current policy.
        
        Returns:
            log_prob, entropy
        """
        action_mean, action_std = self.forward(obs)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    Critic für MAPPO.
    
    Zwei Varianten:
    1. Decentralized Critic: Jeder Spieler hat eigenen Value (lokale Obs)
    2. Centralized Critic: Globaler Value (alle Observations concatenated)
    
    Hier: Flexible Implementierung mit Option für globale Observations.
    """
    
    def __init__(self, obs_dim_per_agent=119, num_agents=4, hidden_dim=256, 
                 centralized=True):
        """
        Args:
            obs_dim_per_agent: Observation Dimension pro Spieler
            num_agents: Anzahl der Spieler
            hidden_dim: Hidden Layer Dimension
            centralized: Wenn True, verwende alle Observations für Critic
        """
        super().__init__()
        
        self.centralized = centralized
        self.num_agents = num_agents
        
        if centralized:
            # Global Critic: concatenated observations from all agents
            critic_input_dim = obs_dim_per_agent * num_agents
        else:
            # Local Critic: single agent observation
            critic_input_dim = obs_dim_per_agent
        
        self.network = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Last layer smaller
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
    
    def forward(self, obs, agent_id=None):
        """
        Args:
            obs: Can be:
                 - (batch_size, obs_dim_per_agent): Single agent obs
                 - (batch_size, num_agents, obs_dim_per_agent): All agents
                 - (batch_size, obs_dim_per_agent * num_agents): Flattened global obs
            agent_id: Ignored for centralized critic
        
        Returns:
            value: (batch_size, 1)
        """
        if self.centralized:
            # Ensure obs is flattened global observation
            if obs.dim() == 3:
                obs = obs.view(obs.size(0), -1)
            # obs shape: (batch_size, obs_dim * num_agents)
        else:
            # Local critic: obs shape (batch_size, obs_dim)
            pass
        
        value = self.network(obs)
        return value


class MAPPOAgent(nn.Module):
    """
    Vollständiger MAPPO Agent für Soccer.
    
    - 4 Actors (einer pro Spieler, shared weights)
    - 1 Critic (centralized oder decentralized)
    """
    
    def __init__(self, obs_dim_per_agent=119, action_dim_per_agent=3, 
                 num_agents=4, hidden_dim=256, centralized_critic=True):
        super().__init__()
        
        self.num_agents = num_agents
        self.obs_dim_per_agent = obs_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        
        # Shared Actor für alle Spieler
        self.actor = MAPPOActor(
            obs_dim_per_agent=obs_dim_per_agent,
            action_dim_per_agent=action_dim_per_agent,
            hidden_dim=hidden_dim
        )
        
        # Critic (centralized oder decentralized)
        self.critic = MAPPOCritic(
            obs_dim_per_agent=obs_dim_per_agent,
            num_agents=num_agents,
            hidden_dim=hidden_dim,
            centralized=centralized_critic
        )
    
    def get_actions(self, all_observations, deterministic=False):
        """
        Get actions for all agents.
        
        Args:
            all_observations: List of observations for each agent
                             [(obs_dim,), (obs_dim,), ...] oder
                             Array (num_agents, obs_dim)
            deterministic: Use deterministic actions
        
        Returns:
            all_actions: List of actions for each agent
            all_log_probs: List of log probabilities
            value: Critic value (global)
        """
        # Convert to tensor
        if isinstance(all_observations, list):
            obs_tensor = torch.FloatTensor(np.array(all_observations))
        else:
            obs_tensor = torch.FloatTensor(all_observations)
        
        # Ensure shape (num_agents, obs_dim)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        num_agents = obs_tensor.size(0)
        
        # Get actions for each agent (shared policy)
        all_actions = []
        all_log_probs = []
        
        for agent_id in range(num_agents):
            agent_obs = obs_tensor[agent_id:agent_id+1]  # (1, obs_dim)
            action, log_prob = self.actor.get_action(agent_obs, deterministic)
            all_actions.append(action.squeeze(0))
            all_log_probs.append(log_prob.squeeze(0))
        
        # Get centralized value
        if self.critic.centralized:
            global_obs = obs_tensor.view(1, -1)  # (1, obs_dim * num_agents)
        else:
            # Use first agent's obs for decentralized critic
            global_obs = obs_tensor[0:1]
        
        value = self.critic(global_obs)
        
        return all_actions, all_log_probs, value
    
    def evaluate_actions(self, all_observations, all_actions):
        """
        Evaluate actions for all agents.
        
        Args:
            all_observations: (num_agents, obs_dim)
            all_actions: (num_agents, action_dim)
        
        Returns:
            all_log_probs: (num_agents, 1)
            value: (1, 1)
            all_entropy: (num_agents, 1)
        """
        obs_tensor = torch.FloatTensor(all_observations)
        actions_tensor = torch.FloatTensor(all_actions)
        
        num_agents = obs_tensor.size(0)
        
        # Evaluate for each agent
        all_log_probs = []
        all_entropy = []
        
        for agent_id in range(num_agents):
            agent_obs = obs_tensor[agent_id:agent_id+1]
            agent_action = actions_tensor[agent_id:agent_id+1]
            log_prob, entropy = self.actor.evaluate_actions(agent_obs, agent_action)
            all_log_probs.append(log_prob.squeeze(0))
            all_entropy.append(entropy.squeeze(0))
        
        # Get value
        if self.critic.centralized:
            global_obs = obs_tensor.view(1, -1)
        else:
            global_obs = obs_tensor[0:1]
        
        value = self.critic(global_obs)
        
        return all_log_probs, value, all_entropy


class MAPPOReplayBuffer:
    """
    Replay Buffer für MAPPO.
    Speichert Trajektorien für alle Agenten.
    """
    
    def __init__(self, max_size=100000, num_agents=4):
        self.max_size = max_size
        self.num_agents = num_agents
        self.reset()
    
    def reset(self):
        self.observations = []  # List of (num_agents, obs_dim)
        self.actions = []       # List of (num_agents, action_dim)
        self.rewards = []       # List of (num_agents, reward)
        self.dones = []         # List of (num_agents, done)
        self.log_probs = []     # List of (num_agents, log_prob)
        self.values = []        # List of values
    
    def add(self, observations, actions, rewards, dones, log_probs, value):
        """
        Args:
            observations: (num_agents, obs_dim)
            actions: (num_agents, action_dim)
            rewards: (num_agents,) or scalar
            dones: (num_agents,) or scalar
            log_probs: (num_agents, 1)
            value: scalar or (1, 1)
        """
        self.observations.append(np.array(observations))
        self.actions.append(np.array(actions))
        
        if np.isscalar(rewards):
            rewards = np.array([rewards] * self.num_agents)
        self.rewards.append(np.array(rewards))
        
        if np.isscalar(dones):
            dones = np.array([dones] * self.num_agents)
        self.dones.append(np.array(dones))
        
        self.log_probs.append(np.array(log_probs))
        
        if np.isscalar(value):
            value = np.array([value])
        self.values.append(np.array(value).flatten())
    
    def get_batch(self):
        """
        Returns all collected data as numpy arrays.
        
        Returns:
            observations: (T, num_agents, obs_dim)
            actions: (T, num_agents, action_dim)
            rewards: (T, num_agents)
            dones: (T, num_agents)
            log_probs: (T, num_agents, 1)
            values: (T,)
        """
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values)
        )
    
    def __len__(self):
        return len(self.observations)


def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """
    Generalized Advantage Estimation für MAPPO.
    Funktioniert identisch zu normalem PPO.
    """
    advantages = []
    gae = 0
    
    # Convert to numpy arrays first for efficiency
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    
    return advantages, returns


def split_obs_by_agent(flat_obs, num_agents=4, obs_dim_per_agent=119):
    """
    Split flattened observation into per-agent observations.
    
    Args:
        flat_obs: Flattened observation (476,) oder (batch, 476)
        num_agents: Anzahl der Spieler
        obs_dim_per_agent: Dimension pro Spieler
    
    Returns:
        List of observations for each agent
    """
    flat_obs = np.array(flat_obs)
    
    if flat_obs.ndim == 1:
        # Single timestep: (obs_dim * num_agents,)
        return [
            flat_obs[i * obs_dim_per_agent:(i + 1) * obs_dim_per_agent]
            for i in range(num_agents)
        ]
    else:
        # Batch: (batch_size, obs_dim * num_agents)
        batch_size = flat_obs.shape[0]
        return [
            flat_obs[:, i * obs_dim_per_agent:(i + 1) * obs_dim_per_agent]
            for i in range(num_agents)
        ]
