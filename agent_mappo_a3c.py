"""
MAPPO A3C Agent für Online-Learning.

Unterstützt:
  - N-Step Returns für schnelles Feedback
  - Parallele Environments (Async/Sync)
  - Shared Global Policy mit Lock-free Updates
  - Multi-Agent mit zentralisiertem Critic (CTDE)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network für A3C.
    
    Architecture:
      - Shared Backbone (Observation → Hidden)
      - Actor Head (Hidden → Action Distribution)
      - Critic Head (Hidden → Value)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_layers: int = 2,
        critic_layers: int = 2,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared Backbone
        actor_layers = max(1, actor_layers)
        critic_layers = max(1, critic_layers)
        
        # Actor Network
        actor_modules = []
        for i in range(actor_layers):
            in_dim = obs_dim if i == 0 else hidden_dim
            actor_modules.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                actor_modules.append(nn.LayerNorm(hidden_dim))
            actor_modules.append(nn.Tanh())
        actor_modules.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor_modules)
        
        # Critic Network
        critic_modules = []
        for i in range(critic_layers):
            in_dim = obs_dim if i == 0 else hidden_dim
            critic_modules.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                critic_modules.append(nn.LayerNorm(hidden_dim))
            critic_modules.append(nn.Tanh())
        critic_modules.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*critic_modules)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - returns action mean and value."""
        # obs: (batch, obs_dim)
        actor_out = self.actor(obs)
        action_mean = actor_out  # Continuous actions
        value = self.critic(obs)
        return action_mean, value
    
    def get_action_log_prob(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        entropy_coef: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute log probability of actions given observations.
        
        Returns:
            log_probs: Log probability of actions
            entropy: Entropy of policy
            values: Value estimates
        """
        action_mean, values = self.forward(obs)
        
        # Continuous actions with Gaussian policy
        # Fixed std for simplicity (can be learned)
        action_std = 0.5
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum over action dims
        entropy = dist.entropy().sum(dim=-1).mean()  # Mean entropy
        
        return log_probs, entropy, values.squeeze(-1)
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of sampled action
            value: Value estimate
        """
        action_mean, values = self.forward(obs)
        
        if deterministic:
            action = action_mean
        else:
            action_std = 0.5
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
        
        # Compute log prob for the sampled action
        if not deterministic:
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            log_prob = torch.zeros_like(action_mean[..., 0])
        
        return action, log_prob, values.squeeze(-1)


class MAPPOA3CAgent:
    """
    Multi-Agent PPO A3C Agent.
    
    Features:
      - Shared actor network for all agents
      - Centralized critic (optional)
      - N-Step Returns
      - Async updates compatible
    """
    
    def __init__(
        self,
        obs_dim_per_agent: int,
        action_dim_per_agent: int,
        num_agents: int,
        hidden_dim: int = 256,
        centralized_critic: bool = True,
        actor_layers: int = 2,
        critic_layers: int = 2,
        use_layer_norm: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.obs_dim_per_agent = obs_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        self.num_agents = num_agents
        self.device = device
        self.centralized_critic = centralized_critic
        
        # Per-agent actor networks (shared architecture, separate weights)
        self.actors = nn.ModuleList([
            ActorCriticNetwork(
                obs_dim=obs_dim_per_agent,
                action_dim=action_dim_per_agent,
                hidden_dim=hidden_dim,
                actor_layers=actor_layers,
                critic_layers=1,  # Not used for actors
                use_layer_norm=use_layer_norm,
            ).to(device)
            for _ in range(num_agents)
        ])
        
        # Centralized critic (takes all observations)
        if centralized_critic:
            critic_input_dim = obs_dim_per_agent * num_agents
            self.critic = ActorCriticNetwork(
                obs_dim=critic_input_dim,
                action_dim=1,  # Not used
                hidden_dim=hidden_dim,
                actor_layers=critic_layers,
                critic_layers=1,
                use_layer_norm=use_layer_norm,
            ).to(device)
        else:
            self.critic = None
        
        # Total parameters
        self.total_params = sum(p.numel() for p in self.parameters())
    
    def parameters(self):
        """Get all parameters for optimizer."""
        params = []
        for actor in self.actors:
            params.extend(actor.parameters())
        if self.critic is not None:
            params.extend(self.critic.parameters())
        return params
    
    def get_actions(
        self,
        obs_list: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions for all agents.
        
        Args:
            obs_list: List of observations (one per agent)
            deterministic: Whether to use deterministic actions
        
        Returns:
            actions: List of actions (one per agent)
            log_probs: List of log probabilities
            values: List of value estimates
        """
        actions = []
        log_probs = []
        values = []
        
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            action, log_prob, value = self.actors[i].get_action(
                obs_tensor, deterministic=deterministic
            )
            
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            values.append(value.squeeze(0))
        
        return actions, log_probs, values
    
    def get_critic_value(self, flat_obs: np.ndarray) -> torch.Tensor:
        """
        Get centralized critic value.
        
        Args:
            flat_obs: Flattened observation of all agents
        
        Returns:
            value: Value estimate
        """
        if self.critic is None:
            # Use mean of individual actor values
            values = []
            for i in range(self.num_agents):
                start = i * self.obs_dim_per_agent
                end = start + self.obs_dim_per_agent
                obs = flat_obs[start:end]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                _, value = self.actors[i].forward(obs_tensor)
                values.append(value.squeeze(0))
            return torch.mean(torch.stack(values))
        else:
            obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(self.device)
            _, value = self.critic.forward(obs_tensor)
            return value.squeeze(0)
    
    def compute_loss(
        self,
        obs_batch: np.ndarray,
        actions_batch: np.ndarray,
        returns_batch: np.ndarray,
        agent_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute A3C loss for a single agent.
        
        Args:
            obs_batch: Observations (T, obs_dim)
            actions_batch: Actions (T, action_dim)
            returns_batch: N-Step returns (T,)
            agent_idx: Agent index
        
        Returns:
            policy_loss: Policy gradient loss
            value_loss: Value function loss
            entropy: Entropy bonus
        """
        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        actions_t = torch.FloatTensor(actions_batch).to(self.device)
        returns_t = torch.FloatTensor(returns_batch).to(self.device)
        
        # Get log probs and values
        log_probs, entropy, values = self.actors[agent_idx].get_action_log_prob(
            obs_t, actions_t
        )
        
        # Advantage = Returns - Value
        advantages = returns_t - values
        
        # Policy loss (policy gradient)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns_t.detach())
        
        return policy_loss, value_loss, entropy
    
    def compute_centralized_loss(
        self,
        flat_obs_batch: np.ndarray,
        returns_batch: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute centralized critic loss.
        
        Args:
            flat_obs_batch: Flattened observations (T, num_agents * obs_dim)
            returns_batch: Returns (T,)
        
        Returns:
            value_loss: Value function loss
        """
        if self.critic is None:
            return torch.tensor(0.0, device=self.device)
        
        obs_t = torch.FloatTensor(flat_obs_batch).to(self.device)
        returns_t = torch.FloatTensor(returns_batch).to(self.device)
        
        _, values = self.critic.forward(obs_t)
        values = values.squeeze(-1)
        
        value_loss = F.mse_loss(values, returns_t.detach())
        return value_loss
    
    def state_dict(self):
        """Get state dict for saving."""
        return {
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict() if self.critic else None,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        for i, actor_state in enumerate(state_dict['actors']):
            self.actors[i].load_state_dict(actor_state)
        if state_dict['critic'] is not None and self.critic is not None:
            self.critic.load_state_dict(state_dict['critic'])


class A3CReplayBuffer:
    """
    Simple buffer for A3C N-Step returns.
    
    Stores trajectory for N steps, then computes returns.
    """
    
    def __init__(self, n_steps: int = 5, num_agents: int = 4):
        self.n_steps = n_steps
        self.num_agents = num_agents
        
        self.reset()
    
    def reset(self):
        """Clear buffer."""
        self.observations = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.dones = [[] for _ in range(self.num_agents)]
        self.flat_observations = []
        self.step_count = 0
    
    def add(
        self,
        obs_per_agent: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: np.ndarray,
        dones: List[bool],
        flat_obs: Optional[np.ndarray] = None,
    ):
        """Add a step to buffer."""
        for i in range(self.num_agents):
            self.observations[i].append(obs_per_agent[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.dones[i].append(dones[i])
        
        if flat_obs is not None:
            self.flat_observations.append(flat_obs)
        
        self.step_count += 1
    
    def is_ready(self) -> bool:
        """Check if buffer has enough steps for N-Step return."""
        return self.step_count >= self.n_steps
    
    def get_batch(self, gamma: float = 0.99) -> dict:
        """
        Get batch with N-Step returns.
        
        Returns:
            dict with observations, actions, returns, dones for each agent
        """
        if self.step_count < self.n_steps:
            raise ValueError(f"Buffer has only {self.step_count} steps, need {self.n_steps}")
        
        batch = {
            'observations': [],
            'actions': [],
            'returns': [],
            'flat_observations': [],
        }
        
        for i in range(self.num_agents):
            rewards = np.array(self.rewards[i][:self.n_steps])
            dones = np.array(self.dones[i][:self.n_steps])
            
            # Compute N-Step returns
            returns = self._compute_nstep_returns(rewards, dones, gamma)
            
            # Observations (exclude last state - no action taken from it)
            obs = np.array(self.observations[i][:self.n_steps])
            actions = np.array(self.actions[i][:self.n_steps])
            
            batch['observations'].append(obs)
            batch['actions'].append(actions)
            batch['returns'].append(returns)
        
        if self.flat_observations:
            batch['flat_observations'] = np.array(self.flat_observations[:self.n_steps])
        
        return batch
    
    def _compute_nstep_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Compute N-Step returns with bootstrapping.
        
        R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{n-1}*r_{t+n-1}
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        # Start from last step and work backwards
        G = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                G = 0.0  # Reset at terminal state
            G = rewards[t] + gamma * G
            returns[t] = G
        
        return returns
    
    def clear_old(self, keep_last: int = 0):
        """
        Clear old steps, keep last `keep_last` steps for bootstrapping.
        """
        if keep_last >= self.step_count:
            return
        
        remove_count = self.step_count - keep_last
        
        for i in range(self.num_agents):
            self.observations[i] = self.observations[i][remove_count:]
            self.actions[i] = self.actions[i][remove_count:]
            self.rewards[i] = self.rewards[i][remove_count:]
            self.dones[i] = self.dones[i][remove_count:]
        
        if self.flat_observations:
            self.flat_observations = self.flat_observations[remove_count:]
        
        self.step_count = keep_last


def compute_nstep_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    n_steps: int = 5,
) -> np.ndarray:
    """
    Compute N-Step returns with value bootstrapping.
    
    R_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V(s_{t+n})
    
    Args:
        rewards: Rewards (T,)
        values: Value estimates (T+1,) - includes bootstrap value
        dones: Done flags (T,)
        gamma: Discount factor
        n_steps: N-Step horizon
    
    Returns:
        returns: N-Step returns (T,)
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    
    for t in range(T):
        # Find end of n-step window
        end = min(t + n_steps, T)
        
        G = 0.0
        for k in range(t, end):
            if dones[k]:
                G = 0.0
            G = rewards[k] + gamma * G
        
        # Add bootstrap value if available
        if end < T:
            G += (gamma ** n_steps) * values[end]
        
        returns[t] = G
    
    return returns
