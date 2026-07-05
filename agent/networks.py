"""Neural network models for MPO: Gaussian policy and twin Q-networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.Tanh, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Gaussian policy with state-dependent mean and log_std (clamped)."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), act_limit=1.0):
        super().__init__()
        self.act_limit = act_limit
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.Tanh)
        self.mean_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-20.0, 2.0)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x) * self.act_limit
        log_prob = dist.log_prob(x).sum(-1)
        # Correction for tanh squashing
        log_prob = log_prob - (2 * (torch.log(torch.tensor(2.0)) -
                           F.logsigmoid(2 * x) - F.softplus(-2 * x))).sum(-1)
        return action, log_prob, mean

    def log_prob(self, obs, action_pre_tanh):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_pre_tanh).sum(-1)
        log_prob = log_prob - (2 * (torch.log(torch.tensor(2.0)) -
                           F.logsigmoid(2 * action_pre_tanh) -
                           F.softplus(-2 * action_pre_tanh))).sum(-1)
        return log_prob

    def get_action(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean) * self.act_limit, mean
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x) * self.act_limit
        return action, x


class QNetwork(nn.Module):
    """Twin Q-networks: two independent MLPs."""

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.Tanh)
        self.q2 = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.Tanh)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)
