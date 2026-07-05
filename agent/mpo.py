"""MPO (Maximum a Posteriori Policy Optimization) agent.

Reference: Abdolmaleki et al., "Maximum a Posteriori Policy Optimization", ICLR 2018.

Simplified but faithful implementation:
  - Twin Q-networks with target networks (SAC-style critic)
  - E-step: sample K actions per state from current policy, compute
    advantage-weighted target distribution via temperature eta (dual descent
    on an entropy constraint).
  - M-step: update policy to minimise weighted NLL under target distribution
    plus Lagrangian KL(π_old || π_new) term with multiplier lambda.

Designed for dm_control environments with 1-D dict observations.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from agent.networks import GaussianPolicy, QNetwork
from agent.replay_buffer import ReplayBuffer


class MPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit=1.0,
        device='cpu',
        gamma=0.99,
        polyak=0.995,
        critic_lr=3e-4,
        actor_lr=1e-4,
        num_action_samples=20,
        action_min_entropy=2.0,
        kl_eps=0.1,
        dual_lr=0.001,
        hidden_sizes=(256, 256),
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.polyak = polyak
        self.K = num_action_samples
        self.action_min_entropy = action_min_entropy
        self.kl_eps = kl_eps
        self.dual_lr = dual_lr

        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, act_limit).to(self.device)
        self.policy_target = copy.deepcopy(self.policy).to(self.device)
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.q = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q_target = copy.deepcopy(self.q).to(self.device)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=critic_lr)

        # Dual variables (learned via multiplicative update, not gradient descent)
        # eta: temperature for E-step, initialised at 1.0
        self.log_eta = torch.tensor(0.0, device=self.device)
        # lambda: Lagrange multiplier for M-step KL constraint
        self.log_lambda = torch.tensor(0.0, device=self.device)

        self.buffer = ReplayBuffer(obs_dim, act_dim, size=200000, device=self.device)

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------
    def obs_to_tensor(self, obs):
        """Convert dm_control observation spec (ordered dict) to flat tensor."""
        if isinstance(obs, dict):
            arr = np.concatenate([np.ravel(v) for v in obs.values()])
        else:
            arr = np.ravel(obs)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def get_action(self, obs_np, deterministic=False):
        """Return a numpy action given a raw dm_control observation (dict)."""
        obs_t = self.obs_to_tensor(obs_np)
        with torch.no_grad():
            action, _ = self.policy.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def store(self, obs, act, rew, next_obs, done):
        o = self.obs_to_tensor(obs).cpu().numpy().flatten()
        o2 = self.obs_to_tensor(next_obs).cpu().numpy().flatten()
        self.buffer.store(o, act, rew, o2, float(done))

    # ------------------------------------------------------------------
    # Critic update (SAC-style double Q-learning)
    # ------------------------------------------------------------------
    def update_critic(self, batch):
        obs, obs2, act, rew, done = batch['obs'], batch['obs2'], batch['act'], batch['rew'], batch['done']

        with torch.no_grad():
            next_act, next_logp, _ = self.policy_target.sample(obs2)
            q1_t, q2_t = self.q_target(obs2, next_act)
            q_target = rew + self.gamma * (1 - done) * (torch.min(q1_t, q2_t) - 0.2 * next_logp)

        q1, q2 = self.q(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()
        return critic_loss.item()

    # ------------------------------------------------------------------
    # E-step: compute target weights w_k for sampled actions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_target_weights(self, obs):
        """For each state, sample K actions from current policy and compute
        target weights w_k = softmax(Q/eta).  Returns weights [B, K],
        sampled pre-tanh actions [B, K, act_dim], and Q-values [B, K]."""
        B = obs.shape[0]
        obs_rep = obs.unsqueeze(1).expand(-1, self.K, -1).reshape(B * self.K, -1)

        with torch.no_grad():
            # Sample K actions per state from current (old) policy
            mean, log_std = self.policy(obs_rep)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            x_pre = dist.rsample()                      # pre-tanh
            actions = torch.tanh(x_pre) * self.policy.act_limit
            q1, q2 = self.q(obs_rep, actions)
            q_vals = torch.min(q1, q2).reshape(B, self.K)

        eta = self.log_eta.exp()
        # Target distribution q(a|s) ∝ exp(Q/eta)
        logits = q_vals / (eta + 1e-8)
        weights = torch.softmax(logits, dim=-1)          # [B, K]

        # Entropy of the target distribution (for dual variable update)
        log_weights = torch.log_softmax(logits, dim=-1)
        entropy = -(weights * log_weights).sum(-1).mean()

        return weights, x_pre.reshape(B, self.K, -1), q_vals, entropy

    # ------------------------------------------------------------------
    # M-step: policy and dual variable updates
    # ------------------------------------------------------------------
    def update_actor(self, batch):
        obs = batch['obs']
        B = obs.shape[0]

        # --- E-step ---
        weights, x_pre, q_vals, target_entropy = self._compute_target_weights(obs)

        # --- M-step: weighted NLL under policy ---
        obs_rep = obs.unsqueeze(1).expand(-1, self.K, -1).reshape(B * self.K, -1)
        x_pre_flat = x_pre.reshape(B * self.K, -1)

        # Log-prob of the sampled (pre-tanh) actions under the *current* policy
        log_pi = self.policy.log_prob(obs_rep, x_pre_flat).reshape(B, self.K)

        # Weighted NLL: -sum_k w_k * log pi(a_k|s)
        nll_loss = -(weights.detach() * log_pi).sum(-1).mean()

        # KL(π_old || π_new) penalty using current lambda
        with torch.no_grad():
            mean_old, log_std_old = self.policy_target(obs_rep)
            std_old = log_std_old.exp()
            log_pi_old = torch.distributions.Normal(mean_old, std_old).log_prob(x_pre_flat).sum(-1).reshape(B, self.K)

        kl_before = (log_pi - log_pi_old).mean()
        lam = self.log_lambda.exp()
        kl_loss = lam.detach() * kl_before

        actor_loss = nll_loss + kl_loss

        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # --- Measure KL AFTER the policy update ---
        with torch.no_grad():
            log_pi_new = self.policy.log_prob(obs_rep, x_pre_flat).reshape(B, self.K)
            kl_after = (log_pi_new - log_pi_old).mean()

        # --- Dual updates (multiplicative, as in original MPO paper) ---
        # eta: ensure target_entropy >= action_min_entropy
        #   If entropy < threshold → eta should increase → log_eta += lr * (threshold - entropy)
        # lambda: ensure KL <= kl_eps
        #   If KL > eps → lambda should increase → log_lambda += lr * (KL - eps)
        with torch.no_grad():
            eta_update = self.dual_lr * (self.action_min_entropy - target_entropy)
            self.log_eta += eta_update
            lam_update = self.dual_lr * (kl_after - self.kl_eps)
            self.log_lambda += lam_update

            self.log_eta.clamp_(-5, 5)
            self.log_lambda.clamp_(-3, 3)

        return actor_loss.item(), kl_after.item(), self.log_eta.exp().item(), self.log_lambda.exp().item()

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------
    def update_targets(self):
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(self.polyak).add_(p.data, alpha=1 - self.polyak)
            for p, p_t in zip(self.policy.parameters(), self.policy_target.parameters()):
                p_t.data.mul_(self.polyak).add_(p.data, alpha=1 - self.polyak)

    # ------------------------------------------------------------------
    # Full update step
    # ------------------------------------------------------------------
    def update(self, batch_size=256, num_critic_updates=1, num_actor_updates=1):
        results = {}
        for _ in range(num_critic_updates):
            batch = self.buffer.sample_batch(batch_size)
            results['critic_loss'] = self.update_critic(batch)
        for _ in range(num_actor_updates):
            batch = self.buffer.sample_batch(batch_size)
            a_loss, kl, eta, lam = self.update_actor(batch)
            results.update(dict(actor_loss=a_loss, kl=kl, eta=eta, lam=lam))
        self.update_targets()
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'policy_target': self.policy_target.state_dict(),
            'log_eta': self.log_eta.item(),
            'log_lambda': self.log_lambda.item(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.q.load_state_dict(ckpt['q'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.policy_target.load_state_dict(ckpt['policy_target'])
        with torch.no_grad():
            self.log_eta.fill_(ckpt['log_eta'])
            self.log_lambda.fill_(ckpt['log_lambda'])
