"""MPO (Maximum a Posteriori Policy Optimization) agent.

Reference: Abdolmaleki et al., "Maximum a Posteriori Policy Optimization", ICLR 2018.

Implementation following the paper:
  - Twin Q-networks with target networks (unregularised Q, no SAC entropy bonus)
  - E-step: sample K actions per state from current policy, compute
    target distribution q(a|s) ∝ π(a|s) exp(Q(s,a)/η) via temperature η.
    η is optimised by dual descent on the KL constraint E[KL(q‖π)] < ε.
  - M-step: update policy to minimise weighted NLL under target distribution
    plus Lagrangian KL(π_old ‖ π_new) penalty with **decoupled** multipliers
    λ_μ (mean) and λ_Σ (covariance), as per Appendix D.3 of the paper.
    ε_μ = 0.1, ε_Σ = 0.0001 (very tight to prevent entropy collapse).

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
        critic_lr=5e-4,
        actor_lr=5e-4,
        num_action_samples=20,
        eps_eta=0.1,
        eps_mu=0.1,
        eps_sigma=0.0001,
        dual_lr=0.001,
        hidden_sizes=(256, 256),
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.polyak = polyak
        self.K = num_action_samples
        self.eps_eta = eps_eta
        self.eps_mu = eps_mu
        self.eps_sigma = eps_sigma
        self.dual_lr = dual_lr

        # E-step entropy floor: KL(q‖π) < ε  ⟺  entropy(q) > log(K) - ε
        self.action_min_entropy = float(np.log(self.K) - eps_eta)

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

        # Dual variables (multiplicative update in log-space, cf. paper Eq. 9 & D.3)
        # eta: E-step temperature (dual descent on KL constraint)
        self.log_eta = torch.tensor(0.0, device=self.device)
        # lambda_mu: M-step Lagrange multiplier for mean KL constraint
        self.log_lam_mu = torch.tensor(0.0, device=self.device)
        # lambda_sigma: M-step Lagrange multiplier for covariance KL constraint
        self.log_lam_sigma = torch.tensor(0.0, device=self.device)

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
        """Unregularised Q-learning (paper Eq. 5: no SAC entropy bonus).

        Target: r + γ (1 - done) * min(Q1', Q2')
        Next actions sampled from the **target** policy (for TD target).
        """
        obs, obs2, act, rew, done = batch['obs'], batch['obs2'], batch['act'], batch['rew'], batch['done']

        with torch.no_grad():
            next_act, _, _ = self.policy_target.sample(obs2)
            q1_t, q2_t = self.q_target(obs2, next_act)
            q_target = rew + self.gamma * (1 - done) * torch.min(q1_t, q2_t)

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
        """M-step (paper Eq. 11, Appendix D.3 Eq. 27).

        Minimises:  -Σ_k w_k log π(a_k|s)  +  λ_μ C_μ  +  λ_Σ C_Σ
        where C_μ is the mean-part and C_Σ the covariance-part of
        KL(π_old ‖ π_new), with **separate** Lagrange multipliers and
        constraints (ε_μ = 0.1, ε_Σ = 0.0001).
        """
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

        # --- Decoupled KL(π_old ‖ π_new) penalty (Appendix D.3, Eq. 27) ---
        # For diagonal Gaussian with old (μ_i, σ_i) and new (μ, σ):
        #   C_μ  = ½ Σ_d (μ_d - μ_i,d)² / σ_d²          (mean part, ε_μ = 0.1)
        #   C_Σ  = ½ Σ_d [σ_i,d²/σ_d² - 1 + 2 log(σ_d/σ_i,d)]  (cov part, ε_Σ = 0.0001)
        mean_new, log_std_new = self.policy(obs_rep)          # [B*K, act_dim]
        std_new = log_std_new.exp()
        with torch.no_grad():
            mean_old, log_std_old = self.policy_target(obs_rep)
            std_old = log_std_old.exp()

        C_mu = 0.5 * (((mean_new - mean_old) / std_new) ** 2).sum(-1).mean()
        C_sigma = 0.5 * ((std_old / std_new) ** 2 - 1
                         + 2 * (log_std_new - log_std_old)).sum(-1).mean()

        lam_mu = self.log_lam_mu.exp()
        lam_sigma = self.log_lam_sigma.exp()
        kl_loss = lam_mu.detach() * C_mu + lam_sigma.detach() * C_sigma

        actor_loss = nll_loss + kl_loss

        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        # --- Measure KL components AFTER the policy update ---
        with torch.no_grad():
            mean_new2, log_std_new2 = self.policy(obs_rep)
            std_new2 = log_std_new2.exp()
            C_mu_after = 0.5 * (((mean_new2 - mean_old) / std_new2) ** 2).sum(-1).mean()
            C_sigma_after = 0.5 * ((std_old / std_new2) ** 2 - 1
                                   + 2 * (log_std_new2 - log_std_old)).sum(-1).mean()

        # --- Dual updates (multiplicative in log-space) ---
        # eta:     entropy(q) must stay ≥ log(K) - ε  →  increase η when too low
        # λ_μ:     C_μ must stay ≤ ε_μ               →  increase λ_μ when exceeded
        # λ_Σ:     C_Σ must stay ≤ ε_Σ               →  increase λ_Σ when exceeded
        with torch.no_grad():
            self.log_eta += self.dual_lr * (self.action_min_entropy - target_entropy)
            self.log_eta.clamp_(-5, 5)

            self.log_lam_mu += self.dual_lr * (C_mu_after - self.eps_mu)
            self.log_lam_mu.clamp_(-3, 3)

            self.log_lam_sigma += self.dual_lr * (C_sigma_after - self.eps_sigma)
            self.log_lam_sigma.clamp_(-3, 3)

        return (actor_loss.item(), C_mu_after.item(), C_sigma_after.item(),
                self.log_eta.exp().item(),
                self.log_lam_mu.exp().item(),
                self.log_lam_sigma.exp().item())

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
    def update(self, batch_size=256, num_critic_updates=10, num_actor_updates=10):
        """Full update step.  Defaults follow paper Algorithm 2: many gradient
        steps between data-collection rounds.  Target networks are updated once
        at the end (paper copies Q-target after M-step; we use Polyak averaging
        for smoother updates)."""
        results = {}
        for _ in range(num_critic_updates):
            batch = self.buffer.sample_batch(batch_size)
            results['critic_loss'] = self.update_critic(batch)
        for _ in range(num_actor_updates):
            batch = self.buffer.sample_batch(batch_size)
            a_loss, c_mu, c_sigma, eta, lam_mu, lam_sigma = self.update_actor(batch)
            results.update(dict(
                actor_loss=a_loss, kl_mu=c_mu, kl_sigma=c_sigma,
                eta=eta, lam_mu=lam_mu, lam_sigma=lam_sigma,
            ))
        self.update_targets()
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path, total_steps=0, best_eval=-1e9):
        """Save full training state for resuming.

        Args:
            total_steps: environment steps completed so far.
            best_eval:   best eval reward achieved so far.
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
            'policy_target': self.policy_target.state_dict(),
            'log_eta': self.log_eta.item(),
            'log_lam_mu': self.log_lam_mu.item(),
            'log_lam_sigma': self.log_lam_sigma.item(),
            'replay_buffer': self.buffer.state_dict(),
            'total_steps': total_steps,
            'best_eval': best_eval,
        }, path)

    def load(self, path):
        """Load full training state.  Returns (total_steps, best_eval)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.q.load_state_dict(ckpt['q'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.policy_target.load_state_dict(ckpt['policy_target'])
        with torch.no_grad():
            self.log_eta.fill_(ckpt['log_eta'])
            self.log_lam_mu.fill_(ckpt['log_lam_mu'])
            self.log_lam_sigma.fill_(ckpt['log_lam_sigma'])
        if 'replay_buffer' in ckpt:
            self.buffer.load_state_dict(ckpt['replay_buffer'])
        total_steps = ckpt.get('total_steps', 0)
        best_eval = ckpt.get('best_eval', -1e9)
        return total_steps, best_eval
