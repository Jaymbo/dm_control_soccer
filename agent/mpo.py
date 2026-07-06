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
    ε_μ = 0.001, ε_Σ = 1e-6 (tight to prevent entropy collapse).

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
        critic_lr=1e-4,
        actor_lr=1e-4,
        num_action_samples=20,
        eps_eta=0.1,
        eps_mu=0.001,
        eps_sigma=1e-6,
        dual_lr=0.01,
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

        # Dual variables — optimised via Lagrangian dual loss + Adam (paper
        # Algorithm 2, lines 16 & 19: δη = ∂_η g(η), [δη_μ, δη_Σ] = α ∂L).
        # Stored as raw parameters; actual values = softplus(param) to
        # guarantee positivity without explicit exp (task 7).
        # eta: E-step temperature (dual descent on KL constraint)
        self.log_eta = torch.tensor(0.0, device=self.device, requires_grad=True)
        # lambda_mu: M-step Lagrange multiplier for mean KL constraint
        self.log_lam_mu = torch.tensor(0.0, device=self.device, requires_grad=True)
        # lambda_sigma: M-step Lagrange multiplier for covariance KL constraint
        self.log_lam_sigma = torch.tensor(10.0, device=self.device, requires_grad=True)
        self.dual_optim = torch.optim.SGD(
            [self.log_eta, self.log_lam_mu, self.log_lam_sigma], lr=dual_lr,
        )

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

        Target: r + γ (1 - done) * mean_over_N_a'[ min(Q1'(s',a'), Q2'(s',a')) ]
        Instead of a single next-action sample, we sample N actions from the
        target policy and average the twin-Q min, giving a lower-variance
        estimate of V(s') = E_a'[Q(s',a')] (cf. paper Algorithm 2 line 11
        which uses multiple samples for integral estimation).
        """
        obs, obs2, act, rew, done = batch['obs'], batch['obs2'], batch['act'], batch['rew'], batch['done']
        B = obs2.shape[0]

        with torch.no_grad():
            # Sample N next actions per state from the target policy
            N = self.K
            obs2_rep = obs2.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
            next_act, _, _ = self.policy_target.sample(obs2_rep)
            q1_t, q2_t = self.q_target(obs2_rep, next_act)
            q_next = torch.min(q1_t, q2_t).reshape(B, N).mean(dim=1)
            q_target = rew + self.gamma * (1 - done) * q_next

        q1, q2 = self.q(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=40.0)
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
            # Sample K actions per state from target (old) policy π(·|s, θi)
            # (paper Algorithm 2, line 11: "sample M additional actions ...
            #  from π(a|s, θi)")
            mean, log_std = self.policy_target(obs_rep)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            x_pre = dist.rsample()                      # pre-tanh
            actions = torch.tanh(x_pre) * self.policy.act_limit
            # Use target Q-networks for stable target weights (paper uses
            # Q-target, not the online Q, to avoid positive feedback loops
            # where a moving Q distorts the target distribution).
            q1, q2 = self.q_target(obs_rep, actions)
            q_vals = torch.min(q1, q2).reshape(B, self.K)

        eta = F.softplus(self.log_eta)
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

        # Mean-part KL: denominator uses the *old* (target) scale, not the
        # new one.  This follows reference MPO implementations (e.g. Acme)
        # where the mean displacement is measured relative to the old
        # distribution's scale.
        C_mu = 0.5 * (((mean_new - mean_old) / std_old) ** 2).sum(-1).mean()
        C_sigma = 0.5 * ((std_old / std_new) ** 2 - 1
                         + 2 * (log_std_new - log_std_old)).sum(-1).mean()

        lam_mu = F.softplus(self.log_lam_mu)
        lam_sigma = F.softplus(self.log_lam_sigma)
        kl_loss = lam_mu.detach() * C_mu + lam_sigma.detach() * C_sigma

        actor_loss = nll_loss + kl_loss

        self.policy_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=40.0)
        self.policy_optim.step()

        # --- Measure KL components AFTER the policy update ---
        with torch.no_grad():
            mean_new2, log_std_new2 = self.policy(obs_rep)
            std_new2 = log_std_new2.exp()
            C_mu_after = 0.5 * (((mean_new2 - mean_old) / std_old) ** 2).sum(-1).mean()
            C_sigma_after = 0.5 * ((std_old / std_new2) ** 2 - 1
                                   + 2 * (log_std_new2 - log_std_old)).sum(-1).mean()

        # --- Dual variable updates via Lagrangian dual loss + Adam ---
        # E-step dual (paper Eq. 9): minimise g(η) where
        #   ∂g/∂η = ε - KL(q‖π) = H(q) - (log K - ε) = entropy - action_min_entropy
        # M-step dual (paper D.3): minimise λ_μ(ε_μ - C_μ) + λ_Σ(ε_Σ - C_Σ)
        # When a constraint is violated (C > ε), the gradient is negative,
        # so gradient descent increases the multiplier.  Correct behaviour.
        eta_val = F.softplus(self.log_eta)
        lam_mu_val = F.softplus(self.log_lam_mu)
        lam_sigma_val = F.softplus(self.log_lam_sigma)

        dual_loss = eta_val * (target_entropy.detach() - self.action_min_entropy) \
                    + lam_mu_val * (self.eps_mu - C_mu_after.detach()) \
                    + lam_sigma_val * (self.eps_sigma - C_sigma_after.detach())

        self.dual_optim.zero_grad()
        dual_loss.backward()
        self.dual_optim.step()

        # Lower-bound only (task 11): prevent collapse to negative regime
        with torch.no_grad():
            self.log_eta.clamp_(min=-18.0)
            self.log_lam_mu.clamp_(min=-18.0)
            self.log_lam_sigma.clamp_(min=-18.0)

        return (actor_loss.item(), C_mu_after.item(), C_sigma_after.item(),
                F.softplus(self.log_eta).item(),
                F.softplus(self.log_lam_mu).item(),
                F.softplus(self.log_lam_sigma).item(),
                target_entropy.item())

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------
    def update_targets(self):
        """Hard-copy online networks to targets (paper Algorithm 2).

        MPO uses hard target copies, not Polyak averaging.  The old policy
        π_old for the KL constraint must be the policy from the *previous*
        iteration, not a running average.  Similarly, Q-target should reflect
        the current Q after each update round.
        """
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.copy_(p.data)
            for p, p_t in zip(self.policy.parameters(), self.policy_target.parameters()):
                p_t.data.copy_(p.data)

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
            a_loss, c_mu, c_sigma, eta, lam_mu, lam_sigma, t_ent = self.update_actor(batch)
            results.update(dict(
                actor_loss=a_loss, kl_mu=c_mu, kl_sigma=c_sigma,
                eta=eta, lam_mu=lam_mu, lam_sigma=lam_sigma,
                target_entropy=t_ent,
            ))
        self.update_targets()
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path, total_steps=0, best_eval=-1e9, final_eval=-1e9):
        """Save full training state for resuming.

        Args:
            total_steps: environment steps completed so far.
            best_eval:   best intermediate eval reward achieved during training.
            final_eval:  robust final eval reward (mean over multiple episodes).
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
            'final_eval': final_eval,
        }, path)

    def load(self, path):
        """Load full training state.  Returns (total_steps, best_eval, final_eval)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.q.load_state_dict(ckpt['q'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.policy_target.load_state_dict(ckpt['policy_target'])
        with torch.no_grad():
            self.log_eta.copy_(ckpt['log_eta'])
            self.log_lam_mu.copy_(ckpt['log_lam_mu'])
            self.log_lam_sigma.copy_(ckpt['log_lam_sigma'])
        if 'replay_buffer' in ckpt:
            self.buffer.load_state_dict(ckpt['replay_buffer'])
        total_steps = ckpt.get('total_steps', 0)
        best_eval = ckpt.get('best_eval', -1e9)
        final_eval = ckpt.get('final_eval', -1e9)
        return total_steps, best_eval, final_eval
