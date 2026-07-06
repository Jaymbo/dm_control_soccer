"""PPO (Proximal Policy Optimization) agent.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.

On-policy actor-critic with:
  - Clipped surrogate objective (ratio clip ε)
  - Generalised Advantage Estimation (GAE-λ)
  - Entropy bonus for exploration
  - Single Value network V(s) as critic
  - Tanh-squashed Gaussian policy (reuses GaussianPolicy from networks.py)

Designed for dm_control environments with 1-D dict observations.
Compatible with the same train/test/hpo infrastructure as MPO.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.networks import GaussianPolicy, ValueNetwork


class RolloutBuffer:
    """On-policy rollout storage for PPO.

    Stores transitions from a single collection phase, computes GAE-λ
    advantages and discounted returns, then yields minibatches for training.
    """

    def __init__(self, obs_dim, act_dim, size, device='cpu'):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def store(self, obs, act, rew, val, logp, done):
        assert self.ptr < self.max_size, "RolloutBuffer overflow"
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.done[self.ptr] = done
        self.ptr += 1
        self.size = self.ptr

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        """Compute GAE-λ advantages and discounted returns.

        ``done[t]`` indicates whether the episode terminated *after* step t.
        When ``done[t]`` is True, the next-state value should not be
        bootstrapped, so we zero the advantage carry-over.
        """
        adv = 0.0
        for t in reversed(range(self.size)):
            not_done = 1.0 - self.done[t]
            if t == self.size - 1:
                next_val = last_val
            else:
                next_val = self.val[t + 1]
            delta = self.rew[t] + gamma * next_val * not_done - self.val[t]
            adv = delta + gamma * lam * not_done * adv
            self.adv[t] = adv
        self.ret[:self.size] = self.adv[:self.size] + self.val[:self.size]

    def get(self):
        """Return all data as tensors (normalised advantages)."""
        adv = torch.as_tensor(self.adv[:self.size], dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return dict(
            obs=torch.as_tensor(self.obs[:self.size], dtype=torch.float32, device=self.device),
            act=torch.as_tensor(self.act[:self.size], dtype=torch.float32, device=self.device),
            adv=adv,
            ret=torch.as_tensor(self.ret[:self.size], dtype=torch.float32, device=self.device),
            logp_old=torch.as_tensor(self.logp[:self.size], dtype=torch.float32, device=self.device),
        )

    def reset(self):
        self.ptr = 0
        self.size = 0


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit=1.0,
        device='cpu',
        gamma=0.99,
        lam=0.95,            # GAE lambda
        clip_eps=0.2,        # PPO ratio clip
        actor_lr=3e-4,
        critic_lr=1e-3,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_sizes=(256, 256),
        rollout_size=2048,
        update_epochs=10,
        num_minibatches=32,
        clip_v_loss=True,
        clip_v_eps=0.2,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.clip_v_loss = clip_v_loss
        self.clip_v_eps = clip_v_eps

        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes, act_limit).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_sizes).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=critic_lr)

        self.rollout = RolloutBuffer(obs_dim, act_dim, rollout_size, device=self.device)

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
        """Return (action, value, log_prob) for a raw dm_control observation."""
        obs_t = self.obs_to_tensor(obs_np)
        with torch.no_grad():
            action, log_prob, mean = self.policy.sample(obs_t)
            value = self.value(obs_t)
        return action.squeeze(0).cpu().numpy(), value.item(), log_prob.item()

    def store(self, obs, act, rew, val, logp, done):
        o = self.obs_to_tensor(obs).cpu().numpy().flatten()
        self.rollout.store(o, act, rew, val, logp, done)

    # ------------------------------------------------------------------
    # Policy & Value update
    # ------------------------------------------------------------------
    def update(self):
        """Run PPO update over the collected rollout. Returns metrics dict."""
        data = self.rollout.get()
        obs, act, adv, ret, logp_old = (
            data['obs'], data['act'], data['adv'],
            data['ret'], data['logp_old'],
        )
        N = obs.shape[0]
        mb_size = max(1, N // self.num_minibatches)

        results = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'clip_frac': 0.0,
            'approx_kl': 0.0,
        }
        n_updates = 0

        for _ in range(self.update_epochs):
            idxs = np.random.permutation(N)
            for start in range(0, N, mb_size):
                end = start + mb_size
                mb_idx = idxs[start:end]

                mb_obs = obs[mb_idx]
                mb_act = act[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]
                mb_logp_old = logp_old[mb_idx]

                # --- Policy loss (clipped surrogate) ---
                mean, log_std = self.policy(mb_obs)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                # Convert tanh-squashed action back to pre-tanh space
                # for log_prob computation.  Use atanh with numerical clamping.
                act_limit = self.policy.act_limit
                act_clamped = mb_act / act_limit
                act_clamped = act_clamped.clamp(-1 + 1e-6, 1 - 1e-6)
                x_pre = torch.atanh(act_clamped)

                logp = dist.log_prob(x_pre).sum(-1)
                # Tanh correction (same as in GaussianPolicy.log_prob)
                logp = logp - (2 * (np.log(2.0)
                                    - F.logsigmoid(2 * x_pre)
                                    - F.softplus(-2 * x_pre))).sum(-1)

                ratio = (logp - mb_logp_old).exp()
                clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -(torch.min(ratio * mb_adv, clipped_ratio * mb_adv)).mean()

                # Entropy bonus
                entropy = dist.entropy().sum(-1).mean()

                # --- Value loss (with optional clipping) ---
                v_pred = self.value(mb_obs)
                if self.clip_v_loss:
                    v_old_np = self.rollout.val[:self.rollout.size][mb_idx.numpy()] if isinstance(mb_idx, torch.Tensor) else self.rollout.val[:self.rollout.size][mb_idx]
                    v_old = torch.as_tensor(v_old_np, dtype=torch.float32, device=self.device)
                    v_pred_clipped = v_old + (v_pred - v_old).clamp(-self.clip_v_eps, self.clip_v_eps)
                    v_loss_unclipped = (v_pred - mb_ret) ** 2
                    v_loss_clipped = (v_pred_clipped - mb_ret) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(v_pred, mb_ret)

                # --- Total policy loss (with entropy bonus) ---
                total_policy_loss = policy_loss - self.entropy_coef * entropy

                self.policy_optim.zero_grad()
                total_policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optim.step()

                self.value_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optim.step()

                # --- Metrics ---
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                    approx_kl = (mb_logp_old - logp).mean()

                results['policy_loss'] += policy_loss.item()
                results['value_loss'] += value_loss.item()
                results['entropy'] += entropy.item()
                results['clip_frac'] += clip_frac.item()
                results['approx_kl'] += approx_kl.item()
                n_updates += 1

        for k in results:
            results[k] /= max(1, n_updates)

        self.rollout.reset()
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path, total_steps=0, best_eval=-1e9, final_eval=-1e9):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'total_steps': total_steps,
            'best_eval': best_eval,
            'final_eval': final_eval,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        total_steps = ckpt.get('total_steps', 0)
        best_eval = ckpt.get('best_eval', -1e9)
        final_eval = ckpt.get('final_eval', -1e9)
        return total_steps, best_eval, final_eval
