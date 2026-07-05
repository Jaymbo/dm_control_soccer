"""Simple replay buffer for off-policy RL."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=100000, device='cpu'):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device),
            obs2=torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32, device=self.device),
            act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32, device=self.device),
            rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32, device=self.device),
            done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device),
        )
        return batch
