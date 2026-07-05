"""Unit tests for agent/replay_buffer.py."""
import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.replay_buffer import ReplayBuffer


def test_buffer_store_and_size():
    """Buffer size should reflect number of stored items."""
    buf = ReplayBuffer(obs_dim=4, act_dim=2, size=100)
    assert buf.size == 0
    for i in range(10):
        buf.store(np.zeros(4), np.zeros(2), 1.0, np.zeros(4), False)
    assert buf.size == 10


def test_buffer_circular_wraparound():
    """Buffer should wrap around when exceeding max_size."""
    buf = ReplayBuffer(obs_dim=4, act_dim=2, size=5)
    for i in range(8):
        buf.store(np.full(4, i, dtype=np.float32), np.zeros(2), float(i), np.zeros(4), False)
    assert buf.size == 5  # capped at max_size
    # ptr=3 after 8 stores (8 % 5 = 3), so indices 3,4 hold the oldest (i=3,4)
    # and indices 0,1,2 hold the newest (i=5,6,7)
    assert np.allclose(buf.rew_buf[:5], [5, 6, 7, 3, 4])
    assert buf.ptr == 3


def test_buffer_sample_batch_shapes():
    """sample_batch returns tensors with correct shapes."""
    buf = ReplayBuffer(obs_dim=4, act_dim=2, size=100, device='cpu')
    for i in range(20):
        buf.store(np.random.randn(4), np.random.randn(2), 1.0, np.random.randn(4), False)
    batch = buf.sample_batch(batch_size=8)
    assert batch['obs'].shape == (8, 4)
    assert batch['obs2'].shape == (8, 4)
    assert batch['act'].shape == (8, 2)
    assert batch['rew'].shape == (8,)
    assert batch['done'].shape == (8,)
    assert batch['obs'].dtype == torch.float32


def test_buffer_state_dict_roundtrip():
    """state_dict / load_state_dict should preserve all data."""
    buf = ReplayBuffer(obs_dim=4, act_dim=2, size=100)
    for i in range(15):
        buf.store(np.full(4, i, dtype=np.float32),
                  np.full(2, i * 0.1, dtype=np.float32),
                  float(i), np.full(4, i + 1, dtype=np.float32),
                  i == 14)

    state = buf.state_dict()
    assert state['size'] == 15

    buf2 = ReplayBuffer(obs_dim=4, act_dim=2, size=100)
    buf2.load_state_dict(state)
    assert buf2.size == 15
    assert np.allclose(buf2.obs_buf[:15], buf.obs_buf[:15])
    assert np.allclose(buf2.rew_buf[:15], buf.rew_buf[:15])
    assert np.allclose(buf2.done_buf[:15], buf.done_buf[:15])
    assert buf2.ptr == buf.ptr
