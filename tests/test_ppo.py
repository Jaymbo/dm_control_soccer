"""Unit tests for agent/ppo.py — PPO agent core logic.

These tests do NOT require dm_control or MuJoCo.  They use synthetic
observation/action tensors to validate rollout storage, GAE computation,
PPO update, and save/load.
"""
import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.ppo import PPO, RolloutBuffer


@pytest.fixture
def agent():
    """Create a small PPO agent for testing."""
    return PPO(
        obs_dim=6, act_dim=2, act_limit=1.0,
        device='cpu', gamma=0.99, lam=0.95,
        clip_eps=0.2, actor_lr=1e-3, critic_lr=1e-3,
        entropy_coef=0.01, value_coef=0.5,
        max_grad_norm=0.5, hidden_sizes=(32, 32),
        rollout_size=64, update_epochs=2, num_minibatches=4,
    )


def _fill_rollout(agent, n=64):
    """Fill the agent's rollout buffer with random transitions."""
    for _ in range(n):
        obs = np.random.randn(6).astype(np.float32)
        act = np.random.randn(2).astype(np.float32)
        rew = float(np.random.randn())
        val = float(np.random.randn())
        logp = float(np.random.randn())
        agent.store(obs, act, rew, val, logp, done=False)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_agent_initialisation(agent):
    """Agent should initialise with expected attributes."""
    assert agent.gamma == 0.99
    assert agent.lam == 0.95
    assert agent.clip_eps == 0.2
    assert agent.policy is not None
    assert agent.value is not None
    assert agent.rollout.max_size == 64


def test_agent_action_shape(agent):
    """get_action returns numpy action, float value, float log_prob."""
    obs = {'pos': np.array([0.1, 0.2]), 'vel': np.array([0.0, 0.1, 0.2, 0.3])}
    action, val, logp = agent.get_action(obs, deterministic=False)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert isinstance(val, float)
    assert isinstance(logp, float)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

def test_rollout_buffer_store_and_reset():
    """RolloutBuffer stores and resets correctly."""
    buf = RolloutBuffer(obs_dim=4, act_dim=2, size=10, device='cpu')
    for i in range(10):
        buf.store(np.zeros(4), np.zeros(2), 1.0, 0.5, -0.3, False)
    assert buf.size == 10
    assert buf.ptr == 10
    buf.reset()
    assert buf.size == 0
    assert buf.ptr == 0


def test_rollout_buffer_gae():
    """GAE should produce advantages and returns of correct shape."""
    buf = RolloutBuffer(obs_dim=4, act_dim=2, size=5, device='cpu')
    for i in range(5):
        buf.store(np.zeros(4), np.zeros(2), float(i), 0.0, 0.0, False)
    buf.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)
    assert buf.adv.shape == (5,)
    assert buf.ret.shape == (5,)
    # Returns should be finite
    assert np.all(np.isfinite(buf.ret))


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def test_update_returns_metrics(agent):
    """update() should return a dict with expected keys."""
    _fill_rollout(agent, 64)
    agent.rollout.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)
    results = agent.update()
    assert 'policy_loss' in results
    assert 'value_loss' in results
    assert 'entropy' in results
    assert 'clip_frac' in results
    assert 'approx_kl' in results
    for v in results.values():
        assert np.isfinite(v)


def test_update_resets_rollout(agent):
    """After update(), the rollout buffer should be reset."""
    _fill_rollout(agent, 64)
    agent.rollout.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)
    agent.update()
    assert agent.rollout.size == 0
    assert agent.rollout.ptr == 0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, agent):
    """Save and load should restore policy and value weights."""
    _fill_rollout(agent, 64)
    agent.rollout.compute_gae(last_val=0.0, gamma=0.99, lam=0.95)
    agent.update()

    save_path = str(tmp_path / 'test_ppo_ckpt.pt')
    agent.save(save_path, total_steps=999, best_eval=42.0, final_eval=38.5)

    agent2 = PPO(obs_dim=6, act_dim=2, hidden_sizes=(32, 32), device='cpu',
                 rollout_size=64)
    total_steps, best_eval, final_eval = agent2.load(save_path)

    assert total_steps == 999
    assert best_eval == 42.0
    assert final_eval == 38.5

    # Policy weights should match
    p1 = list(agent.policy.parameters())[0].data
    p2 = list(agent2.policy.parameters())[0].data
    torch.testing.assert_close(p1, p2)

    # Value weights should match
    v1 = list(agent.value.parameters())[0].data
    v2 = list(agent2.value.parameters())[0].data
    torch.testing.assert_close(v1, v2)


# ---------------------------------------------------------------------------
# ValueNetwork (from networks.py)
# ---------------------------------------------------------------------------

def test_value_network_output_shape():
    """ValueNetwork should output shape [batch_size]."""
    from agent.networks import ValueNetwork
    vn = ValueNetwork(obs_dim=6, hidden_sizes=(32, 32))
    obs = torch.randn(8, 6)
    out = vn(obs)
    assert out.shape == (8,)
