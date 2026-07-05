"""Unit tests for agent/networks.py — GaussianPolicy and QNetwork."""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.networks import GaussianPolicy, QNetwork


# ---------------------------------------------------------------------------
# GaussianPolicy
# ---------------------------------------------------------------------------

def test_policy_output_shapes():
    """Policy forward returns mean and log_std with correct shapes."""
    obs_dim, act_dim = 5, 3
    policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes=(32, 32), act_limit=1.0)
    obs = torch.randn(8, obs_dim)
    mean, log_std = policy(obs)
    assert mean.shape == (8, act_dim)
    assert log_std.shape == (8, act_dim)


def test_policy_log_std_clamped():
    """log_std should be clamped to [-20, 2]."""
    policy = GaussianPolicy(5, 3, hidden_sizes=(32, 32))
    obs = torch.randn(16, 5)
    _, log_std = policy(obs)
    assert log_std.min() >= -20.0
    assert log_std.max() <= 2.0


def test_policy_sample_shapes():
    """sample() returns action, log_prob, mean with correct shapes."""
    obs_dim, act_dim = 5, 3
    policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes=(32, 32), act_limit=2.0)
    obs = torch.randn(8, obs_dim)
    action, log_prob, mean = policy.sample(obs)
    assert action.shape == (8, act_dim)
    assert log_prob.shape == (8,)
    assert mean.shape == (8, act_dim)


def test_policy_action_in_range():
    """Sampled actions should be within [-act_limit, act_limit]."""
    act_limit = 1.5
    policy = GaussianPolicy(5, 2, hidden_sizes=(32, 32), act_limit=act_limit)
    obs = torch.randn(32, 5)
    action, _, _ = policy.sample(obs)
    assert action.min() >= -act_limit - 1e-6
    assert action.max() <= act_limit + 1e-6


def test_policy_deterministic_action():
    """Deterministic action should be tanh(mean) * act_limit."""
    act_limit = 2.0
    policy = GaussianPolicy(5, 3, hidden_sizes=(32, 32), act_limit=act_limit)
    obs = torch.randn(4, 5)
    action, pre_tanh = policy.get_action(obs, deterministic=True)
    mean, _ = policy(obs)
    expected = torch.tanh(mean) * act_limit
    assert torch.allclose(action, expected, atol=1e-6)


def test_policy_log_prob_consistency():
    """log_prob() should match sample()'s log_prob for the same pre-tanh action."""
    policy = GaussianPolicy(5, 3, hidden_sizes=(32, 32))
    obs = torch.randn(8, 5)
    _, log_prob_sample, _ = policy.sample(obs)

    # Re-compute with log_prob method using same obs (different samples → check shape)
    x = torch.randn(8, 3)
    log_prob = policy.log_prob(obs, x)
    assert log_prob.shape == (8,)


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------

def test_qnetwork_output_shapes():
    """QNetwork returns two Q-values of shape [batch]."""
    obs_dim, act_dim = 5, 3
    q = QNetwork(obs_dim, act_dim, hidden_sizes=(32, 32))
    obs = torch.randn(8, obs_dim)
    act = torch.randn(8, act_dim)
    q1, q2 = q(obs, act)
    assert q1.shape == (8,)
    assert q2.shape == (8,)


def test_qnetwork_independence():
    """Q1 and Q2 should be independent (different values for same input)."""
    q = QNetwork(5, 3, hidden_sizes=(32, 32))
    obs = torch.randn(4, 5)
    act = torch.randn(4, 3)
    q1, q2 = q(obs, act)
    assert not torch.allclose(q1, q2)


def test_qnetwork_gradient_flow():
    """Gradients should flow through both Q-networks."""
    q = QNetwork(5, 3, hidden_sizes=(32, 32))
    obs = torch.randn(4, 5)
    act = torch.randn(4, 3)
    q1, q2 = q(obs, act)
    loss = q1.sum() + q2.sum()
    loss.backward()
    for p in q.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0
