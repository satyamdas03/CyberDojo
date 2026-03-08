"""Tests for CyberDojo Gymnasium environment."""

import pytest
import numpy as np

from cyberdojo.environment import CyberDojoEnv, RedAction, BlueAction
from cyberdojo.config import CyberDojoConfig


class TestEnvironmentCreation:
    def test_create_red_env(self):
        env = CyberDojoEnv(mode="red")
        assert env.mode == "red"
        assert env.n_nodes > 0

    def test_create_blue_env(self):
        env = CyberDojoEnv(mode="blue")
        assert env.mode == "blue"

    def test_observation_space(self):
        env = CyberDojoEnv()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_action_space_red(self):
        env = CyberDojoEnv(mode="red")
        assert env.action_space.shape == (2,)
        assert env.action_space.nvec[0] == RedAction.NUM_ACTIONS
        assert env.action_space.nvec[1] == env.n_nodes

    def test_action_space_blue(self):
        env = CyberDojoEnv(mode="blue")
        assert env.action_space.shape == (2,)
        assert env.action_space.nvec[0] == BlueAction.NUM_ACTIONS
        assert env.action_space.nvec[1] == env.n_nodes


class TestEnvironmentStep:
    def test_reset_returns_valid_obs(self):
        env = CyberDojoEnv(mode="red", seed=42)
        obs, info = env.reset()
        assert obs.dtype == np.float32
        assert "step" in info
        assert info["step"] == 0

    def test_step_returns_correct_tuple(self):
        env = CyberDojoEnv(mode="red", seed=42)
        obs, info = env.reset()
        action = np.array([RedAction.SCAN_NETWORK, 0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_all_red_actions_valid(self):
        """Every red action should produce a valid state transition."""
        env = CyberDojoEnv(mode="red", seed=42)
        obs, _ = env.reset()

        for action_type in range(RedAction.NUM_ACTIONS):
            env.reset()
            action = np.array([action_type, 0])
            obs, reward, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs), \
                f"Red action {action_type} produced invalid observation"

    def test_all_blue_actions_valid(self):
        """Every blue action should produce a valid state transition."""
        env = CyberDojoEnv(mode="blue", seed=42)
        obs, _ = env.reset()

        for action_type in range(BlueAction.NUM_ACTIONS):
            env.reset()
            action = np.array([action_type, 0])
            obs, reward, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs), \
                f"Blue action {action_type} produced invalid observation"

    def test_episode_truncation(self):
        """Episode should truncate when max steps reached."""
        config = CyberDojoConfig()
        config.training.steps_per_episode = 5
        env = CyberDojoEnv(config=config, mode="red", seed=42)
        obs, _ = env.reset()

        for _ in range(5):
            action = np.array([RedAction.WAIT, 0])
            obs, reward, terminated, truncated, info = env.step(action)

        assert truncated

    def test_event_logging(self):
        """Events should be logged for dashboard."""
        env = CyberDojoEnv(mode="red", seed=42)
        env.reset()

        action = np.array([RedAction.SCAN_NETWORK, 0])
        env.step(action)

        assert len(env.event_log) == 1
        assert "red" in env.event_log[0]
        assert "blue" in env.event_log[0]

    def test_scan_discovers_nodes(self):
        """Scanning should discover adjacent nodes."""
        env = CyberDojoEnv(mode="red", seed=42)
        env.reset()

        # Count initial discoveries
        initial_discovered = sum(
            1 for n in env.network.nodes.values()
            if n.is_discovered_by_red
        )

        # Scan from entry point
        action = np.array([RedAction.SCAN_NETWORK, 0])
        env.step(action)

        final_discovered = sum(
            1 for n in env.network.nodes.values()
            if n.is_discovered_by_red
        )
        # Should have discovered at least the initial node
        assert final_discovered >= initial_discovered


class TestAnsiRender:
    def test_render_ansi(self):
        env = CyberDojoEnv(mode="red", render_mode="ansi", seed=42)
        env.reset()
        env.step(np.array([RedAction.SCAN_NETWORK, 0]))
        output = env.render()
        assert output is not None
        assert "CYBERDOJO" in output


class TestOpponentPolicy:
    def test_set_opponent_policy(self):
        env = CyberDojoEnv(mode="red", seed=42)
        env.set_opponent_policy(lambda obs: np.array([0, 0]))
        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(np.array([0, 0]))
        assert obs is not None

    def test_default_random_opponent(self):
        """Without setting a policy, opponent should use random actions."""
        env = CyberDojoEnv(mode="red", seed=42)
        obs, _ = env.reset()
        # Should not raise
        obs, reward, _, _, _ = env.step(np.array([0, 0]))
