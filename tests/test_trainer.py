"""Tests for CyberDojo trainer and agents."""

import pytest
import numpy as np

from cyberdojo.config import CyberDojoConfig
from cyberdojo.environment import CyberDojoEnv, RedAction, BlueAction
from cyberdojo.agents import (
    RandomAgent, ScriptedRedAgent, ScriptedBlueAgent,
)
from cyberdojo.trainer import CoEvolutionaryTrainer, EloRating, BattleResult


class TestEloRating:
    def test_initial_rating(self):
        elo = EloRating(initial=1000)
        assert elo.get_rating("player1") == 1000

    def test_winner_gains_rating(self):
        elo = EloRating(k_factor=32, initial=1000)
        new_winner, new_loser = elo.update("winner", "loser", episode=1)
        assert new_winner > 1000
        assert new_loser < 1000

    def test_draw_minimal_change(self):
        elo = EloRating(k_factor=32, initial=1000)
        new_a, new_b = elo.update("a", "b", episode=1, draw=True)
        # Equal ratings draw should result in no change
        assert abs(new_a - 1000) < 1
        assert abs(new_b - 1000) < 1

    def test_upset_bigger_change(self):
        """Underdog winning should cause larger rating changes."""
        elo = EloRating(k_factor=32, initial=1000)
        # Give player_a a high rating
        elo.ratings["strong"] = 1400
        elo.ratings["weak"] = 800

        # Weak player wins (upset)
        new_weak, new_strong = elo.update("weak", "strong", episode=1)
        change = new_weak - 800
        assert change > 20  # large change for upset

    def test_history_tracking(self):
        elo = EloRating()
        elo.update("a", "b", episode=1)
        elo.update("b", "a", episode=2)
        assert len(elo.history["a"]) == 2
        assert len(elo.history["b"]) == 2


class TestRandomAgent:
    def test_random_agent_acts(self):
        agent = RandomAgent("red", RedAction.NUM_ACTIONS, 8)
        obs = np.zeros(64, dtype=np.float32)
        action = agent.act(obs)
        assert action.shape == (2,)
        assert 0 <= action[0] < RedAction.NUM_ACTIONS
        assert 0 <= action[1] < 8

    def test_random_agent_clone(self):
        agent = RandomAgent("red", RedAction.NUM_ACTIONS, 8)
        clone = agent.clone()
        assert clone.team == agent.team
        assert clone.n_actions == agent.n_actions


class TestScriptedAgents:
    def test_scripted_red_acts(self):
        agent = ScriptedRedAgent(8)
        obs = np.zeros(64, dtype=np.float32)
        # Set first node as discovered
        obs[0] = 1.0

        for _ in range(20):
            action = agent.act(obs)
            assert action.shape == (2,)
            assert 0 <= action[0] < RedAction.NUM_ACTIONS

    def test_scripted_blue_acts(self):
        agent = ScriptedBlueAgent(8)
        obs = np.zeros(64, dtype=np.float32)

        for _ in range(20):
            action = agent.act(obs)
            assert action.shape == (2,)
            assert 0 <= action[0] < BlueAction.NUM_ACTIONS

    def test_scripted_red_progresses_phases(self):
        """Scripted red should progress through attack phases."""
        agent = ScriptedRedAgent(8)
        obs = np.zeros(64, dtype=np.float32)
        obs[0] = 1.0  # node 0 discovered

        actions_taken = set()
        for _ in range(30):
            action = agent.act(obs)
            actions_taken.add(int(action[0]))

        # Should use multiple different action types
        assert len(actions_taken) > 1


class TestBattleResult:
    def test_to_dict(self):
        result = BattleResult(
            episode=1, red_name="RL-Red", blue_name="RL-Blue",
            winner="red", red_score=45.0, blue_score=30.0,
            steps=50, compromised_nodes=3, total_nodes=8,
            data_stolen=15.0, detections=2,
        )
        d = result.to_dict()
        assert d["winner"] == "red"
        assert d["red_score"] == 45.0
        assert d["episode"] == 1


class TestTrainerComponents:
    def test_trainer_creation(self):
        """Trainer should create without errors."""
        config = CyberDojoConfig()
        config.training.total_episodes = 1
        config.training.steps_per_episode = 10
        trainer = CoEvolutionaryTrainer(config=config, verbose=0)
        assert trainer.red_agent is not None
        assert trainer.blue_agent is not None

    def test_opponent_pools_seeded(self):
        """Opponent pools should be seeded with baselines."""
        config = CyberDojoConfig()
        trainer = CoEvolutionaryTrainer(config=config, verbose=0)
        assert len(trainer.red_pool) >= 2  # scripted + random
        assert len(trainer.blue_pool) >= 2

    def test_run_battle_scripted(self):
        """Should be able to run a battle with scripted agents."""
        config = CyberDojoConfig()
        config.training.steps_per_episode = 20

        n_nodes = sum(config.network.nodes_per_subnet)
        red = ScriptedRedAgent(n_nodes)
        blue = ScriptedBlueAgent(n_nodes)

        trainer = CoEvolutionaryTrainer(
            config=config, red_agent=red, blue_agent=blue, verbose=0
        )
        result = trainer.run_single_battle(visualize=False)

        assert result.winner in ("red", "blue", "draw")
        assert result.steps > 0
        assert result.total_nodes == n_nodes
