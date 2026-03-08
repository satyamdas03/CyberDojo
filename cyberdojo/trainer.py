"""
CyberDojo Co-Evolutionary Trainer

The heart of CyberDojo: orchestrates adversarial training where
Red and Blue agents co-evolve through competitive self-play with
historical opponent pools and Elo rating tracking.
"""

import os
import time
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cyberdojo.config import CyberDojoConfig
from cyberdojo.environment import CyberDojoEnv, RedAction, BlueAction
from cyberdojo.agents import (
    BaseAgent, RedTeamAgent, BlueTeamAgent,
    RandomAgent, ScriptedRedAgent, ScriptedBlueAgent,
)

logger = logging.getLogger(__name__)


@dataclass
class BattleResult:
    """Result of a single battle between red and blue agents."""
    episode: int
    red_name: str
    blue_name: str
    winner: str  # 'red', 'blue', or 'draw'
    red_score: float
    blue_score: float
    steps: int
    compromised_nodes: int
    total_nodes: int
    data_stolen: float
    detections: int
    events: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "episode": self.episode,
            "red": self.red_name,
            "blue": self.blue_name,
            "winner": self.winner,
            "red_score": self.red_score,
            "blue_score": self.blue_score,
            "steps": self.steps,
            "compromised_nodes": self.compromised_nodes,
            "total_nodes": self.total_nodes,
            "data_stolen": self.data_stolen,
            "detections": self.detections,
        }


class EloRating:
    """Elo rating system for tracking agent skill progression."""

    def __init__(self, k_factor: float = 32.0, initial: float = 1000.0):
        self.k_factor = k_factor
        self.initial = initial
        self.ratings: Dict[str, float] = {}
        self.history: Dict[str, List[Tuple[int, float]]] = {}

    def get_rating(self, name: str) -> float:
        return self.ratings.get(name, self.initial)

    def update(self, winner: str, loser: str, episode: int, draw: bool = False) -> Tuple[float, float]:
        """Update ratings after a battle."""
        r_w = self.get_rating(winner)
        r_l = self.get_rating(loser)

        # Expected scores
        e_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400.0))
        e_l = 1.0 - e_w

        if draw:
            s_w, s_l = 0.5, 0.5
        else:
            s_w, s_l = 1.0, 0.0

        # New ratings
        new_r_w = r_w + self.k_factor * (s_w - e_w)
        new_r_l = r_l + self.k_factor * (s_l - e_l)

        self.ratings[winner] = new_r_w
        self.ratings[loser] = new_r_l

        # Track history
        self.history.setdefault(winner, []).append((episode, new_r_w))
        self.history.setdefault(loser, []).append((episode, new_r_l))

        return new_r_w, new_r_l


class CoEvolutionaryTrainer:
    """
    Orchestrates adversarial co-evolutionary training.
    
    Training loop:
    1. Red agent trains for N steps against the current (or sampled) Blue policy
    2. Blue agent trains for N steps against the current (or sampled) Red policy
    3. Both agents are evaluated against each other
    4. History opponents are sampled to prevent overfitting
    5. Elo ratings are updated
    6. Repeat
    """

    def __init__(
        self,
        config: Optional[CyberDojoConfig] = None,
        red_agent: Optional[BaseAgent] = None,
        blue_agent: Optional[BaseAgent] = None,
        verbose: int = 1,
    ):
        self.config = config or CyberDojoConfig()
        self.verbose = verbose

        # Create environments for each agent
        self.red_env = CyberDojoEnv(config=self.config, mode="red")
        self.blue_env = CyberDojoEnv(config=self.config, mode="blue")

        n_nodes = len(self.red_env.network.nodes)

        # Create or use provided agents
        self.red_agent = red_agent or RedTeamAgent(
            env=self.red_env,
            config={
                "learning_rate": self.config.training.learning_rate,
                "gamma": self.config.training.gamma,
                "gae_lambda": self.config.training.gae_lambda,
                "clip_range": self.config.training.clip_range,
                "n_epochs": self.config.training.n_epochs,
                "batch_size": self.config.training.batch_size,
                "n_steps": self.config.training.steps_per_training_phase,
                "tensorboard_log": self.config.training.tensorboard_log,
            },
        )
        self.blue_agent = blue_agent or BlueTeamAgent(
            env=self.blue_env,
            config={
                "learning_rate": self.config.training.learning_rate,
                "gamma": self.config.training.gamma,
                "gae_lambda": self.config.training.gae_lambda,
                "clip_range": self.config.training.clip_range,
                "n_epochs": self.config.training.n_epochs,
                "batch_size": self.config.training.batch_size,
                "n_steps": self.config.training.steps_per_training_phase,
                "tensorboard_log": self.config.training.tensorboard_log,
            },
        )
        
        # Give agents semantic context of topology names
        node_names = [n.name for _, n in self.red_env.network.nodes.items()]
        self.red_agent.node_names = node_names
        self.blue_agent.node_names = node_names

        # Opponent pools
        self.red_pool: deque = deque(maxlen=self.config.training.opponent_pool_size)
        self.blue_pool: deque = deque(maxlen=self.config.training.opponent_pool_size)

        # Seed pools with scripted baselines
        self.red_pool.append(ScriptedRedAgent(n_nodes))
        self.red_pool.append(RandomAgent("red", RedAction.NUM_ACTIONS, n_nodes))
        self.blue_pool.append(ScriptedBlueAgent(n_nodes))
        self.blue_pool.append(RandomAgent("blue", BlueAction.NUM_ACTIONS, n_nodes))

        # Elo tracking
        self.elo = EloRating(
            k_factor=self.config.training.elo_k_factor,
            initial=self.config.training.initial_elo,
        )

        # Training history
        self.battle_history: List[BattleResult] = []
        self.training_stats: Dict[str, List] = {
            "red_rewards": [],
            "blue_rewards": [],
            "red_elo": [],
            "blue_elo": [],
            "episode_lengths": [],
        }

        # Dashboard callback (set externally)
        self._dashboard_callback = None

    def set_dashboard_callback(self, callback) -> None:
        """Set callback for pushing updates to the dashboard."""
        self._dashboard_callback = callback

    def _sample_opponent(self, team: str) -> BaseAgent:
        """
        Sample an opponent from the pool or use current opponent.
        With probability `opponent_sample_prob`, sample from history.
        """
        pool = self.red_pool if team == "red" else self.blue_pool
        rng = np.random.RandomState()

        if pool and rng.random() < self.config.training.opponent_sample_prob:
            idx = rng.randint(0, len(pool))
            return pool[idx]

        # Use current opponent
        if team == "red":
            return self.red_agent
        return self.blue_agent

    def _run_battle(self, red: BaseAgent, blue: BaseAgent, episode: int) -> BattleResult:
        """Run a single battle between two agents."""
        # Create a fresh environment for the battle
        env = CyberDojoEnv(config=self.config, mode="red")

        # Set blue as the opponent
        env.set_opponent_policy(lambda obs: blue.act(obs))

        obs, info = env.reset()
        total_red_reward = 0.0
        total_blue_reward = 0.0

        for step in range(self.config.training.steps_per_episode):
            red_action = red.act(obs)
            obs, reward, terminated, truncated, info = env.step(red_action)
            total_red_reward += reward

            if self._dashboard_callback:
                self._dashboard_callback(env.step_data)
                time.sleep(0.5)

            if terminated or truncated:
                break

        # Determine winner
        compromised = info.get("compromised_ratio", 0)
        data_stolen = info.get("data_stolen_ratio", 0)
        stats = info.get("stats", {})

        # Score: red wins if significant compromise, blue wins if contained
        red_score = compromised * 50 + data_stolen * 50
        blue_score = (1 - compromised) * 30 + (1 - data_stolen) * 30 + \
                     min(20, stats.get("blue_patches", 0) * 5)

        if red_score > blue_score + 10:
            winner = "red"
        elif blue_score > red_score + 10:
            winner = "blue"
        else:
            winner = "draw"

        result = BattleResult(
            episode=episode,
            red_name=red.name,
            blue_name=blue.name,
            winner=winner,
            red_score=red_score,
            blue_score=blue_score,
            steps=env.current_step,
            compromised_nodes=info.get("compromised_nodes", 0),
            total_nodes=info.get("total_nodes", 0),
            data_stolen=stats.get("red_data_stolen", 0),
            detections=stats.get("red_detections", 0),
            events=env.event_log,
        )

        return result

    def _update_elo(self, result: BattleResult) -> None:
        """Update Elo ratings based on battle result."""
        if result.winner == "red":
            self.elo.update(result.red_name, result.blue_name, result.episode)
        elif result.winner == "blue":
            self.elo.update(result.blue_name, result.red_name, result.episode)
        else:
            self.elo.update(result.red_name, result.blue_name,
                            result.episode, draw=True)

    def train(
        self,
        total_episodes: Optional[int] = None,
        progress_callback=None,
    ) -> Dict:
        """
        Run the full co-evolutionary training loop.
        
        Returns training statistics.
        """
        episodes = total_episodes or self.config.training.total_episodes
        steps_per_phase = self.config.training.steps_per_training_phase

        if self.verbose:
            self._print_header()

        start_time = time.time()

        for episode in range(1, episodes + 1):
            episode_start = time.time()

            # ── Phase 1: Train Red Agent ──
            blue_opponent = self._sample_opponent("blue")
            self.red_env.set_opponent_policy(lambda obs: blue_opponent.act(obs))
            red_result = self.red_agent.learn(
                self.red_env, total_timesteps=steps_per_phase,
            )

            # ── Phase 2: Train Blue Agent ──
            red_opponent = self._sample_opponent("red")
            self.blue_env.set_opponent_policy(lambda obs: red_opponent.act(obs))
            blue_result = self.blue_agent.learn(
                self.blue_env, total_timesteps=steps_per_phase,
            )

            # ── Phase 3: Evaluation Battle ──
            battle_result = self._run_battle(
                self.red_agent, self.blue_agent, episode
            )
            self.battle_history.append(battle_result)

            # ── Phase 4: Update Elo ──
            self._update_elo(battle_result)
            red_elo = self.elo.get_rating(self.red_agent.name)
            blue_elo = self.elo.get_rating(self.blue_agent.name)

            # ── Phase 5: Update Opponent Pools ──
            if episode % 5 == 0:
                try:
                    self.red_pool.append(self.red_agent.clone())
                    self.blue_pool.append(self.blue_agent.clone())
                except Exception:
                    pass  # cloning may fail for non-RL agents

            # Track stats
            self.training_stats["red_rewards"].append(battle_result.red_score)
            self.training_stats["blue_rewards"].append(battle_result.blue_score)
            self.training_stats["red_elo"].append(red_elo)
            self.training_stats["blue_elo"].append(blue_elo)
            self.training_stats["episode_lengths"].append(battle_result.steps)

            # ── Logging ──
            if self.verbose and episode % max(1, episodes // 20) == 0:
                elapsed = time.time() - episode_start
                self._print_episode(
                    episode, episodes, battle_result,
                    red_elo, blue_elo, elapsed,
                )

            # ── Checkpointing ──
            if episode % self.config.training.checkpoint_frequency == 0:
                self._save_checkpoint(episode)

            # ── Progress Callback ──
            if progress_callback:
                progress_callback({
                    "episode": episode,
                    "total": episodes,
                    "battle": battle_result.to_dict(),
                    "red_elo": red_elo,
                    "blue_elo": blue_elo,
                })

        total_time = time.time() - start_time

        if self.verbose:
            self._print_summary(episodes, total_time)

        return {
            "total_episodes": episodes,
            "total_time": total_time,
            "final_red_elo": self.elo.get_rating(self.red_agent.name),
            "final_blue_elo": self.elo.get_rating(self.blue_agent.name),
            "red_wins": sum(1 for b in self.battle_history if b.winner == "red"),
            "blue_wins": sum(1 for b in self.battle_history if b.winner == "blue"),
            "draws": sum(1 for b in self.battle_history if b.winner == "draw"),
            "stats": self.training_stats,
        }

    def run_single_battle(self, visualize: bool = True) -> BattleResult:
        """Run a single battle with optional visualization."""
        if visualize:
            env = CyberDojoEnv(
                config=self.config, mode="red", render_mode="ansi"
            )
        else:
            env = CyberDojoEnv(config=self.config, mode="red")

        env.set_opponent_policy(lambda obs: self.blue_agent.act(obs))

        obs, info = env.reset()
        total_reward = 0

        for step in range(self.config.training.steps_per_episode):
            action = self.red_agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if visualize:
                output = env.render()
                if output:
                    print(output)

            if self._dashboard_callback:
                self._dashboard_callback(env.step_data)
                if visualize:
                    time.sleep(1.0)

            if terminated or truncated:
                break

        stats = info.get("stats", {})
        compromised = info.get("compromised_ratio", 0)
        data_stolen = info.get("data_stolen_ratio", 0)

        red_score = compromised * 50 + data_stolen * 50
        blue_score = (1 - compromised) * 30 + (1 - data_stolen) * 30

        if red_score > blue_score + 10:
            winner = "red"
        elif blue_score > red_score + 10:
            winner = "blue"
        else:
            winner = "draw"

        result = BattleResult(
            episode=0,
            red_name=self.red_agent.name,
            blue_name=self.blue_agent.name,
            winner=winner,
            red_score=red_score,
            blue_score=blue_score,
            steps=env.current_step,
            compromised_nodes=info.get("compromised_nodes", 0),
            total_nodes=info.get("total_nodes", 0),
            data_stolen=stats.get("red_data_stolen", 0),
            detections=stats.get("red_detections", 0),
            events=env.event_log,
        )

        return result

    def benchmark(self) -> Dict:
        """Run agents against scripted baselines."""
        n_nodes = len(self.red_env.network.nodes)
        baselines = {
            "red": [
                ScriptedRedAgent(n_nodes),
                RandomAgent("red", RedAction.NUM_ACTIONS, n_nodes),
            ],
            "blue": [
                ScriptedBlueAgent(n_nodes),
                RandomAgent("blue", BlueAction.NUM_ACTIONS, n_nodes),
            ],
        }

        results = {}

        # Red agent vs Blue baselines
        for blue_baseline in baselines["blue"]:
            battle = self._run_battle(self.red_agent, blue_baseline, 0)
            results[f"RedRL_vs_{blue_baseline.name}"] = battle.to_dict()

        # Blue agent vs Red baselines
        for red_baseline in baselines["red"]:
            env = CyberDojoEnv(config=self.config, mode="blue")
            env.set_opponent_policy(lambda obs: red_baseline.act(obs))

            obs, info = env.reset()
            total_reward = 0.0
            for _ in range(self.config.training.steps_per_episode):
                action = self.blue_agent.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                if term or trunc:
                    break

            results[f"BlueRL_vs_{red_baseline.name}"] = {
                "blue_score": total_reward,
                "compromised_ratio": info.get("compromised_ratio", 0),
            }

        return results

    # ─────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        ckpt_dir = self.config.training.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.red_agent.save(os.path.join(ckpt_dir, f"red_ep{episode}"))
        self.blue_agent.save(os.path.join(ckpt_dir, f"blue_ep{episode}"))

        # Save training stats
        stats_path = os.path.join(ckpt_dir, "training_stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "episode": episode,
                "stats": {k: [float(v) for v in vals]
                          for k, vals in self.training_stats.items()},
                "elo_ratings": dict(self.elo.ratings),
                "battles": [b.to_dict() for b in self.battle_history[-50:]],
            }, f, indent=2)

    def load_checkpoint(self, episode: int) -> None:
        """Load training checkpoint."""
        ckpt_dir = self.config.training.checkpoint_dir
        self.red_agent.load(os.path.join(ckpt_dir, f"red_ep{episode}"))
        self.blue_agent.load(os.path.join(ckpt_dir, f"blue_ep{episode}"))

    # ─────────────────────────────────────────────────────────
    # Pretty Printing
    # ─────────────────────────────────────────────────────────

    def _print_header(self) -> None:
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel.fit(
                "[bold red]🔴 RED TEAM[/] vs [bold blue]🔵 BLUE TEAM[/]\n"
                "[dim]Co-Evolutionary Adversarial Training[/]",
                title="[bold]⚔️  CYBERDOJO ⚔️[/]",
                border_style="bright_cyan",
            ))
        except ImportError:
            print("\n" + "=" * 55)
            print("  ⚔️  CYBERDOJO — Co-Evolutionary Training  ⚔️")
            print("  🔴 RED TEAM  vs  🔵 BLUE TEAM")
            print("=" * 55)

    def _print_episode(
        self, episode: int, total: int, result: BattleResult,
        red_elo: float, blue_elo: float, elapsed: float,
    ) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()

            winner_str = {
                "red": "[bold red]🔴 RED WINS[/]",
                "blue": "[bold blue]🔵 BLUE WINS[/]",
                "draw": "[bold yellow]🤝 DRAW[/]",
            }[result.winner]

            table = Table(title=f"Episode {episode}/{total}", show_header=False)
            table.add_row("Winner", winner_str)
            table.add_row("Red Elo", f"[red]{red_elo:.0f}[/]")
            table.add_row("Blue Elo", f"[blue]{blue_elo:.0f}[/]")
            table.add_row("Compromised", f"{result.compromised_nodes}/{result.total_nodes}")
            table.add_row("Data Stolen", f"{result.data_stolen:.1f}")
            table.add_row("Detections", str(result.detections))
            table.add_row("Steps", str(result.steps))
            table.add_row("Time", f"{elapsed:.1f}s")
            console.print(table)
        except ImportError:
            icon = {"red": "🔴", "blue": "🔵", "draw": "🤝"}[result.winner]
            print(f"\n  Ep {episode}/{total} | {icon} {result.winner.upper()} | "
                  f"Elo R:{red_elo:.0f} B:{blue_elo:.0f} | "
                  f"Comp:{result.compromised_nodes}/{result.total_nodes} | "
                  f"Data:{result.data_stolen:.1f} | "
                  f"Det:{result.detections} | {elapsed:.1f}s")

    def _print_summary(self, episodes: int, total_time: float) -> None:
        red_wins = sum(1 for b in self.battle_history if b.winner == "red")
        blue_wins = sum(1 for b in self.battle_history if b.winner == "blue")
        draws = sum(1 for b in self.battle_history if b.winner == "draw")

        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel.fit(
                f"[bold]Training Complete[/]\n\n"
                f"Episodes: {episodes}\n"
                f"Time: {total_time:.1f}s\n\n"
                f"[red]Red Wins: {red_wins}[/] | "
                f"[blue]Blue Wins: {blue_wins}[/] | "
                f"[yellow]Draws: {draws}[/]\n\n"
                f"Final Elo:\n"
                f"  [red]🔴 Red: {self.elo.get_rating(self.red_agent.name):.0f}[/]\n"
                f"  [blue]🔵 Blue: {self.elo.get_rating(self.blue_agent.name):.0f}[/]",
                title="[bold]📊 Results[/]",
                border_style="bright_green",
            ))
        except ImportError:
            print(f"\n{'='*55}")
            print(f"  Training Complete — {episodes} episodes in {total_time:.1f}s")
            print(f"  Red:{red_wins} Blue:{blue_wins} Draw:{draws}")
            print(f"  Elo — Red:{self.elo.get_rating(self.red_agent.name):.0f} "
                  f"Blue:{self.elo.get_rating(self.blue_agent.name):.0f}")
            print(f"{'='*55}\n")
