"""
CyberDojo Configuration Management

Centralized configuration for network topology, training hyperparameters,
reward weights, and dashboard settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os


@dataclass
class NetworkConfig:
    """Configuration for the simulated network topology."""
    num_subnets: int = 3
    nodes_per_subnet: List[int] = field(default_factory=lambda: [3, 3, 2])
    connectivity: float = 0.3  # inter-subnet connectivity probability
    vulnerability_density: float = 0.4  # probability a service has a vulnerability
    services_per_node: tuple = (1, 4)  # min, max services per node
    has_dmz: bool = True  # whether to include a DMZ subnet
    has_critical_server: bool = True  # whether to include a high-value target
    scenario_data: Optional[Dict] = None  # if present, override automatic building

    @classmethod
    def small(cls) -> "NetworkConfig":
        """Small network: 3 subnets, 8 nodes."""
        return cls(
            num_subnets=3,
            nodes_per_subnet=[3, 3, 2],
            connectivity=0.3,
            vulnerability_density=0.4,
        )

    @classmethod
    def medium(cls) -> "NetworkConfig":
        """Medium network: 5 subnets, 20 nodes."""
        return cls(
            num_subnets=5,
            nodes_per_subnet=[4, 5, 4, 4, 3],
            connectivity=0.25,
            vulnerability_density=0.35,
        )

    @classmethod
    def large(cls) -> "NetworkConfig":
        """Large network: 10 subnets, 50 nodes."""
        return cls(
            num_subnets=10,
            nodes_per_subnet=[5, 6, 5, 5, 4, 5, 5, 5, 5, 5],
            connectivity=0.2,
            vulnerability_density=0.3,
        )


@dataclass
class TrainingConfig:
    """Configuration for co-evolutionary training."""
    total_episodes: int = 1000
    steps_per_episode: int = 100
    steps_per_training_phase: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    n_epochs: int = 10  # PPO epochs per update
    batch_size: int = 64
    opponent_pool_size: int = 10  # number of historical opponents to keep
    opponent_sample_prob: float = 0.3  # probability of sampling historical opponent
    elo_k_factor: float = 32.0  # Elo rating K-factor
    initial_elo: float = 1000.0
    checkpoint_frequency: int = 50  # save every N episodes
    tensorboard_log: str = "./logs/tensorboard/"
    checkpoint_dir: str = "./checkpoints/"


@dataclass
class RewardConfig:
    """Reward weights for Red and Blue agents."""
    # Red Team rewards
    red_compromise_node: float = 10.0
    red_root_access: float = 15.0
    red_data_exfiltration: float = 25.0
    red_persistence: float = 8.0
    red_lateral_move: float = 5.0
    red_stealth_bonus: float = 3.0
    red_detected_penalty: float = -20.0
    red_blocked_penalty: float = -5.0
    red_critical_target: float = 50.0

    # Blue Team rewards
    blue_detect_attacker: float = 15.0
    blue_contain_threat: float = 20.0
    blue_patch_vuln: float = 5.0
    blue_false_positive_penalty: float = -8.0
    blue_downtime_penalty: float = -10.0
    blue_honeypot_catch: float = 25.0
    blue_full_containment_bonus: float = 40.0
    blue_data_breach_penalty: float = -30.0

    # Curriculum scaling
    difficulty_scaling: bool = True
    scaling_start_episode: int = 100
    scaling_factor: float = 1.5


@dataclass
class DashboardConfig:
    """Configuration for the web dashboard."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    update_interval_ms: int = 200  # how often to push updates (ms)


@dataclass
class CyberDojoConfig:
    """Top-level configuration combining all sub-configs."""
    network: NetworkConfig = field(default_factory=NetworkConfig.small)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "network": self.network.__dict__,
                "training": self.training.__dict__,
                "rewards": self.rewards.__dict__,
                "dashboard": self.dashboard.__dict__,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CyberDojoConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        if "network" in data:
            config.network = NetworkConfig(**data["network"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "rewards" in data:
            config.rewards = RewardConfig(**data["rewards"])
        if "dashboard" in data:
            config.dashboard = DashboardConfig(**data["dashboard"])
        return config
