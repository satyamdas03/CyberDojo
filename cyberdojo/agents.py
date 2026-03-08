"""
CyberDojo AI Agents

Red Team (attacker) and Blue Team (defender) agent implementations
using Stable-Baselines3 for RL training, plus scripted baselines.
"""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True

    # ── Monkey-patch for Python 3.14 compatibility ──
    # SB3's OnPolicyAlgorithm.dump_logs() crashes because
    # ep_info_buffer can contain ints instead of dicts in Py3.14.
    try:
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

        _orig_dump_logs = OnPolicyAlgorithm.dump_logs

        def _patched_dump_logs(self, iteration):
            try:
                _orig_dump_logs(self, iteration)
            except TypeError:
                # Gracefully skip the broken len() call on int items
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

        OnPolicyAlgorithm.dump_logs = _patched_dump_logs
    except Exception:
        pass  # If patching fails, SB3 will still work for non-training use

except ImportError:
    HAS_SB3 = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base class for all CyberDojo agents."""

    def __init__(self, name: str, team: str):
        self.name = name
        self.team = team  # 'red' or 'blue'
        self.elo_rating = 1000.0
        self.games_played = 0
        self.wins = 0
        self.node_names: List[str] = []

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Choose an action given an observation."""
        pass

    @abstractmethod
    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        """Train the agent on the environment."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseAgent":
        """Load the agent from disk."""
        pass

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)

    def clone(self) -> "BaseAgent":
        """Create a snapshot of the current agent for the opponent pool."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# Random Agent (Baseline)
# ─────────────────────────────────────────────────────────────

class RandomAgent(BaseAgent):
    """Agent that takes random actions — used as a baseline."""

    def __init__(self, team: str, n_actions: int, n_nodes: int, seed: int = 42):
        super().__init__(name=f"Random-{team.title()}", team=team)
        self.n_actions = n_actions
        self.n_nodes = n_nodes
        self.rng = np.random.RandomState(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.array([
            self.rng.randint(0, self.n_actions),
            self.rng.randint(0, self.n_nodes),
        ])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "RandomAgent does not learn"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "RandomAgent":
        return self

    def clone(self) -> "RandomAgent":
        return RandomAgent(self.team, self.n_actions, self.n_nodes)


# ─────────────────────────────────────────────────────────────
# Scripted Red Agent (Baseline)
# ─────────────────────────────────────────────────────────────

class ScriptedRedAgent(BaseAgent):
    """
    Hand-coded attack strategy for baseline comparison.
    
    Strategy: Methodical attacker
    1. Scan network from entry point
    2. Scan for vulnerabilities on discovered nodes
    3. Exploit the best vulnerability found
    4. Escalate privileges
    5. Lateral move to next subnet
    6. Install backdoor on compromised nodes
    7. Exfiltrate data from high-value nodes
    """

    def __init__(self, n_nodes: int):
        super().__init__(name="Scripted-Red", team="red")
        self.n_nodes = n_nodes
        self.phase = 0  # current attack phase
        self.step_count = 0
        self.current_target = 0
        self.rng = np.random.RandomState(42)

    def act(self, observation: np.ndarray) -> np.ndarray:
        self.step_count += 1
        obs = observation.reshape(-1, 8)  # reshape to per-node features

        # Find discovered and compromised nodes
        discovered = [i for i in range(min(len(obs), self.n_nodes)) if obs[i][0] > 0.5]
        compromised = [i for i in range(min(len(obs), self.n_nodes)) if obs[i][4] > 0]

        if not discovered:
            # Scan from node 0 (entry point)
            return np.array([0, 0])  # scan_network, node 0

        if self.phase == 0:  # Scanning phase
            target = discovered[-1]  # scan from latest discovery
            if len(discovered) > 3 or self.step_count > 5:
                self.phase = 1
            return np.array([0, target])  # scan_network

        elif self.phase == 1:  # Vulnerability scanning
            # Pick an uncompromised discovered node
            targets = [i for i in discovered if i not in compromised]
            if targets:
                target = targets[0]
                self.phase = 2
                self.current_target = target
                return np.array([1, target])  # scan_vulnerability
            else:
                self.phase = 3
                return np.array([11, 0])  # wait

        elif self.phase == 2:  # Exploitation
            target = self.current_target
            if target in compromised:
                self.phase = 3
            else:
                if self.step_count % 3 == 0:
                    self.phase = 3  # advance even if stuck
                return np.array([2, target])  # exploit

        elif self.phase == 3:  # Privilege escalation
            if compromised:
                target = compromised[-1]
                if obs[target][4] >= 1.0:  # already root
                    self.phase = 4
                else:
                    self.phase = 4  # advance after trying
                    return np.array([3, target])  # privilege_escalate
            self.phase = 4

        elif self.phase == 4:  # Lateral movement
            if compromised:
                target = self.rng.choice(compromised)
                self.phase = 5
                return np.array([4, target])  # lateral_move
            self.phase = 5

        elif self.phase == 5:  # Persistence + Exfiltration
            if compromised:
                target = compromised[-1]
                if self.step_count % 2 == 0:
                    return np.array([5, target])  # install_backdoor
                else:
                    self.phase = 0  # restart cycle
                    return np.array([6, target])  # exfiltrate_data
            self.phase = 0  # restart cycle

        # Default: scan from a random discovered node
        self.phase = 0
        target = self.rng.choice(discovered) if discovered else 0
        return np.array([0, target])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "ScriptedRedAgent does not learn"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "ScriptedRedAgent":
        return self

    def clone(self) -> "ScriptedRedAgent":
        return ScriptedRedAgent(self.n_nodes)


# ─────────────────────────────────────────────────────────────
# Scripted Blue Agent (Baseline)
# ─────────────────────────────────────────────────────────────

class ScriptedBlueAgent(BaseAgent):
    """
    Hand-coded defense strategy for baseline comparison.
    
    Strategy: Alert-driven responder
    1. Monitor traffic on all subnets
    2. When alerts appear, analyze them
    3. Isolate confirmed threats
    4. Patch known vulnerabilities
    5. Restore compromised nodes
    6. Deploy honeypots proactively
    """

    def __init__(self, n_nodes: int):
        super().__init__(name="Scripted-Blue", team="blue")
        self.n_nodes = n_nodes
        self.rng = np.random.RandomState(42)
        self.step_count = 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        self.step_count += 1
        obs = observation.reshape(-1, 8)  # reshape to per-node features

        # Find nodes with high alert levels
        alert_nodes = []
        for i in range(min(len(obs), self.n_nodes)):
            if obs[i][0] > 0.3:  # alert_level
                alert_nodes.append((i, obs[i][0]))

        alert_nodes.sort(key=lambda x: x[1], reverse=True)

        # Find isolated nodes
        isolated = [i for i in range(min(len(obs), self.n_nodes)) if obs[i][3] > 0.5]

        # Priority 1: Analyze high alerts
        if alert_nodes and alert_nodes[0][1] > 0.5:
            target = alert_nodes[0][0]
            if alert_nodes[0][1] > 0.7:
                return np.array([2, target])  # isolate (high confidence)
            return np.array([1, target])  # analyze_alert

        # Priority 2: Monitor traffic (rotate subnets)
        if self.step_count % 5 == 0:
            target = (self.step_count // 5) % self.n_nodes
            return np.array([0, target])  # monitor_traffic

        # Priority 3: Patch vulnerabilities
        if self.step_count % 3 == 0:
            # Patch nodes with known vulns
            vuln_nodes = [
                i for i in range(min(len(obs), self.n_nodes))
                if obs[i][2] > 0  # has known vulns
            ]
            if vuln_nodes:
                return np.array([3, vuln_nodes[0]])  # patch_vulnerability

        # Priority 4: Deploy honeypots early
        if self.step_count < 10 and self.step_count % 4 == 0:
            target = self.rng.randint(0, self.n_nodes)
            return np.array([4, target])  # deploy_honeypot

        # Priority 5: Forensics on suspicious but not yet alerted nodes
        if alert_nodes:
            target = alert_nodes[-1][0]  # lowest alert, investigate
            return np.array([7, target])  # forensic_analysis

        # Priority 6: Restore isolated nodes
        if isolated:
            return np.array([5, isolated[0]])  # restore_backup

        # Default: monitor
        target = self.rng.randint(0, self.n_nodes)
        return np.array([0, target])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "ScriptedBlueAgent does not learn"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "ScriptedBlueAgent":
        return self

    def clone(self) -> "ScriptedBlueAgent":
        return ScriptedBlueAgent(self.n_nodes)


# ─────────────────────────────────────────────────────────────
# RL Red Team Agent
# ─────────────────────────────────────────────────────────────

class RedTeamAgent(BaseAgent):
    """Red Team agent using PPO from Stable-Baselines3."""

    def __init__(self, env: Any = None, config: Optional[Dict] = None):
        super().__init__(name="RL-Red", team="red")
        self.config = config or {}
        self.model = None
        self._env = env

        if env is not None and HAS_SB3:
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.get("learning_rate", 3e-4),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_range=self.config.get("clip_range", 0.2),
                n_epochs=self.config.get("n_epochs", 10),
                batch_size=self.config.get("batch_size", 64),
                n_steps=self.config.get("n_steps", 2048),
                verbose=0,
                tensorboard_log=self.config.get("tensorboard_log"),
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                ),
            )

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([0, 0])
        action, _ = self.model.predict(observation, deterministic=False)
        return action

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        if not HAS_SB3:
            logger.warning("Stable-Baselines3 not installed. Cannot train.")
            return {"error": "SB3 not installed"}

        if self.model is None:
            self.model = PPO("MlpPolicy", env, verbose=0,
                             policy_kwargs=dict(
                                 net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                             ))
        else:
            self.model.set_env(env)

        callback = kwargs.get("callback")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=kwargs.get("progress_bar", False),
        )
        return {"timesteps": total_timesteps}

    def save(self, path: str) -> None:
        if self.model is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.model.save(path)

    def load(self, path: str) -> "RedTeamAgent":
        if HAS_SB3:
            self.model = PPO.load(path)
        return self

    def clone(self) -> "RedTeamAgent":
        """Create a snapshot for the opponent pool."""
        import tempfile
        agent = RedTeamAgent(config=self.config)
        agent.elo_rating = self.elo_rating
        agent.games_played = self.games_played
        if self.model is not None:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                self.model.save(f.name)
                agent.model = PPO.load(f.name)
                os.unlink(f.name)
        return agent


# ─────────────────────────────────────────────────────────────
# RL Blue Team Agent
# ─────────────────────────────────────────────────────────────

class BlueTeamAgent(BaseAgent):
    """Blue Team agent using PPO from Stable-Baselines3."""

    def __init__(self, env: Any = None, config: Optional[Dict] = None):
        super().__init__(name="RL-Blue", team="blue")
        self.config = config or {}
        self.model = None
        self._env = env

        if env is not None and HAS_SB3:
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.get("learning_rate", 3e-4),
                gamma=self.config.get("gamma", 0.99),
                gae_lambda=self.config.get("gae_lambda", 0.95),
                clip_range=self.config.get("clip_range", 0.2),
                n_epochs=self.config.get("n_epochs", 10),
                batch_size=self.config.get("batch_size", 64),
                n_steps=self.config.get("n_steps", 2048),
                verbose=0,
                tensorboard_log=self.config.get("tensorboard_log"),
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                ),
            )

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.array([0, 0])
        action, _ = self.model.predict(observation, deterministic=False)
        return action

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        if not HAS_SB3:
            logger.warning("Stable-Baselines3 not installed. Cannot train.")
            return {"error": "SB3 not installed"}

        if self.model is None:
            self.model = PPO("MlpPolicy", env, verbose=0,
                             policy_kwargs=dict(
                                 net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                             ))
        else:
            self.model.set_env(env)

        callback = kwargs.get("callback")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=kwargs.get("progress_bar", False),
        )
        return {"timesteps": total_timesteps}

    def save(self, path: str) -> None:
        if self.model is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.model.save(path)

    def load(self, path: str) -> "BlueTeamAgent":
        if HAS_SB3:
            self.model = PPO.load(path)
        return self

    def clone(self) -> "BlueTeamAgent":
        """Create a snapshot for the opponent pool."""
        import tempfile
        agent = BlueTeamAgent(config=self.config)
        agent.elo_rating = self.elo_rating
        agent.games_played = self.games_played
        if self.model is not None:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                self.model.save(f.name)
                agent.model = PPO.load(f.name)
                os.unlink(f.name)
        return agent
