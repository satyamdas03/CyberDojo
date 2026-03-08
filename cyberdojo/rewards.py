"""
CyberDojo Reward Shaping

Configurable reward functions for Red and Blue agents with
curriculum-based difficulty scaling and intrinsic curiosity bonuses.
"""

from typing import Dict, Optional
from cyberdojo.config import RewardConfig


class RewardCalculator:
    """Calculates rewards for Red and Blue agents based on game events."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.current_episode = 0
        self._scale = 1.0

    def set_episode(self, episode: int) -> None:
        """Update current episode for curriculum scaling."""
        self.current_episode = episode
        if self.config.difficulty_scaling and episode > self.config.scaling_start_episode:
            progress = (episode - self.config.scaling_start_episode) / 500.0
            self._scale = 1.0 + progress * (self.config.scaling_factor - 1.0)
            self._scale = min(self._scale, self.config.scaling_factor)
        else:
            self._scale = 1.0

    # ─────────────────────────────────────────────────────────
    # Red Team Rewards
    # ─────────────────────────────────────────────────────────

    def red_reward(self, events: Dict[str, float]) -> float:
        """
        Calculate total Red Team reward from game events.
        
        Possible events:
        - 'compromised_node': number of newly compromised nodes
        - 'got_root': number of root escalations
        - 'exfiltrated_data': value of exfiltrated data
        - 'installed_backdoor': number of backdoors installed
        - 'lateral_move': number of successful lateral moves
        - 'stayed_hidden': 1.0 if not detected this step
        - 'got_detected': number of detections
        - 'got_blocked': number of blocked actions
        - 'hit_critical': 1.0 if critical target compromised
        """
        reward = 0.0

        reward += events.get('compromised_node', 0) * self.config.red_compromise_node
        reward += events.get('got_root', 0) * self.config.red_root_access
        reward += events.get('exfiltrated_data', 0) * self.config.red_data_exfiltration
        reward += events.get('installed_backdoor', 0) * self.config.red_persistence
        reward += events.get('lateral_move', 0) * self.config.red_lateral_move
        reward += events.get('stayed_hidden', 0) * self.config.red_stealth_bonus
        reward += events.get('got_detected', 0) * self.config.red_detected_penalty
        reward += events.get('got_blocked', 0) * self.config.red_blocked_penalty
        reward += events.get('hit_critical', 0) * self.config.red_critical_target

        return reward * self._scale

    def red_exploration_bonus(self, n_discovered: int, total_nodes: int) -> float:
        """Intrinsic curiosity bonus for discovering new nodes."""
        coverage = n_discovered / max(1, total_nodes)
        return coverage * 2.0  # small bonus for network coverage

    # ─────────────────────────────────────────────────────────
    # Blue Team Rewards
    # ─────────────────────────────────────────────────────────

    def blue_reward(self, events: Dict[str, float]) -> float:
        """
        Calculate total Blue Team reward from game events.
        
        Possible events:
        - 'detected_attacker': number of successful detections
        - 'contained_threat': number of threats contained (isolated)
        - 'patched_vuln': number of vulnerabilities patched
        - 'false_positive': number of false positive isolations
        - 'caused_downtime': amount of unnecessary downtime
        - 'honeypot_triggered': 1.0 if honeypot caught attacker
        - 'full_containment': 1.0 if all threats fully contained
        - 'data_breach': value of breached data
        """
        reward = 0.0

        reward += events.get('detected_attacker', 0) * self.config.blue_detect_attacker
        reward += events.get('contained_threat', 0) * self.config.blue_contain_threat
        reward += events.get('patched_vuln', 0) * self.config.blue_patch_vuln
        reward += events.get('false_positive', 0) * self.config.blue_false_positive_penalty
        reward += events.get('caused_downtime', 0) * self.config.blue_downtime_penalty
        reward += events.get('honeypot_triggered', 0) * self.config.blue_honeypot_catch
        reward += events.get('full_containment', 0) * self.config.blue_full_containment_bonus
        reward += events.get('data_breach', 0) * self.config.blue_data_breach_penalty

        return reward * self._scale

    def blue_vigilance_bonus(self, alert_response_rate: float) -> float:
        """Bonus for maintaining high alert response rate."""
        return alert_response_rate * 1.5

    # ─────────────────────────────────────────────────────────
    # End-of-Episode Bonuses
    # ─────────────────────────────────────────────────────────

    def red_episode_bonus(self, compromised_ratio: float, data_stolen_ratio: float,
                          was_detected: bool) -> float:
        """
        End-of-episode bonus for Red based on overall performance.
        """
        bonus = 0.0
        bonus += compromised_ratio * 20.0  # % of network owned
        bonus += data_stolen_ratio * 30.0  # % of data stolen
        if not was_detected:
            bonus += 15.0  # stealth bonus
        return bonus * self._scale

    def blue_episode_bonus(self, contained_ratio: float, uptime_ratio: float,
                           data_protected_ratio: float) -> float:
        """
        End-of-episode bonus for Blue based on overall performance.
        """
        bonus = 0.0
        bonus += contained_ratio * 20.0  # % of threats contained
        bonus += uptime_ratio * 15.0  # % network uptime
        bonus += data_protected_ratio * 25.0  # % data protected
        return bonus * self._scale
