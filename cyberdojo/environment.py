"""
CyberDojo Gymnasium Environment

Multi-agent RL environment where a Red Team (attacker) and Blue Team (defender)
compete in a simulated network. Implements the Gymnasium Env interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import random
import logging

from cyberdojo.network import (
    Network, Node, AccessLevel, NodeStatus, ServiceType, VulnSeverity
)
from cyberdojo.config import CyberDojoConfig, NetworkConfig
from cyberdojo.rewards import RewardCalculator
from cyberdojo.mitre import get_red_mitre, get_blue_mitre

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Action Definitions
# ─────────────────────────────────────────────────────────────

class RedAction:
    """Red Team action types."""
    SCAN_NETWORK = 0
    SCAN_VULNERABILITY = 1
    EXPLOIT = 2
    PRIVILEGE_ESCALATE = 3
    LATERAL_MOVE = 4
    INSTALL_BACKDOOR = 5
    EXFILTRATE_DATA = 6
    COVER_TRACKS = 7
    DEPLOY_RANSOMWARE = 8
    PHISH_USER = 9
    DDOS_SERVICE = 10
    WAIT = 11
    NUM_ACTIONS = 12

    NAMES = [
        "scan_network", "scan_vulnerability", "exploit",
        "privilege_escalate", "lateral_move", "install_backdoor",
        "exfiltrate_data", "cover_tracks", "deploy_ransomware",
        "phish_user", "ddos_service", "wait",
    ]


class BlueAction:
    """Blue Team action types."""
    MONITOR_TRAFFIC = 0
    ANALYZE_ALERT = 1
    ISOLATE_NODE = 2
    PATCH_VULNERABILITY = 3
    DEPLOY_HONEYPOT = 4
    RESTORE_BACKUP = 5
    UPDATE_FIREWALL = 6
    FORENSIC_ANALYSIS = 7
    DEPLOY_IDS_RULE = 8
    WAIT = 9
    NUM_ACTIONS = 10

    NAMES = [
        "monitor_traffic", "analyze_alert", "isolate_node",
        "patch_vulnerability", "deploy_honeypot", "restore_backup",
        "update_firewall", "forensic_analysis", "deploy_ids_rule", "wait",
    ]


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

class CyberDojoEnv(gym.Env):
    """
    CyberDojo: Adversarial AI War Games Environment
    
    A multi-agent environment where:
    - Red Team tries to compromise the network, steal data, maintain persistence
    - Blue Team tries to detect, contain, and remediate threats
    
    The environment supports two modes:
    1. 'red' mode: Train the Red agent (Blue uses fixed/scripted policy)
    2. 'blue' mode: Train the Blue agent (Red uses fixed/scripted policy)
    
    For co-evolutionary training, the trainer alternates between modes.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[CyberDojoConfig] = None,
        mode: str = "red",  # 'red' or 'blue' — which agent is being trained
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or CyberDojoConfig()
        self.mode = mode
        self.render_mode = render_mode
        self._seed = seed

        # Build network
        self.network = Network(config=self.config.network, seed=seed)
        self.n_nodes = len(self.network.nodes)
        self.node_ids = sorted(self.network.nodes.keys())

        # Reward calculator
        self.reward_calc = RewardCalculator(self.config.rewards)

        # Observation and action spaces
        obs_size = self.n_nodes * 8  # 8 features per node
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        if mode == "red":
            # Red action: [action_type, target_node_index]
            self.action_space = spaces.MultiDiscrete(
                [RedAction.NUM_ACTIONS, self.n_nodes]
            )
        else:
            # Blue action: [action_type, target_node_index]
            self.action_space = spaces.MultiDiscrete(
                [BlueAction.NUM_ACTIONS, self.n_nodes]
            )

        # Episode state
        self.current_step = 0
        self.max_steps = self.config.training.steps_per_episode
        self.episode_count = 0

        # Event log (for dashboard)
        self.event_log: List[Dict] = []
        self.step_data: Dict = {}

        # Opponent policy (set externally by trainer)
        self._opponent_policy = None

        # RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Track episode stats
        self._episode_stats = {
            "red_detections": 0,
            "red_compromises": 0,
            "red_data_stolen": 0.0,
            "blue_patches": 0,
            "blue_isolations": 0,
            "blue_false_positives": 0,
        }

    def set_opponent_policy(self, policy_fn) -> None:
        """
        Set the opponent's policy function.
        policy_fn: callable(observation) -> action
        """
        self._opponent_policy = policy_fn

    def _get_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        """Get the opponent's action."""
        if self._opponent_policy is not None:
            return self._opponent_policy(obs)
        # Default: random action
        if self.mode == "red":
            return np.array([
                self.rng.randint(0, BlueAction.NUM_ACTIONS - 1),
                self.rng.randint(0, self.n_nodes - 1),
            ])
        else:
            return np.array([
                self.rng.randint(0, RedAction.NUM_ACTIONS - 1),
                self.rng.randint(0, self.n_nodes - 1),
            ])

    # ─────────────────────────────────────────────────────────
    # Gymnasium Interface
    # ─────────────────────────────────────────────────────────

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

        self.network.reset()
        self.current_step = 0
        self.event_log = []
        self.episode_count += 1
        self.reward_calc.set_episode(self.episode_count)

        self._episode_stats = {
            "red_detections": 0,
            "red_compromises": 0,
            "red_data_stolen": 0.0,
            "blue_patches": 0,
            "blue_isolations": 0,
            "blue_false_positives": 0,
        }

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        The training agent takes the given action, and the opponent
        takes an action determined by its policy.
        """
        self.current_step += 1

        # Get opponent's observation and action
        opponent_obs = self._get_opponent_observation()
        opponent_action = self._get_opponent_action(opponent_obs)

        # Resolve actions simultaneously
        if self.mode == "red":
            red_action = action
            blue_action = opponent_action
        else:
            red_action = opponent_action
            blue_action = action

        # Execute actions and collect events
        red_events = self._execute_red_action(int(red_action[0]), int(red_action[1]))
        blue_events = self._execute_blue_action(int(blue_action[0]), int(blue_action[1]))

        # Cross-effects: blue actions may affect red outcomes and vice versa
        self._resolve_interactions(red_events, blue_events)

        # Calculate reward for the training agent
        if self.mode == "red":
            reward = self.reward_calc.red_reward(red_events)
            reward += self.reward_calc.red_exploration_bonus(
                sum(1 for n in self.network.nodes.values() if n.is_discovered_by_red),
                self.n_nodes,
            )
        else:
            reward = self.reward_calc.blue_reward(blue_events)

        # Log event for dashboard
        self._log_event(red_action, blue_action, red_events, blue_events)

        # Check termination
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        # Add end-of-episode bonuses
        if terminated or truncated:
            reward += self._episode_end_bonus()

        obs = self._get_observation()
        info = self._get_info()
        info["red_events"] = red_events
        info["blue_events"] = blue_events

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get observation for the training agent."""
        if self.mode == "red":
            return self.network.get_red_observation()
        return self.network.get_blue_observation()

    def _get_opponent_observation(self) -> np.ndarray:
        """Get observation for the opponent."""
        if self.mode == "red":
            return self.network.get_blue_observation()
        return self.network.get_red_observation()

    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            "step": self.current_step,
            "episode": self.episode_count,
            "compromised_nodes": len(self.network.get_compromised_nodes()),
            "total_nodes": self.n_nodes,
            "compromised_ratio": len(self.network.get_compromised_nodes()) / self.n_nodes,
            "data_stolen_ratio": self.network.compromised_data_value / max(1, self.network.total_data_value),
            "stats": dict(self._episode_stats),
        }

    # ─────────────────────────────────────────────────────────
    # Red Team Action Execution
    # ─────────────────────────────────────────────────────────

    def _execute_red_action(self, action_type: int, target_idx: int) -> Dict[str, float]:
        """Execute a Red Team action and return events."""
        events: Dict[str, float] = {}
        target_id = self.node_ids[min(target_idx, len(self.node_ids) - 1)]
        target = self.network.get_node(target_id)

        if target is None or target.is_isolated:
            events['got_blocked'] = 1.0
            return events

        if action_type == RedAction.SCAN_NETWORK:
            events.update(self._red_scan_network(target))
        elif action_type == RedAction.SCAN_VULNERABILITY:
            events.update(self._red_scan_vuln(target))
        elif action_type == RedAction.EXPLOIT:
            events.update(self._red_exploit(target))
        elif action_type == RedAction.PRIVILEGE_ESCALATE:
            events.update(self._red_privesc(target))
        elif action_type == RedAction.LATERAL_MOVE:
            events.update(self._red_lateral_move(target))
        elif action_type == RedAction.INSTALL_BACKDOOR:
            events.update(self._red_install_backdoor(target))
        elif action_type == RedAction.EXFILTRATE_DATA:
            events.update(self._red_exfiltrate(target))
        elif action_type == RedAction.COVER_TRACKS:
            events.update(self._red_cover_tracks(target))
        elif action_type == RedAction.DEPLOY_RANSOMWARE:
            events.update(self._red_ransomware(target))
        elif action_type == RedAction.PHISH_USER:
            events.update(self._red_phish(target))
        elif action_type == RedAction.DDOS_SERVICE:
            events.update(self._red_ddos(target))
        elif action_type == RedAction.WAIT:
            events['stayed_hidden'] = 1.0

        return events

    def _red_scan_network(self, target: Node) -> Dict[str, float]:
        """Scan network from a foothold to discover adjacent nodes."""
        events = {}
        if not target.is_discovered_by_red:
            events['got_blocked'] = 1.0
            return events

        adjacent = self.network.get_adjacent_nodes(target.node_id)
        discovered = 0
        for adj in adjacent:
            if not adj.is_discovered_by_red and not adj.is_isolated:
                adj.is_discovered_by_red = True
                discovered += 1

        # Scanning increases suspicion (but basic scans are quiet)
        target.scan_count += 1
        detection_chance = 0.03 + target.scan_count * 0.02
        subnet = self.network.subnets[target.subnet_id]
        detection_chance += subnet.traffic_monitor_level * 0.15

        if self.rng.random() < detection_chance:
            target.alert_level = min(1.0, target.alert_level + 0.2)
            events['got_detected'] = 1.0
            self._episode_stats['red_detections'] += 1

        return events

    def _red_scan_vuln(self, target: Node) -> Dict[str, float]:
        """Scan a discovered node for vulnerabilities."""
        events = {}
        if not target.is_discovered_by_red:
            events['got_blocked'] = 1.0
            return events

        target.scan_count += 1
        # Vulnerability scanning is noisier than basic scans
        detection_chance = 0.08 + target.scan_count * 0.04
        subnet = self.network.subnets[target.subnet_id]
        detection_chance += subnet.traffic_monitor_level * 0.2

        if self.rng.random() < detection_chance:
            target.alert_level = min(1.0, target.alert_level + 0.3)
            events['got_detected'] = 1.0
            self._episode_stats['red_detections'] += 1

        return events

    def _red_exploit(self, target: Node) -> Dict[str, float]:
        """Attempt to exploit a vulnerability on a target node."""
        events = {}
        if not target.is_discovered_by_red:
            events['got_blocked'] = 1.0
            return events

        if target.is_compromised and target.access_level == AccessLevel.ROOT:
            return events  # already fully compromised

        # Find exploitable vulnerabilities
        exploitable = target.unpatched_vulns
        if not exploitable:
            events['got_blocked'] = 1.0
            return events

        # Try the best vulnerability
        best_vuln = max(exploitable, key=lambda v: v.exploit_probability)
        if self.rng.random() < best_vuln.exploit_probability:
            # Successful exploit!
            if target.status != NodeStatus.COMPROMISED:
                target.status = NodeStatus.COMPROMISED
                events['compromised_node'] = 1.0
                self._episode_stats['red_compromises'] += 1

            if best_vuln.grants_root:
                target.access_level = AccessLevel.ROOT
                events['got_root'] = 1.0
            elif target.access_level == AccessLevel.NONE:
                target.access_level = AccessLevel.USER

            # Check if critical target
            if target.is_critical:
                events['hit_critical'] = 1.0

            # Honeypot check
            if target.is_honeypot:
                target.alert_level = 1.0
                events['got_detected'] = 1.0
                self._episode_stats['red_detections'] += 1
        else:
            events['got_blocked'] = 1.0

        # Exploitation generates some noise
        target.scan_count += 1
        detection_chance = 0.15 + target.scan_count * 0.05
        subnet = self.network.subnets[target.subnet_id]
        detection_chance += subnet.traffic_monitor_level * 0.25
        if self.rng.random() < min(0.7, detection_chance):
            target.alert_level = min(1.0, target.alert_level + 0.3)
            events['got_detected'] = events.get('got_detected', 0) + 1.0
            self._episode_stats['red_detections'] += 1

        return events

    def _red_privesc(self, target: Node) -> Dict[str, float]:
        """Attempt privilege escalation on a compromised node."""
        events = {}
        if not target.is_compromised or target.access_level == AccessLevel.ROOT:
            events['got_blocked'] = 1.0
            return events

        # Privesc difficulty depends on OS
        success_chance = 0.6 if target.os_type == "linux" else 0.7
        if self.rng.random() < success_chance:
            target.access_level = AccessLevel.ROOT
            events['got_root'] = 1.0
        else:
            events['got_blocked'] = 1.0

        # Moderate noise
        target.scan_count += 1
        if self.rng.random() < 0.2:
            target.alert_level = min(1.0, target.alert_level + 0.2)

        return events

    def _red_lateral_move(self, target: Node) -> Dict[str, float]:
        """Move laterally from a compromised node to an adjacent one."""
        events = {}
        if not target.is_compromised:
            events['got_blocked'] = 1.0
            return events

        # Find adjacent undiscovered/uncompromised nodes
        adjacent = self.network.get_adjacent_nodes(target.node_id)
        targets = [n for n in adjacent
                   if not n.is_compromised and not n.is_isolated]

        if not targets:
            events['got_blocked'] = 1.0
            return events

        # Pick a target and try to move
        lateral_target = self.rng.choice(targets)

        # Check firewall
        if target.subnet_id != lateral_target.subnet_id:
            fw_key = (target.subnet_id, lateral_target.subnet_id)
            fw_rule = self.network.firewall_rules.get(fw_key)
            if fw_rule and fw_rule.is_active:
                # Need an allowed port
                target_ports = {s.port for s in lateral_target.services}
                allowed = target_ports & fw_rule.allowed_ports
                if not allowed:
                    events['got_blocked'] = 1.0
                    return events

        # Lateral movement using compromised credentials
        success_chance = 0.7 if target.access_level == AccessLevel.ROOT else 0.5
        if self.rng.random() < success_chance:
            lateral_target.is_discovered_by_red = True
            lateral_target.status = NodeStatus.COMPROMISED
            lateral_target.access_level = AccessLevel.USER
            events['lateral_move'] = 1.0
            events['compromised_node'] = 1.0
            self._episode_stats['red_compromises'] += 1

            if lateral_target.is_critical:
                events['hit_critical'] = 1.0
            if lateral_target.is_honeypot:
                lateral_target.alert_level = 1.0
                events['got_detected'] = 1.0
        else:
            events['got_blocked'] = 1.0

        # Lateral movement is noisy
        target.scan_count += 1
        lateral_target.scan_count += 1
        if self.rng.random() < 0.25:
            target.alert_level = min(1.0, target.alert_level + 0.3)

        return events

    def _red_install_backdoor(self, target: Node) -> Dict[str, float]:
        """Install a persistence backdoor on a compromised node."""
        events = {}
        if not target.is_compromised or target.has_backdoor:
            events['got_blocked'] = 1.0
            return events

        if target.access_level.value >= AccessLevel.USER.value:
            target.has_backdoor = True
            events['installed_backdoor'] = 1.0
            # Quiet action
            if self.rng.random() < 0.1:
                target.alert_level = min(1.0, target.alert_level + 0.15)
        else:
            events['got_blocked'] = 1.0

        return events

    def _red_exfiltrate(self, target: Node) -> Dict[str, float]:
        """Exfiltrate data from a compromised node."""
        events = {}
        if not target.is_compromised:
            events['got_blocked'] = 1.0
            return events

        data_value = target.data_value
        if target.access_level == AccessLevel.ROOT:
            data_value *= 1.5  # root gets more data

        events['exfiltrated_data'] = data_value
        self._episode_stats['red_data_stolen'] += data_value

        # Data exfiltration is moderately noisy (network traffic)
        target.scan_count += 1
        subnet = self.network.subnets[target.subnet_id]
        detection_chance = 0.2 + subnet.traffic_monitor_level * 0.4
        if self.rng.random() < detection_chance:
            target.alert_level = min(1.0, target.alert_level + 0.4)
            events['got_detected'] = 1.0
            self._episode_stats['red_detections'] += 1

        return events

    def _red_cover_tracks(self, target: Node) -> Dict[str, float]:
        """Clear logs and reduce detection indicators."""
        events = {}
        if not target.is_compromised:
            events['got_blocked'] = 1.0
            return events

        # Reduce alert level and scan count
        target.alert_level = max(0.0, target.alert_level - 0.4)
        target.scan_count = max(0, target.scan_count - 2)
        events['stayed_hidden'] = 1.0

        return events

    def _red_ransomware(self, target: Node) -> Dict[str, float]:
        """Deploy ransomware on a compromised node."""
        events = {}
        if not target.is_compromised or target.access_level != AccessLevel.ROOT:
            events['got_blocked'] = 1.0
            return events

        if not target.is_encrypted:
            target.is_encrypted = True
            events['exfiltrated_data'] = target.data_value * 0.5  # partial value

            # Very noisy!
            target.alert_level = 1.0
            events['got_detected'] = 1.0
            self._episode_stats['red_detections'] += 1

        return events

    def _red_phish(self, target: Node) -> Dict[str, float]:
        """Attempt social engineering attack on a node's user."""
        events = {}
        # Can phish any discovered node
        if not target.is_discovered_by_red or target.is_compromised:
            events['got_blocked'] = 1.0
            return events

        # Phishing success depends on OS (Windows users more susceptible)
        success_chance = 0.3 if target.os_type == "windows" else 0.15
        if self.rng.random() < success_chance:
            target.status = NodeStatus.COMPROMISED
            target.access_level = AccessLevel.USER
            events['compromised_node'] = 1.0
            self._episode_stats['red_compromises'] += 1
            if target.is_critical:
                events['hit_critical'] = 1.0
        else:
            events['got_blocked'] = 1.0

        # Phishing is quiet unless it fails obviously
        if self.rng.random() < 0.1:
            target.alert_level = min(1.0, target.alert_level + 0.2)

        return events

    def _red_ddos(self, target: Node) -> Dict[str, float]:
        """Launch a DDoS attack against a node's services."""
        events = {}
        if not target.is_discovered_by_red:
            events['got_blocked'] = 1.0
            return events

        # Disable services
        for service in target.services:
            if self.rng.random() < 0.7:
                service.is_running = False

        # Very noisy
        target.alert_level = 1.0
        events['got_detected'] = 1.0
        self._episode_stats['red_detections'] += 1

        return events

    # ─────────────────────────────────────────────────────────
    # Blue Team Action Execution
    # ─────────────────────────────────────────────────────────

    def _execute_blue_action(self, action_type: int, target_idx: int) -> Dict[str, float]:
        """Execute a Blue Team action and return events."""
        events: Dict[str, float] = {}
        target_id = self.node_ids[min(target_idx, len(self.node_ids) - 1)]
        target = self.network.get_node(target_id)

        if target is None:
            return events

        if action_type == BlueAction.MONITOR_TRAFFIC:
            events.update(self._blue_monitor(target))
        elif action_type == BlueAction.ANALYZE_ALERT:
            events.update(self._blue_analyze(target))
        elif action_type == BlueAction.ISOLATE_NODE:
            events.update(self._blue_isolate(target))
        elif action_type == BlueAction.PATCH_VULNERABILITY:
            events.update(self._blue_patch(target))
        elif action_type == BlueAction.DEPLOY_HONEYPOT:
            events.update(self._blue_honeypot(target))
        elif action_type == BlueAction.RESTORE_BACKUP:
            events.update(self._blue_restore(target))
        elif action_type == BlueAction.UPDATE_FIREWALL:
            events.update(self._blue_firewall(target))
        elif action_type == BlueAction.FORENSIC_ANALYSIS:
            events.update(self._blue_forensics(target))
        elif action_type == BlueAction.DEPLOY_IDS_RULE:
            events.update(self._blue_ids_rule(target))
        elif action_type == BlueAction.WAIT:
            pass  # observe only

        return events

    def _blue_monitor(self, target: Node) -> Dict[str, float]:
        """Increase traffic monitoring on target's subnet."""
        events = {}
        subnet = self.network.subnets[target.subnet_id]
        subnet.traffic_monitor_level = min(1.0, subnet.traffic_monitor_level + 0.15)

        # Monitoring may reveal suspicious activity
        if target.is_compromised and self.rng.random() < subnet.traffic_monitor_level:
            target.alert_level = min(1.0, target.alert_level + 0.3)
            events['detected_attacker'] = 1.0

        return events

    def _blue_analyze(self, target: Node) -> Dict[str, float]:
        """Investigate alerts on a node."""
        events = {}
        if target.alert_level < 0.2:
            return events  # nothing to analyze

        # Analysis accuracy depends on alert level
        if target.is_compromised:
            # True positive
            detection_chance = 0.5 + target.alert_level * 0.4
            if self.rng.random() < detection_chance:
                events['detected_attacker'] = 1.0
                target.status = NodeStatus.DETECTED
                # Reveal known vulnerabilities
                for vuln in target.unpatched_vulns:
                    vuln.is_known_to_defender = True
        elif target.alert_level > 0.3:
            # Potential false positive scenario — don't penalize analysis itself
            pass

        return events

    def _blue_isolate(self, target: Node) -> Dict[str, float]:
        """Isolate a suspicious node from the network."""
        events = {}
        if target.is_isolated:
            return events

        # Blue must have some evidence before isolating (prevents blind spam)
        if target.alert_level < 0.3 and not target.is_compromised:
            events['false_positive'] = 1.0
            events['caused_downtime'] = 1.5  # heavy penalty for blind isolation
            self._episode_stats['blue_false_positives'] += 1
            return events

        target.status = NodeStatus.ISOLATED

        if target.is_compromised or target.alert_level > 0.5:
            # Good isolation — contained a threat
            events['contained_threat'] = 1.0
            self._episode_stats['blue_isolations'] += 1
        else:
            # False positive — isolated a clean node
            events['false_positive'] = 1.0
            events['caused_downtime'] = 1.0
            self._episode_stats['blue_false_positives'] += 1

        return events

    def _blue_patch(self, target: Node) -> Dict[str, float]:
        """Patch a known vulnerability on a node."""
        events = {}
        patched = 0
        for service in target.services:
            for vuln in service.vulnerabilities:
                if not vuln.is_patched and vuln.is_known_to_defender:
                    vuln.is_patched = True
                    patched += 1

        if patched > 0:
            events['patched_vuln'] = float(patched)
            self._episode_stats['blue_patches'] += patched
        else:
            # Try to discover vulns via scanning
            for service in target.services:
                for vuln in service.vulnerabilities:
                    if not vuln.is_known_to_defender and self.rng.random() < 0.3:
                        vuln.is_known_to_defender = True

        return events

    def _blue_honeypot(self, target: Node) -> Dict[str, float]:
        """Deploy a honeypot disguised as the target node."""
        events = {}
        if not target.is_honeypot and not target.is_compromised:
            target.is_honeypot = True
            # Honeypots look attractive to attackers
            target.data_value += 2.0

        return events

    def _blue_restore(self, target: Node) -> Dict[str, float]:
        """Restore a compromised/isolated node from backup."""
        events = {}
        if target.is_compromised or target.is_isolated or target.is_encrypted:
            target.status = NodeStatus.RESTORED
            target.access_level = AccessLevel.NONE
            target.has_backdoor = False
            target.is_encrypted = False

            # Restored node goes back to clean after a brief period
            target.status = NodeStatus.CLEAN
            target.alert_level = 0.0

            events['contained_threat'] = 1.0

            # But restoration causes brief downtime
            events['caused_downtime'] = 0.5
        else:
            events['caused_downtime'] = 0.3  # unnecessary restore, small downtime cost

        return events

    def _blue_firewall(self, target: Node) -> Dict[str, float]:
        """Update firewall rules for the target's subnet."""
        events = {}
        subnet_id = target.subnet_id

        # Tighten firewall rules involving this subnet
        for (src, dst), rule in self.network.firewall_rules.items():
            if src == subnet_id or dst == subnet_id:
                # Reduce allowed ports (more restrictive)
                if rule.allowed_ports:
                    # Keep only essential ports (80, 443)
                    rule.allowed_ports = rule.allowed_ports & {80, 443}

        # This can block attacker but also cause legitimate traffic issues
        if self.rng.random() < 0.2:
            events['caused_downtime'] = 0.3

        return events

    def _blue_forensics(self, target: Node) -> Dict[str, float]:
        """Deep forensic analysis of a node."""
        events = {}
        if target.is_compromised or target.has_backdoor:
            # Forensics is very effective at finding compromise
            if self.rng.random() < 0.8:
                events['detected_attacker'] = 1.0
                target.status = NodeStatus.DETECTED
                # Reveal backdoors
                if target.has_backdoor:
                    target.alert_level = 1.0
                # Reveal all vulnerabilities
                for vuln in target.unpatched_vulns:
                    vuln.is_known_to_defender = True
        elif target.alert_level > 0.2:
            # Clean node with some alerts — reduce false alarm
            target.alert_level = max(0.0, target.alert_level - 0.3)

        return events

    def _blue_ids_rule(self, target: Node) -> Dict[str, float]:
        """Deploy a new IDS detection rule for the target's subnet."""
        events = {}
        subnet = self.network.subnets[target.subnet_id]
        subnet.ids_sensitivity = min(1.0, subnet.ids_sensitivity + 0.2)

        # Higher sensitivity means more detection but also more false positives
        for node in self.network.get_nodes_in_subnet(target.subnet_id):
            if node.is_compromised and self.rng.random() < subnet.ids_sensitivity * 0.3:
                node.alert_level = min(1.0, node.alert_level + 0.3)
                events['detected_attacker'] = events.get('detected_attacker', 0) + 1.0

        return events

    # ─────────────────────────────────────────────────────────
    # Interaction Resolution
    # ─────────────────────────────────────────────────────────

    def _resolve_interactions(
        self, red_events: Dict[str, float], blue_events: Dict[str, float]
    ) -> None:
        """Resolve cross-effects between red and blue actions."""
        # If blue detected red this step, increase future detection probability
        if blue_events.get('detected_attacker', 0) > 0:
            for node in self.network.nodes.values():
                if node.is_compromised:
                    node.alert_level = min(1.0, node.alert_level + 0.1)

        # If blue contained threats, check if red has backdoors for re-entry
        if blue_events.get('contained_threat', 0) > 0:
            backdoor_nodes = [n for n in self.network.nodes.values() if n.has_backdoor]
            for node in backdoor_nodes:
                if not node.is_isolated:
                    # Backdoor allows re-compromise after restoration
                    if node.status == NodeStatus.CLEAN:
                        node.status = NodeStatus.COMPROMISED
                        node.access_level = AccessLevel.USER

    # ─────────────────────────────────────────────────────────
    # Termination & Scoring
    # ─────────────────────────────────────────────────────────

    def _check_terminated(self) -> bool:
        """Check if the episode should end early."""
        # Red wins: compromised the critical target with root AND exfiltrated
        critical = self.network.get_node(self.network.critical_target_id) \
            if self.network.critical_target_id else None
        if critical and critical.is_compromised and \
                critical.access_level == AccessLevel.ROOT and \
                self._episode_stats['red_data_stolen'] > critical.data_value:
            return True

        # Blue wins: all compromised nodes are isolated or restored
        compromised = self.network.get_compromised_nodes()
        if self._episode_stats['red_compromises'] > 0 and len(compromised) == 0:
            return True

        return False

    def _episode_end_bonus(self) -> float:
        """Calculate end-of-episode bonus for the training agent."""
        compromised_ratio = len(self.network.get_compromised_nodes()) / self.n_nodes
        data_stolen_ratio = (
            self._episode_stats['red_data_stolen'] /
            max(1, self.network.total_data_value)
        )
        was_detected = self._episode_stats['red_detections'] > 0

        if self.mode == "red":
            return self.reward_calc.red_episode_bonus(
                compromised_ratio, data_stolen_ratio, was_detected
            )
        else:
            contained = self._episode_stats['blue_isolations']
            total_threats = max(1, self._episode_stats['red_compromises'])
            uptime = 1.0 - (self._episode_stats['blue_false_positives'] / self.n_nodes)
            data_protected = 1.0 - data_stolen_ratio

            return self.reward_calc.blue_episode_bonus(
                contained / total_threats,
                max(0, uptime),
                data_protected,
            )

    # ─────────────────────────────────────────────────────────
    # Event Logging
    # ─────────────────────────────────────────────────────────

    def _log_event(
        self,
        red_action: np.ndarray,
        blue_action: np.ndarray,
        red_events: Dict,
        blue_events: Dict,
    ) -> None:
        """Log events for dashboard visualization."""
        red_target_id = self.node_ids[min(int(red_action[1]), len(self.node_ids) - 1)]
        blue_target_id = self.node_ids[min(int(blue_action[1]), len(self.node_ids) - 1)]
        
        red_target_name = self.network.nodes[red_target_id].name
        blue_target_name = self.network.nodes[blue_target_id].name

        event = {
            "step": self.current_step,
            "red": {
                "action": RedAction.NAMES[int(red_action[0])],
                "target": red_target_id,
                "target_name": red_target_name,
                "mitre": get_red_mitre(RedAction.NAMES[int(red_action[0])]),
                "events": dict(red_events),
            },
            "blue": {
                "action": BlueAction.NAMES[int(blue_action[0])],
                "target": blue_target_id,
                "target_name": blue_target_name,
                "mitre": get_blue_mitre(BlueAction.NAMES[int(blue_action[0])]),
                "events": dict(blue_events),
            },
            "network_state": self.network.get_topology_data(),
        }
        self.event_log.append(event)
        self.step_data = event

    # ─────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────

    def render(self) -> Optional[str]:
        """Render the current state."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Render as ANSI text for terminal."""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  CYBERDOJO — Step {self.current_step}/{self.max_steps}")
        lines.append(f"{'='*60}")

        # Network overview per subnet
        for sid, subnet in self.network.subnets.items():
            lines.append(f"\n  [{subnet.name}] (Monitor: {subnet.traffic_monitor_level:.1f})")
            for node in subnet.nodes.values():
                status_icon = {
                    NodeStatus.CLEAN: "🟢",
                    NodeStatus.COMPROMISED: "🔴",
                    NodeStatus.DETECTED: "🟡",
                    NodeStatus.ISOLATED: "⚪",
                    NodeStatus.RESTORED: "🔵",
                }.get(node.status, "❓")

                access = {
                    AccessLevel.NONE: "",
                    AccessLevel.USER: "[USER]",
                    AccessLevel.ROOT: "[ROOT]",
                }.get(node.access_level, "")

                flags = []
                if node.is_critical:
                    flags.append("⭐")
                if node.is_honeypot:
                    flags.append("🍯")
                if node.has_backdoor:
                    flags.append("🚪")
                if node.is_encrypted:
                    flags.append("🔒")

                flag_str = " ".join(flags)
                alert = f" ⚠{node.alert_level:.1f}" if node.alert_level > 0.2 else ""
                lines.append(
                    f"    {status_icon} {node.name} {access} {flag_str}{alert}"
                )

        # Latest event
        if self.step_data:
            lines.append(f"\n  Latest:")
            red = self.step_data.get("red", {})
            blue = self.step_data.get("blue", {})
            lines.append(f"    🔴 Red: {red.get('action', '?')} → {red.get('target', '?')}")
            lines.append(f"    🔵 Blue: {blue.get('action', '?')} → {blue.get('target', '?')}")

        # Stats
        stats = self._episode_stats
        lines.append(f"\n  Stats: Compromises={stats['red_compromises']} "
                      f"Detections={stats['red_detections']} "
                      f"Data Stolen={stats['red_data_stolen']:.1f}")
        lines.append(f"{'='*60}\n")

        return "\n".join(lines)
