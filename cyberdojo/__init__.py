"""
CyberDojo — Adversarial AI War Games Engine

A co-evolutionary reinforcement learning platform where Red Team (attacker)
and Blue Team (defender) AI agents compete and evolve in a simulated
network environment.
"""

__version__ = "0.1.0"
__author__ = "CyberDojo Team"

from cyberdojo.network import Network, Node, Subnet, Service, Vulnerability
from cyberdojo.environment import CyberDojoEnv
from cyberdojo.agents import RedTeamAgent, BlueTeamAgent, RandomAgent
from cyberdojo.trainer import CoEvolutionaryTrainer
