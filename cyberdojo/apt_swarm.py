"""
CyberDojo — Multi-Agent APT Swarm

Three specialized Red Team LLM agents that coordinate attacks through
a shared "Hacker Chatroom":

  🔍 SCOUT   — Reconnaissance & discovery (scan_network, scan_vulnerability)
  💥 BREACHER — Exploitation & access (exploit, privilege_escalate, lateral_move)
  📤 EXFILTRATOR — Data theft & persistence (exfiltrate_data, install_backdoor, cover_tracks)

Each agent sees the same network state but has a different specialization.
They communicate via a shared message log, visible on the dashboard
as the "Hacker Chatroom" panel.
"""

import os
import numpy as np
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from cyberdojo.agents import BaseAgent
from cyberdojo.environment import RedAction

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


# ─────────────────────────────────────────────────────────────
# Structured Output Schemas
# ─────────────────────────────────────────────────────────────

if HAS_LANGCHAIN:
    class SwarmAgentPlan(BaseModel):
        chat_message: str = Field(description="A short message to the hacker chatroom (1-2 sentences) explaining what you're doing and why. Address your teammates by role.")
        action_name: str = Field(description="The exact action name to take. MUST be one of the permitted actions for your role.")
        target_node_index: int = Field(description="The integer index (0-N) of the target node.")


# ─────────────────────────────────────────────────────────────
# Agent Role Definitions
# ─────────────────────────────────────────────────────────────

SCOUT_PROMPT = """You are SCOUT 🔍 — the reconnaissance specialist in a 3-agent APT (Advanced Persistent Threat) team.

YOUR ROLE: Discover the network topology and find vulnerabilities. You are the team's eyes.
YOUR ALLOWED ACTIONS (ONLY these): scan_network, scan_vulnerability, wait
YOUR TEAMMATES: BREACHER 💥 (exploitation) and EXFILTRATOR 📤 (data theft)

STRATEGY:
- In early turns, use scan_network to discover new nodes
- Once nodes are discovered, use scan_vulnerability to find entry points
- Report your findings to the team via the chat message
- Prioritize scanning nodes that haven't been scanned yet
- Call out high-value targets (critical nodes, databases, servers with data)

Current Network State:
{observation}

Recent Hacker Chatroom (last few messages):
{chatroom}

Number of nodes: {n_nodes}. Pick target index 0 to {n_nodes}-1."""

BREACHER_PROMPT = """You are BREACHER 💥 — the exploitation specialist in a 3-agent APT team.

YOUR ROLE: Exploit vulnerabilities, escalate privileges, and move laterally. You are the team's weapon.
YOUR ALLOWED ACTIONS (ONLY these): exploit, privilege_escalate, lateral_move, phish_user, ddos_service, wait
YOUR TEAMMATES: SCOUT 🔍 (recon) and EXFILTRATOR 📤 (data theft)

STRATEGY:
- Wait for SCOUT to identify vulnerable nodes before attacking
- Exploit nodes that SCOUT has flagged as having vulnerabilities
- After exploiting, escalate privileges to root if possible
- Use lateral_move to reach nodes deeper in the network
- Use phish_user on workstations/desktops for initial access
- Report successful compromises to teammates

Current Network State:
{observation}

Recent Hacker Chatroom (last few messages):
{chatroom}

Number of nodes: {n_nodes}. Pick target index 0 to {n_nodes}-1."""

EXFILTRATOR_PROMPT = """You are EXFILTRATOR 📤 — the data theft and persistence specialist in a 3-agent APT team.

YOUR ROLE: Steal data from compromised nodes and maintain persistent access. You are the team's payoff.
YOUR ALLOWED ACTIONS (ONLY these): exfiltrate_data, install_backdoor, cover_tracks, deploy_ransomware, wait
YOUR TEAMMATES: SCOUT 🔍 (recon) and BREACHER 💥 (exploitation)

STRATEGY:
- Wait for BREACHER to compromise nodes before trying to exfiltrate
- Prioritize exfiltrating from critical nodes and databases (high data_value)
- Install backdoors on compromised nodes for persistent access
- Use cover_tracks after exfiltration to avoid detection
- Only deploy ransomware as a final devastating move
- Report stolen data amounts to teammates

Current Network State:
{observation}

Recent Hacker Chatroom (last few messages):
{chatroom}

Number of nodes: {n_nodes}. Pick target index 0 to {n_nodes}-1."""

# Map role → allowed actions
ROLE_ACTIONS = {
    "scout": ["scan_network", "scan_vulnerability", "wait"],
    "breacher": ["exploit", "privilege_escalate", "lateral_move", "phish_user", "ddos_service", "wait"],
    "exfiltrator": ["exfiltrate_data", "install_backdoor", "cover_tracks", "deploy_ransomware", "wait"],
}


# ─────────────────────────────────────────────────────────────
# Hacker Chatroom
# ─────────────────────────────────────────────────────────────

class HackerChatroom:
    """Shared communication channel between APT agents."""
    
    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
    
    def post(self, role: str, icon: str, message: str):
        """Post a message to the chatroom."""
        self.messages.append({"role": role, "icon": icon, "text": message})
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent(self, n: int = 6) -> str:
        """Get recent messages as a formatted string for the LLM prompt."""
        if not self.messages:
            return "(No messages yet — you are the first to act this round)"
        
        recent = self.messages[-n:]
        return "\n".join(
            f"[{m['icon']} {m['role']}]: {m['text']}" for m in recent
        )
    
    def get_all_formatted(self) -> List[Dict[str, str]]:
        """Get all messages for dashboard display."""
        return list(self.messages)


# ─────────────────────────────────────────────────────────────
# APT Swarm Agent
# ─────────────────────────────────────────────────────────────

class APTSwarmAgent(BaseAgent):
    """
    Multi-Agent APT Swarm: Three coordinated LLM attackers.
    
    Each turn, all three agents see the network state and the shared
    chatroom, then the active agent (rotating) chooses an action.
    """

    def __init__(self, n_nodes: int, model_name: str = "gpt-4o"):
        super().__init__(name="APT-Swarm", team="red")
        self.n_nodes = n_nodes
        self.model_name = model_name
        self.chatroom = HackerChatroom()
        self.turn_counter = 0
        
        # Agent rotation: Scout → Breacher → Exfiltrator → Scout → ...
        self.roles = ["scout", "breacher", "exfiltrator"]
        self.icons = {"scout": "🔍", "breacher": "💥", "exfiltrator": "📤"}
        self.prompts_text = {
            "scout": SCOUT_PROMPT,
            "breacher": BREACHER_PROMPT,
            "exfiltrator": EXFILTRATOR_PROMPT,
        }
        
        if not HAS_LANGCHAIN:
            logger.error("LangChain not installed. APT Swarm cannot run.")
            self.llm = None
            return
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. APT Swarm will fail.")
            api_key = "dummy-key"
        
        self.llm = ChatOpenAI(model=model_name, api_key=api_key).with_structured_output(SwarmAgentPlan)
        
        # Build prompts for each role
        self.prompts = {}
        for role, prompt_text in self.prompts_text.items():
            self.prompts[role] = ChatPromptTemplate.from_messages([
                ("system", prompt_text),
                ("user", "Choose your next action. Coordinate with your team via the chat message.")
            ])

    def _get_active_role(self) -> str:
        """Determine which agent acts this turn (rotating)."""
        return self.roles[self.turn_counter % len(self.roles)]

    def _format_observation(self, obs: np.ndarray) -> str:
        """Format the observation for the Red Team perspective."""
        from cyberdojo.llm_agents import format_red_observation
        names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
        return format_red_observation(obs, names)

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.llm is None:
            return np.array([0, 0])
        
        role = self._get_active_role()
        icon = self.icons[role]
        prompt = self.prompts[role]
        
        try:
            obs_text = self._format_observation(observation)
            chatroom_text = self.chatroom.get_recent()
            
            # Invoke the LLM for this agent
            chain = prompt | self.llm
            result: SwarmAgentPlan = chain.invoke({
                "observation": obs_text,
                "n_nodes": self.n_nodes,
                "chatroom": chatroom_text,
            })
            
            # Post to the hacker chatroom
            self.chatroom.post(role.upper(), icon, result.chat_message)
            
            # Print to terminal
            print(f"\n[{icon} APT-{role.upper()}] {result.chat_message}")
            print(f"  └─ Action: {result.action_name} → Node {result.target_node_index}")
            
            # Broadcast to dashboard
            try:
                from dashboard.server import socketio as sio
                sio.emit("hacker_chat", {
                    "role": role,
                    "icon": icon,
                    "text": result.chat_message,
                    "action": result.action_name,
                    "target": result.target_node_index,
                })
            except ImportError:
                pass
            
            # Map action to index, constraining to role's allowed actions
            action_idx = RedAction.WAIT
            allowed = ROLE_ACTIONS.get(role, [])
            
            # First try exact match
            if result.action_name in RedAction.NAMES:
                idx = RedAction.NAMES.index(result.action_name)
                # Verify it's in this role's allowed actions
                if result.action_name in allowed:
                    action_idx = idx
                else:
                    # Fall back to the first allowed action
                    action_idx = RedAction.NAMES.index(allowed[0])
                    logger.warning(f"APT-{role}: Tried {result.action_name}, not allowed. Falling back to {allowed[0]}")
            
            target = max(0, min(self.n_nodes - 1, result.target_node_index))
            
            # Trigger exploit generator if exploiting
            if action_idx == RedAction.EXPLOIT:
                try:
                    from cyberdojo.llm_agents import _run_exploit_hook
                    raw_llm = ChatOpenAI(model=self.model_name, api_key=os.environ.get("OPENAI_API_KEY", ""))
                    names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
                    node_name = names[target] if target < len(names) else f"Node {target}"
                    _run_exploit_hook(raw_llm, node_name, names, target, self.n_nodes)
                except Exception as e:
                    logger.warning(f"Exploit hook failed in swarm: {e}")
            
            self.turn_counter += 1
            return np.array([action_idx, target])
            
        except Exception as e:
            logger.warning(f"APT-{role} failed: {e}. Falling back to scan.")
            self.chatroom.post(role.upper(), icon, f"Error — falling back to network scan. ({e})")
            self.turn_counter += 1
            return np.array([RedAction.SCAN_NETWORK, 0])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "APTSwarmAgent does not use RL training"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "APTSwarmAgent":
        return self

    def clone(self) -> "APTSwarmAgent":
        return APTSwarmAgent(self.n_nodes, self.model_name)
