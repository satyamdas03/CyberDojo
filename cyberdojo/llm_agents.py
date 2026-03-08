"""
CyberDojo LLM Agents

Autonomous agents powered by Large Language Models (LLMs) that reason 
about the network state and choose actions dynamically.
"""

import os
import numpy as np
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from cyberdojo.agents import BaseAgent
from cyberdojo.environment import RedAction, BlueAction
from cyberdojo.remediation import get_vulnerable_config, generate_remediation_prompt, evaluate_patch
from cyberdojo.exploit_gen import get_exploit_target, generate_exploit_prompt, evaluate_exploit

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class LLMRedActionPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning on what to do next based on the network state.")
    action_name: str = Field(description="The exact name of the action to take (e.g., 'scan_network', 'exploit', 'lateral_move'). MUST be one of the permitted actions.")
    target_node_index: int = Field(description="The numerical index (0-N) of the target node to perform the action on.")


class LLMBlueActionPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning on what to do next based on the network state.")
    action_name: str = Field(description="The exact name of the defense action to take (e.g., 'monitor_traffic', 'isolate_node', 'patch_vulnerability'). MUST be one of the permitted actions.")
    target_node_index: int = Field(description="The numerical index (0-N) of the target node to perform the action on.")


def _run_remediation_hook(llm, node_name: str, node_names: list, target_idx: int, n_nodes: int):
    """
    Auto-Remediation Hook: When patch_vulnerability is selected, generate
    a vulnerable config file and ask the LLM to write the remediation code.
    The output is broadcast to the dashboard chat.
    """
    try:
        from dashboard.server import socketio as sio
    except ImportError:
        return

    # Find a vulnerability on the target node from the environment
    try:
        from cyberdojo.network import VULN_TEMPLATES
        # Pick a relevant CVE from the templates
        available_cves = []
        for svc_type, vulns in VULN_TEMPLATES.items():
            for v in vulns:
                config_data = get_vulnerable_config(v[0])
                if config_data:
                    available_cves.append((v[0], v[1], config_data))

        if not available_cves:
            return

        # Cycle through CVEs based on target index for variety
        cve_id, cve_desc, config_data = available_cves[target_idx % len(available_cves)]

        # Notify dashboard that remediation is starting
        sio.emit("chat_broadcast", {
            "sender": "system",
            "text": f"🔧 AUTO-REMEDIATION: Generating patch for {cve_id} on {node_name}...\n"
                    f"📄 Analyzing: {config_data['filename']}\n"
                    f"⚠️ Vuln: {cve_desc}"
        })

        # Ask the LLM to write the remediation
        prompt = generate_remediation_prompt(
            node_name=node_name,
            vuln_id=cve_id,
            vuln_description=cve_desc,
            service_name=config_data["service"],
            config_data=config_data,
        )

        patched_config = llm.invoke(prompt)
        patch_text = patched_config.content if hasattr(patched_config, 'content') else str(patched_config)

        # Evaluate the patch
        is_valid = evaluate_patch(config_data["config"], patch_text, cve_id)

        status = "✅ PATCH VERIFIED" if is_valid else "⚠️ PATCH NEEDS REVIEW"

        # Broadcast the patch code to the dashboard
        sio.emit("chat_broadcast", {
            "sender": "llm",
            "text": f"{status} — {cve_id} on {node_name}\n"
                    f"━━━ PATCHED {config_data['filename']} ━━━\n"
                    f"{patch_text[:800]}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{'Patch applied successfully.' if is_valid else 'Patch generated but requires manual review.'}"
        })

        logger.info(f"Remediation {'VALID' if is_valid else 'INVALID'} for {cve_id} on {node_name}")

    except Exception as e:
        logger.warning(f"Remediation hook failed: {e}")
        sio.emit("chat_broadcast", {
            "sender": "system",
            "text": f"⚠️ Remediation engine error: {e}"
        })


def _run_exploit_hook(llm, node_name: str, node_names: list, target_idx: int, n_nodes: int):
    """
    Exploit Generator Hook: When the Red LLM selects 'exploit', generate
    a realistic exploit script targeting the node's vulnerability.
    The output is printed to the terminal and broadcast to the dashboard.
    """
    try:
        from dashboard.server import socketio as sio
    except ImportError:
        sio = None

    try:
        from cyberdojo.network import VULN_TEMPLATES
        # Pick a relevant CVE for the exploit
        available_cves = []
        for svc_type, vulns in VULN_TEMPLATES.items():
            for v in vulns:
                target_data = get_exploit_target(v[0])
                if target_data:
                    available_cves.append((v[0], v[1], target_data))

        if not available_cves:
            return

        # Cycle through CVEs based on target index for variety
        cve_id, cve_desc, target_data = available_cves[target_idx % len(available_cves)]
        target_ip = f"10.0.{target_idx // 4 + 1}.{(target_idx % 254) + 10}"

        print(f"\n[💀 EXPLOIT GEN] Writing exploit for {cve_id} targeting {node_name} ({target_ip})...")

        if sio:
            sio.emit("chat_broadcast", {
                "sender": "system",
                "text": f"💀 EXPLOIT GENERATOR: Red Team is crafting an exploit for {cve_id} on {node_name}..."
            })

        # Ask the LLM to write the exploit
        prompt = generate_exploit_prompt(
            node_name=node_name,
            vuln_id=cve_id,
            vuln_description=cve_desc,
            service_name=target_data["service"],
            target_data=target_data,
            target_ip=target_ip,
        )

        exploit_result = llm.invoke(prompt)
        exploit_code = exploit_result.content if hasattr(exploit_result, 'content') else str(exploit_result)

        # Evaluate the exploit
        is_valid = evaluate_exploit(exploit_code, cve_id)

        status = "🔓 EXPLOIT SUCCESSFUL" if is_valid else "❌ EXPLOIT FAILED"

        # Print to terminal
        print(f"\n[{status}] {cve_id} → {node_name}")
        print(f"━━━ exploit_{cve_id.replace('-','_').lower()}.py ━━━")
        # Print first 600 chars to terminal
        for line in exploit_code[:600].split('\n'):
            print(f"  {line}")
        if len(exploit_code) > 600:
            print(f"  ... ({len(exploit_code) - 600} more characters)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Broadcast to dashboard
        if sio:
            sio.emit("chat_broadcast", {
                "sender": "system",
                "text": f"{status} — {cve_id} on {node_name}\n"
                        f"━━━ exploit_{cve_id.replace('-','_').lower()}.py ━━━\n"
                        f"{exploit_code[:600]}\n"
                        f"{'...(truncated)' if len(exploit_code) > 600 else ''}\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"{'Exploit delivered payload successfully.' if is_valid else 'Exploit attempt failed — target may be patched.'}"
            })

        logger.info(f"Exploit {'VALID' if is_valid else 'INVALID'} for {cve_id} on {node_name}")

    except Exception as e:
        logger.warning(f"Exploit generator hook failed: {e}")
        print(f"[⚠️ Exploit Gen Error] {e}")


def format_red_observation(obs: np.ndarray, node_names: List[str]) -> str:
    """Convert the numerical observation into a human-readable summary for the Red LLM."""
    obs = obs.reshape(-1, 8)
    n_nodes = len(node_names)
    lines = []
    lines.append("Current Network State (Visible to Attacker):")
    discovered_any = False
    
    for i in range(min(len(obs), n_nodes)):
        is_discovered = obs[i][0] > 0.5
        is_compromised = obs[i][4] > 0
        is_root = obs[i][4] >= 1.0
        
        if is_discovered:
            discovered_any = True
            status = "Compromised " if is_compromised else "Discovered"
            if is_root: status += "(ROOT)"
            lines.append(f"- Node {i} ({node_names[i]}): Status=[{status}]")

    if not discovered_any:
        lines.append("- No nodes discovered yet. You must scan the network.")
        
    lines.append("\nAvailable Actions:")
    lines.append(", ".join(RedAction.NAMES))
    
    return "\n".join(lines)


def format_blue_observation(obs: np.ndarray, node_names) -> str:
    """Convert the numerical observation into a human-readable summary for the Blue LLM."""
    obs = obs.reshape(-1, 8)
    # Accept either a list of names or an integer count
    if isinstance(node_names, int):
        node_names = [f"Node {i}" for i in range(node_names)]
    n_nodes = len(node_names)
    lines = []
    lines.append("Current Network State (Visible to Defender):")
    
    for i in range(min(len(obs), n_nodes)):
        alert_level = obs[i][0]
        has_vulns = obs[i][2] > 0
        is_isolated = obs[i][3] > 0.5
        
        flags = []
        if alert_level > 0.0: flags.append(f"AlertLevel={alert_level:.2f}")
        if has_vulns: flags.append("HasKnownVulns")
        if is_isolated: flags.append("Isolated")
        
        if flags:
            lines.append(f"- Node {i} ({node_names[i]}): {', '.join(flags)}")
        else:
            lines.append(f"- Node {i} ({node_names[i]}): Normal")
            
    lines.append("\nAvailable Actions:")
    lines.append(", ".join(BlueAction.NAMES))
    
    return "\n".join(lines)


class LLMRedAgent(BaseAgent):
    """Red Team agent powered by a Large Language Model (LangChain)."""

    def __init__(self, n_nodes: int, model_name: str = "gpt-4o"):
        super().__init__(name="LLM-Red", team="red")
        self.n_nodes = n_nodes
        self.model_name = model_name
        
        if not HAS_LANGCHAIN:
            logger.error("LangChain is not installed. LLM agents cannot be used.")
            self.llm = None
            return
            
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Fallback handling to prevent crashing entirely if key missing during init
            logger.warning("OPENAI_API_KEY not found in environment. LLM calls will fail unless provided.")
            api_key = "dummy-key-for-testing"
            
        self.llm = ChatOpenAI(model=model_name, api_key=api_key).with_structured_output(LLMRedActionPlan)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an elite, autonomous AI Red Team Hacker. Your goal is to logically compromise a target network by methodically navigating through the nodes. "
                       "You must analyze the network observation, think aloud, and choose the most optimal cyber attack action from the allowed list. "
                       "The target nodes are formatted as 'Node index (Name)'. Even if a string name is present, you MUST return the integer index (0 to {n_nodes}-1) of your chosen target node."),
            ("user", "{observation}")
        ])

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.llm is None:
            return np.array([0, 0])
            
        obs_text = format_red_observation(observation, getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)])
        
        try:
            chain = self.prompt | self.llm
            result: LLMRedActionPlan = chain.invoke({"observation": obs_text, "n_nodes": self.n_nodes})
            
            # Print thoughts to console for the "thinking aloud" effect
            print(f"\n[🤖 LLM-Red Thoughts] {result.reasoning}")
            print(f"[🔴 LLM-Red Action] {result.action_name} -> Node {result.target_node_index}")
            
            # Map name to index
            action_idx = 0
            if result.action_name in RedAction.NAMES:
                action_idx = RedAction.NAMES.index(result.action_name)
                
            target = max(0, min(self.n_nodes - 1, result.target_node_index))
            
            # Exploit Generator: if exploiting, generate the attack script
            if action_idx == RedAction.NAMES.index('exploit') and self.llm is not None:
                names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
                node_name = names[target] if target < len(names) else f"Node {target}"
                _run_exploit_hook(self.llm, node_name, names, target, self.n_nodes)
            
            return np.array([action_idx, target])
            
        except Exception as e:
            logger.warning(f"LLM Red Agent failed to parse action: {e}. Falling back to default.")
            return np.array([0, 0])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "LLMRedAgent does not use RL training"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "LLMRedAgent":
        return self

    def clone(self) -> "LLMRedAgent":
        return LLMRedAgent(self.n_nodes, self.model_name)


class LLMBlueAgent(BaseAgent):
    """Blue Team agent powered by a Large Language Model (LangChain)."""

    def __init__(self, n_nodes: int, model_name: str = "gpt-4o"):
        super().__init__(name="LLM-Blue", team="blue")
        self.n_nodes = n_nodes
        self.model_name = model_name
        
        if not HAS_LANGCHAIN:
            logger.error("LangChain is not installed. LLM agents cannot be used.")
            self.llm = None
            return
            
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment. LLM calls will fail unless provided.")
            api_key = "dummy-key-for-testing"
        
        self.llm = ChatOpenAI(model=model_name, api_key=api_key).with_structured_output(LLMBlueActionPlan)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an elite, autonomous AI Blue Team Defender. Your goal is to protect the organizational network. "
                       "You must analyze the network observation, looking for alerts and vulnerabilities. "
                       "Think aloud step-by-step, then choose the most optimal defensive action from the allowed list. "
                       "The target nodes are formatted as 'Node index (Name)'. Even if you pick a node conceptually by name, you MUST return its integer index (0 to {n_nodes}-1). Do not repeatedly isolate normal nodes."),
            ("user", "{observation}")
        ])

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.llm is None:
            return np.array([0, 0])
            
        obs_text = format_blue_observation(observation, self.n_nodes)
        
        try:
            chain = self.prompt | self.llm
            result: LLMBlueActionPlan = chain.invoke({"observation": obs_text, "n_nodes": self.n_nodes})
            
            print(f"\n[🤖 LLM-Blue Thoughts] {result.reasoning}")
            print(f"[🔵 LLM-Blue Action] {result.action_name} -> Node {result.target_node_index}")
            
            action_idx = 0
            if result.action_name in BlueAction.NAMES:
                action_idx = BlueAction.NAMES.index(result.action_name)
                
            target = max(0, min(self.n_nodes - 1, result.target_node_index))
            
            # Auto-Remediation: if patching, generate the config fix
            if action_idx == BlueAction.PATCH_VULNERABILITY and self.llm is not None:
                names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
                node_name = names[target] if target < len(names) else f"Node {target}"
                _run_remediation_hook(self.llm, node_name, names, target, self.n_nodes)
            
            return np.array([action_idx, target])
            
        except Exception as e:
            logger.warning(f"LLM Blue Agent failed to parse action: {e}. Falling back to default.")
            return np.array([0, 0])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "LLMBlueAgent does not use RL training"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "LLMBlueAgent":
        return self

    def clone(self) -> "LLMBlueAgent":
        return LLMBlueAgent(self.n_nodes, self.model_name)


class CyberCommanderAgent(BaseAgent):
    """Blue Team agent driven by Human-in-the-Loop text commands via the Dashboard."""

    def __init__(self, n_nodes: int, model_name: str = "gpt-4o"):
        super().__init__(name="Cyber-Commander", team="blue")
        self.n_nodes = n_nodes
        self.model_name = model_name
        
        if not HAS_LANGCHAIN:
            logger.error("LangChain is not installed. Cyber Commander cannot be used.")
            self.llm = None
            return
            
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment. Cyber Commander calls will fail.")
            api_key = "dummy-key-for-testing"
        
        # We only need the LLM to parse the human's intent into a valid Action
        self.llm = ChatOpenAI(model=model_name, api_key=api_key).with_structured_output(LLMBlueActionPlan)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Cyber Commander AI Assistant. The Human Commander has given you a natural language order. "
                       "You must translate their order into the most appropriate exact action from the allowed list. "
                       "The human might mention nodes by name (e.g. 'dns-master-node' or 'node 2' or 'database'). You must correlate their intent to the correct integer node index based on the observation below. "
                       "Choose the single most appropriate action that fulfills the Commander's intent. "
                       "If the Commander's intent is unclear, default to 'monitor_traffic' or 'wait'. "
                       "Think step-by-step, pick the valid target node index (0 to {n_nodes}-1), and then format your output.\n\n"
                       "Current Network State for context:\n{observation}"),
            ("user", "Commander's Order: {human_command}")
        ])

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.llm is None:
            return np.array([0, 0])
            
        try:
            from dashboard.server import _commander_queue, _commander_cv
            import socketio
            
            # Use the global socketio from the server module to push a message
            from dashboard.server import socketio as sio
            
            sio.emit("chat_broadcast", {"sender": "system", "text": "Awaiting Commander's Orders..."})
            
            # Block and wait for human input from the queue
            human_command = None
            with _commander_cv:
                # wait_for prevents spurious wakeups and immediately catches populated elements
                _commander_cv.wait_for(lambda: len(_commander_queue) > 0, timeout=60.0)
                
                if _commander_queue:
                    human_command = _commander_queue.pop(0)

            if not human_command:
                sio.emit("chat_broadcast", {"sender": "system", "text": "Timeout waiting for orders. Taking default action (Wait)."})
                return np.array([BlueAction.WAIT, 0])
                
            obs_text = format_blue_observation(observation, getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)])
            
            # PEN TEST MODE: Preview what the Red Team could exploit
            if human_command.lower().startswith("pen test"):
                names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
                # Extract node name from command
                pen_target = human_command[len("pen test"):].strip()
                target_idx = 0
                for i, name in enumerate(names):
                    if pen_target.lower() in name.lower():
                        target_idx = i
                        break
                
                sio.emit("chat_broadcast", {
                    "sender": "llm",
                    "text": f"🔬 PEN TEST MODE: Running simulated attack preview on {names[target_idx]}...\n"
                            f"This shows what the Red Team COULD do. No actual changes to network state."
                })
                
                # Use a raw LLM (not structured) for exploit generation
                raw_llm = ChatOpenAI(model=self.model_name, api_key=os.environ.get("OPENAI_API_KEY", ""))
                _run_exploit_hook(raw_llm, names[target_idx], names, target_idx, self.n_nodes)
                
                # Default to monitoring the target after pen test
                return np.array([BlueAction.MONITOR_TRAFFIC, target_idx])
            
            # Use LLM to translate Human intent to exact Action parameters
            chain = self.prompt | self.llm
            result: LLMBlueActionPlan = chain.invoke({
                "observation": obs_text, 
                "n_nodes": self.n_nodes,
                "human_command": human_command
            })
            
            sio.emit("chat_broadcast", {
                "sender": "llm", 
                "text": f"Acknowledged. Executing: {result.action_name} on Node {result.target_node_index}\nReasoning: {result.reasoning}"
            })
            
            action_idx = BlueAction.WAIT
            for idx, name in enumerate(BlueAction.NAMES):
                if name.replace("_", "").lower() == result.action_name.replace("_", "").lower():
                    action_idx = idx
                    break
                    
            target = max(0, min(self.n_nodes - 1, result.target_node_index))
            
            # Auto-Remediation: if patching, generate the config fix
            if action_idx == BlueAction.PATCH_VULNERABILITY and self.llm is not None:
                names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
                node_name = names[target] if target < len(names) else f"Node {target}"
                _run_remediation_hook(self.llm, node_name, names, target, self.n_nodes)
            
            return np.array([action_idx, target])
            
        except Exception as e:
            logger.warning(f"Cyber Commander failed to parse action: {e}. Falling back to default.")
            return np.array([BlueAction.WAIT, 0])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "CyberCommanderAgent does not use RL training"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "CyberCommanderAgent":
        return self

    def clone(self) -> "CyberCommanderAgent":
        return CyberCommanderAgent(self.n_nodes, self.model_name)


class CyberRedCommanderAgent(BaseAgent):
    """Red Team agent driven by Human-in-the-Loop — the user plays as the attacker."""

    def __init__(self, n_nodes: int, model_name: str = "gpt-4o"):
        super().__init__(name="Red-Commander", team="red")
        self.n_nodes = n_nodes
        self.model_name = model_name

        if not HAS_LANGCHAIN:
            logger.error("LangChain is not installed. Red Commander cannot be used.")
            self.llm = None
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Red Commander calls will fail.")
            api_key = "dummy-key-for-testing"

        self.llm = ChatOpenAI(model=model_name, api_key=api_key).with_structured_output(LLMRedActionPlan)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are the Red Team Commander AI Assistant executing a Human Hacker's DIRECT orders. "
                       "CRITICAL RULE: You MUST execute the EXACT action the human requests. Do NOT substitute a different action. "
                       "If the human says 'exploit', you MUST choose 'exploit'. If they say 'exfiltrate', you MUST choose 'exfiltrate_data'. "
                       "If they say 'backdoor', you MUST choose 'install_backdoor'. NEVER override their intent with scan or wait. "
                       "Allowed actions: scan_network, scan_vulnerability, exploit, privilege_escalate, lateral_move, install_backdoor, "
                       "exfiltrate_data, cover_tracks, deploy_ransomware, phish_user, ddos_service, wait. "
                       "The human might mention nodes by name (e.g. 'web-server' or 'node 3'). Correlate to the closest matching node index. "
                       "If the exact node isn't found, pick the closest match by name. "
                       "Pick the valid target node index (0 to {n_nodes}-1).\n\n"
                       "Current Network State:\n{observation}"),
            ("user", "Hacker's Order: {human_command}")
        ])

    def act(self, observation: np.ndarray) -> np.ndarray:
        if self.llm is None:
            return np.array([0, 0])

        try:
            from dashboard.server import _commander_queue, _commander_cv
            from dashboard.server import socketio as sio

            sio.emit("chat_broadcast", {"sender": "system", "text": "⚔️ Awaiting Hacker's Orders..."})

            # Block and wait for human input
            human_command = None
            with _commander_cv:
                _commander_cv.wait_for(lambda: len(_commander_queue) > 0, timeout=60.0)
                if _commander_queue:
                    human_command = _commander_queue.pop(0)

            if not human_command:
                sio.emit("chat_broadcast", {"sender": "system", "text": "Timeout. Auto-scanning network..."})
                return np.array([RedAction.SCAN_NETWORK, 0])

            obs_text = format_red_observation(observation, getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)])

            chain = self.prompt | self.llm
            result: LLMRedActionPlan = chain.invoke({
                "observation": obs_text,
                "n_nodes": self.n_nodes,
                "human_command": human_command
            })

            sio.emit("chat_broadcast", {
                "sender": "llm",
                "text": f"🔴 Executing: {result.action_name} on Node {result.target_node_index}\nReasoning: {result.reasoning}"
            })

            action_idx = RedAction.WAIT
            for idx, name in enumerate(RedAction.NAMES):
                if name.replace("_", "").lower() == result.action_name.replace("_", "").lower():
                    action_idx = idx
                    break

            target = max(0, min(self.n_nodes - 1, result.target_node_index))

            # Exploit Generator hook
            if action_idx == RedAction.EXPLOIT and self.llm is not None:
                names = getattr(self, "node_names", None) or [f"Node {i}" for i in range(self.n_nodes)]
                node_name = names[target] if target < len(names) else f"Node {target}"
                # Use the raw LLM (without structured output) for exploit gen
                raw_llm = ChatOpenAI(model=self.model_name, api_key=os.environ.get("OPENAI_API_KEY", ""))
                _run_exploit_hook(raw_llm, node_name, names, target, self.n_nodes)

            return np.array([action_idx, target])

        except Exception as e:
            logger.warning(f"Red Commander failed: {e}. Falling back to scan.")
            return np.array([RedAction.SCAN_NETWORK, 0])

    def learn(self, env: Any, total_timesteps: int, **kwargs) -> Dict:
        return {"message": "CyberRedCommanderAgent does not use RL training"}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> "CyberRedCommanderAgent":
        return self

    def clone(self) -> "CyberRedCommanderAgent":
        return CyberRedCommanderAgent(self.n_nodes, self.model_name)
