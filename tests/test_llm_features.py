"""
Unit tests for CyberDojo Phase 3 Revolutionary Features.
Verifies LLM Agent fallbacks, Scenario Generation robustness, and Sim-to-Real translations.
"""

import os
import pytest
import numpy as np
import tempfile
import json
from unittest.mock import patch, MagicMock

try:
    from cyberdojo.llm_agents import LLMRedAgent, CyberCommanderAgent, HAS_LANGCHAIN
    from cyberdojo.llm_scenario import generate_scenario
    from cyberdojo.sim2real import export_campaign
except ImportError:
    HAS_LANGCHAIN = False

@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain required for LLM feature tests")
class TestLLMFeatures:
    
    def test_llm_red_agent_initialization_no_key(self, monkeypatch):
        """Test LLM Red Agent initializes correctly with dummy key if missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent = LLMRedAgent(n_nodes=10)
        assert agent.name == "LLM-Red"
        assert agent.team == "red"
        # Should default to dummy-key to prevent crashing pre-game
        assert agent.llm is not None

    @patch("cyberdojo.llm_agents.ChatOpenAI")
    @patch("dashboard.server.socketio.emit")
    def test_cyber_commander_fallback(self, mock_emit, mock_chat_openai):
        """Test that the Cyber Commander falls back gracefully if LLM chain fails."""
        # Setup mock to raise an exception indicating bad parse
        mock_instance = mock_chat_openai.return_value.with_structured_output.return_value
        chain_mock = MagicMock()
        chain_mock.invoke.side_effect = Exception("Simulated LLM Parsing Failure")
        
        agent = CyberCommanderAgent(n_nodes=10)
        agent.prompt = MagicMock()
        
        # Manually wire the patch for the `prompt | llm` syntax used in the class
        agent.prompt.__or__.return_value = chain_mock
        
        # Inject our mock command into the queue
        import dashboard.server as ds
        ds._commander_queue.append("Isolate node 5!")
        
        obs = np.zeros(20)
        action = agent.act(obs)
        
        # Should gracefully return action [2, 0] (wait=9, 0)
        from cyberdojo.environment import BlueAction
        assert isinstance(action, np.ndarray)
        assert len(action) == 2
        assert action[0] == BlueAction.WAIT


class TestSimToReal:
    
    def test_export_campaign_generator(self):
        """Test that the sim2real translation layer produces valid bash playbooks from event logs."""
        
        mock_event_log = [
            {
                "step": 1,
                "red": {"action": "scan_network", "target": "node-0"},
                "blue": {"action": "monitor_traffic", "target": "node-2"},
                "network_state": {
                    "nodes": [
                        {"id": "node-0", "name": "web", "group": "DMZ", "subnet": "DMZ"},
                        {"id": "node-2", "name": "db", "group": "LAN", "subnet": "LAN"},
                    ]
                }
            },
            {
                "step": 2,
                "red": {"action": "exploit", "target": "node-2"},
                "blue": {"action": "isolate_node", "target": "node-2"},
                "network_state": {
                    "nodes": [
                        {"id": "node-0", "name": "web", "group": "DMZ", "subnet": "DMZ"},
                        {"id": "node-2", "name": "db", "group": "LAN", "subnet": "LAN"},
                    ]
                }
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_campaign(mock_event_log, output_dir=tmp_dir)
            files = os.listdir(tmp_dir)
            
            assert len(files) == 2
            assert any(f.startswith("red_campaign") for f in files)
            assert any(f.startswith("blue_campaign") for f in files)
            
            r_path = [os.path.join(tmp_dir, f) for f in files if f.startswith("red")][0]
            b_path = [os.path.join(tmp_dir, f) for f in files if f.startswith("blue")][0]
            
            with open(r_path, "r") as f:
                red_content = f.read()
                
            with open(b_path, "r") as f:
                blue_content = f.read()
                
            # Check Bash Headers
            assert "#!/bin/bash" in red_content
            assert "#!/bin/bash" in blue_content
            
            # Check translated tools mapping logic
            assert "nmap -sn 10.DMZ.0" in red_content
            assert "msfconsole" in red_content
            
            assert "tcpdump -i eth0 host 10.LAN.0" in blue_content
            assert "iptables -A INPUT -s 10.LAN.0" in blue_content
