<div align="center">

# 🥋 CyberDojo

### Adversarial AI War Games — Where LLMs Battle for Network Supremacy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI GPT-4o](https://img.shields.io/badge/LLM-GPT--4o-412991.svg)](https://openai.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**A co-evolutionary LLM-augmented adversarial simulation platform for autonomous cyber attack and defense with multi-agent APT swarms.**

[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Game Modes](#-game-modes) · [Research Paper](#-research-paper) · [Screenshots](#-screenshots)

</div>

---

## 🧠 What is CyberDojo?

CyberDojo is a **first-of-its-kind** cybersecurity simulation platform where AI-powered **Red Team (attackers)** and **Blue Team (defenders)** battle autonomously over a simulated network — and **you can take command of either side**.

Unlike traditional cyber ranges that use scripted scenarios, CyberDojo combines:

- 🤖 **Reinforcement Learning** (PPO/DQN) for agents that learn optimal strategies
- 🧠 **Large Language Models** (GPT-4o) for agents that reason in natural language
- 👤 **Human-in-the-Loop** for you to command either team through a live dashboard
- 🐝 **Multi-Agent APT Swarms** where 3 specialized LLM hackers coordinate attacks
- ⚡ **LLM Code Generation** that writes real Python exploits and config patches during battle

> **The result?** Agents that don't just choose abstract "attack" or "defend" — they write actual exploit scripts, generate configuration patches, communicate in a hacker chatroom, and produce interpretable reasoning for every decision.

---

## ✨ Features

### 🔴 Red Team (Attack)
| Feature | Description |
|---------|------------|
| **12 Attack Actions** | scan_network, scan_vulnerability, exploit, privilege_escalate, lateral_move, install_backdoor, exfiltrate_data, cover_tracks, deploy_ransomware, phish_user, ddos_service, wait |
| **LLM Exploit Generator** | GPT-4o writes full Python exploit scripts (SQL injection, buffer overflow, SSH bypass, DNS poisoning) targeting specific CVEs on specific nodes |
| **APT Swarm Mode** | 3 coordinated LLM agents — Scout 🔍, Breacher 💥, Exfiltrator 📤 — communicate via a shared Hacker Chatroom |
| **Red Commander** | Play as the hacker yourself with point-and-click attack controls |
| **MITRE ATT&CK** | Every action maps to real-world ATT&CK techniques (T1046, T1190, T1041, etc.) |

### 🔵 Blue Team (Defense)
| Feature | Description |
|---------|------------|
| **12 Defense Actions** | monitor_traffic, analyze_alert, patch_vulnerability, isolate_node, restore_backup, update_firewall, deploy_honeypot, forensic_analysis, deploy_ids_rule, segment_network, rotate_credentials, wait |
| **Auto-Remediation** | LLM generates actual config patches (nginx.conf, sshd_config, smb.conf) for detected vulnerabilities |
| **Pen Test Preview** | Preview what exploits Red could deploy against any node before they attack |
| **Blue Commander** | Command the defense team against AI-driven attackers |

### 🌐 Platform
| Feature | Description |
|---------|------------|
| **Infinite Scenarios** | Type "Hospital Network" or "Military Base" → LLM generates a complete network topology with realistic nodes, services, subnets, and CVEs |
| **Real-Time Dashboard** | D3.js network graph, live team stats, battle log, commander chat, hacker chatroom |
| **Co-Evolutionary Training** | Red and Blue agents evolve together with ELO rating system |
| **Sim-to-Real Export** | Export defensive playbooks as executable Bash scripts |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for LLM features)

### Installation

```bash
# Clone the repository
git clone https://github.com/satyamdas03/CyberDojo.git
cd CyberDojo

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
$env:OPENAI_API_KEY = "your-api-key-here"    # PowerShell
# export OPENAI_API_KEY="your-api-key-here"  # Linux/Mac
```

### Run Your First Battle

```bash
# 🤖 AI vs AI — Watch LLM agents battle each other
python main.py battle --red llm --blue llm --scenario "Hospital Network" --visualize

# 🎮 Play as the Hacker — You attack, AI defends
python main.py battle --red commander --blue llm --scenario "Corporate Network" --visualize

# 🛡️ Play as the Defender — AI attacks, you defend
python main.py battle --red llm --blue commander --scenario "Military Base" --visualize

# 🐝 APT Swarm vs You — 3 coordinated hackers attack your network
python main.py battle --red swarm --blue commander --scenario "Bank" --visualize

# 🏋️ Train RL agents through co-evolution
python main.py train --red-algo ppo --blue-algo ppo --timesteps 100000
```
---

## 🏗️ Architecture

```
CyberDojo/
├── main.py                    # CLI entry point (train, battle, demo, benchmark)
├── requirements.txt           # Python dependencies
├── cyberdojo/                 # Core simulation engine
│   ├── __init__.py
│   ├── environment.py         # OpenAI Gym environment (CyberDojoEnv)
│   ├── network.py             # Network topology & node management
│   ├── agents.py              # RL agents (PPO, DQN, Scripted, Random)
│   ├── llm_agents.py          # LLM agents (Red, Blue, Commander, Red Commander)
│   ├── apt_swarm.py           # Multi-Agent APT Swarm (Scout, Breacher, Exfiltrator)
│   ├── exploit_gen.py         # LLM exploit code generation engine
│   ├── remediation.py         # Auto-remediation config patch generator
│   ├── llm_scenario.py        # Infinite LLM scenario generator
│   ├── rewards.py             # Reward shaping for RL training
│   ├── trainer.py             # Co-evolutionary training loop with ELO
│   ├── mitre.py               # MITRE ATT&CK technique mapping
│   ├── sim2real.py            # Sim-to-Real playbook export
│   └── config.py              # Configuration management
├── dashboard/                 # Real-time web dashboard
│   ├── server.py              # Flask-SocketIO backend
│   ├── index.html             # Dashboard UI
│   ├── dashboard.js           # D3.js network visualization + WebSocket logic
│   └── styles.css             # Cyberpunk-themed styling
└── tests/                     # Test suite
    └── test_*.py
```

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CYBERDOJO ENGINE                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              CyberDojoEnv (OpenAI Gym)                    │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │   │
│  │  │ Network     │  │ State Machine│  │ Reward Engine │   │   │
│  │  │ Topology    │  │ 12 Red acts  │  │ Red & Blue    │   │   │
│  │  │ Nodes/Edges │  │ 12 Blue acts │  │ reward shaping│   │   │
│  │  └─────────────┘  └──────────────┘  └───────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─── RED TEAM ──────────────┐  ┌─── BLUE TEAM ─────────────┐  │
│  │ • RL Agent (PPO/DQN)      │  │ • RL Agent (PPO/DQN)      │  │
│  │ • LLM Agent (GPT-4o)     │  │ • LLM Agent (GPT-4o)      │  │
│  │ • Red Commander (Human)   │  │ • Blue Commander (Human)   │  │
│  │ • APT Swarm (3× LLM)     │  │ • Auto-Remediation Engine │  │
│  │ • Exploit Code Generator  │  │ • Pen Test Preview         │  │
│  └───────────────────────────┘  └────────────────────────────┘  │
│                                                                  │
│  ┌─── DASHBOARD (Flask + D3.js + WebSocket) ─────────────────┐  │
│  │ Network Graph │ Team Stats │ Battle Log │ Commander Chat   │  │
│  │ Hacker Chatroom │ Action Menu │ Exploit Code Display      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Game Modes

### 1. 🤖 AI vs AI (`--red llm --blue llm`)
Watch two GPT-4o agents battle it out. The Red LLM reasons step-by-step about attack strategy while the Blue LLM analyzes threats and deploys defenses.

### 2. 🎮 Red Commander (`--red commander`)
**Play as the hacker.** Click on nodes to select targets, then choose from 8 attack buttons (Scan, Exploit, Backdoor, Phish, etc.). The LLM translates your orders into precise game actions.

### 3. 🛡️ Blue Commander (`--blue commander`)
**Play as the SOC analyst.** Defend your network against AI-driven attacks. Monitor alerts, isolate compromised nodes, deploy patches, and use the Pen Test button to preview potential exploits.

### 4. 🐝 APT Swarm (`--red swarm`)
Three coordinated LLM attackers assault your network:
- **Scout 🔍** — Maps the network and finds vulnerabilities
- **Breacher 💥** — Exploits the targets Scout identifies
- **Exfiltrator 📤** — Steals data from nodes Breacher compromises

They coordinate through a **Hacker Chatroom** visible as an "Intercepted Communications" panel on the dashboard.

### 5. 🏋️ RL Training (`train`)
Train Red and Blue RL agents through co-evolutionary self-play. Agents compete in rounds, earn ELO ratings, and the weakest are pruned while the strongest are cloned.

---

## 💀 LLM Exploit Generator

When the Red Team executes an `exploit` action, the LLM generates a **full Python exploit script** in real time:

```python
#!/usr/bin/env python3
# CVE-2024-2187 — SQL Injection on email-gateway (10.0.1.11)

import requests

def exploit_sql_injection(target_url):
    payload = "' UNION SELECT null, user, password FROM users -- "
    vulnerable_url = f"{target_url}/search?query={payload}"
    
    response = requests.get(vulnerable_url, timeout=10)
    if response.status_code == 200:
        print(f"[+] Data extracted: {response.text[:200]}")
        return True
    return False

if __name__ == "__main__":
    exploit_sql_injection("http://10.0.1.11")
```

**Supported exploit types:** SQL Injection, Buffer Overflow, SSH Auth Bypass, Remote Code Execution, LDAP Injection, DNS Cache Poisoning, SMB RCE, Privilege Escalation, Phishing payloads.

---

## 🔧 Auto-Remediation Engine

When the Blue Team selects `patch_vulnerability`, the LLM generates **actual configuration patches**:

```diff
# sshd_config — Fixing CVE-2024-4321 (SSH Buffer Overflow)
- Protocol 1,2
+ Protocol 2
- Ciphers aes128-cbc,3des-cbc,aes256-cbc
+ Ciphers aes256-gcm@openssh.com,chacha20-poly1305@openssh.com
- MaxAuthTries 10
+ MaxAuthTries 3
+ LoginGraceTime 30
```

**Supported configs:** nginx.conf, sshd_config, smb.conf, apache2.conf, mysql.cnf, postgresql.conf, firewall rules, kubernetes manifests, docker-compose.yml.

---

## 📊 Performance

| Red Agent | Nodes Compromised | Data Stolen | Exploit Scripts Generated |
|-----------|------------------|-------------|--------------------------|
| Random | 0.8 | 1.2 | 0 |
| Scripted | 1.5 | 3.8 | 0 |
| RL (PPO) | 2.8 | 8.5 | 0 |
| LLM (GPT-4o) | 2.5 | 7.2 | ~5 per battle |
| **APT Swarm** | **3.2** | **12.0** | ~8 per battle |

The APT Swarm achieves the **highest data exfiltration** due to its coordinated Scout→Breacher→Exfiltrator pipeline, while RL agents achieve better stealth (fewer detections).

---

## 📚 Research Paper

This project accompanies a research paper:

> **"CyberDojo: A Co-Evolutionary LLM-Augmented Adversarial Simulation Platform for Autonomous Cyber Attack and Defense with Multi-Agent APT Swarms"**
>
> *Satyam Das and S.P. Raja*

### Novel Contributions

1. **LLM Agents That Write Real Code** — First cybersecurity simulation where agents generate executable exploit scripts and configuration patches during gameplay
2. **Multi-Agent APT Swarm with Natural Language Coordination** — Three specialized LLM agents coordinate through a shared chatroom, mimicking real-world APT group behavior
3. **Hybrid RL+LLM Architecture** — First platform combining trained RL policies with LLM reasoning in cybersecurity
4. **Dual Human-in-the-Loop** — Seamlessly play as either the attacker or defender with full AI opposition
5. **Infinite LLM Scenario Generation** — Generate complete network topologies from text descriptions

### Competitive Landscape

| Capability | CyberBattleSim | CALDERA | PentestGPT | **CyberDojo** |
|-----------|:---:|:---:|:---:|:---:|
| RL Agents | ✅ | ❌ | ❌ | ✅ |
| LLM Agents | ❌ | ❌ | ✅ | ✅ |
| Multi-Agent APT | ❌ | Partial | ❌ | ✅ |
| Exploit Code Gen | ❌ | ❌ | ❌ | ✅ |
| Auto-Remediation | ❌ | ❌ | ❌ | ✅ |
| Human-in-the-Loop | ❌ | ❌ | Chat | ✅ |
| Infinite Scenarios | ❌ | ❌ | ❌ | ✅ |
| Live Dashboard | ❌ | Web UI | ❌ | ✅ |

---

## 🖥️ CLI Reference

```bash
# ─── Battle Mode ───
python main.py battle [OPTIONS]
  --red      {rl,scripted,random,llm,commander,swarm}   Red Team agent
  --blue     {rl,scripted,random,llm,commander}          Blue Team agent
  --scenario "Theme Name"     LLM-generated scenario
  --network-size {small,medium,large}
  --visualize                 Open the live dashboard
  --sim2real                  Export defensive playbook

# ─── Training Mode ───
python main.py train [OPTIONS]
  --red-algo  {ppo,dqn}      RL algorithm for Red
  --blue-algo {ppo,dqn}      RL algorithm for Blue
  --timesteps 100000         Training steps
  --co-evolve                Enable co-evolutionary training

# ─── Other ───
python main.py demo          Quick demo with visualizations
python main.py benchmark     Run performance benchmarks
python main.py dashboard     Launch dashboard only
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Core Engine | Python, NumPy, OpenAI Gym |
| RL Training | PyTorch, Stable-Baselines3 |
| LLM Integration | LangChain, OpenAI GPT-4o |
| Structured Output | Pydantic v2 |
| Dashboard | Flask, Flask-SocketIO, D3.js |
| Network Graph | NetworkX |
| Export | Bash, Jinja2 |

---

## ⚠️ Ethical Disclaimer

CyberDojo is designed **exclusively for educational and research purposes**. All exploit code is generated against simulated, fictional network infrastructure within a sandboxed environment. The platform includes safety measures:

- Exploits target only simulated IP addresses
- No real network connections are made
- The auto-remediation engine focuses on defensive applications
- Commander mode is designed for training security professionals

**Do not use generated exploit code against real systems without explicit authorization.**

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Satyam Das**
- GitHub: [@satyamdas03](https://github.com/satyamdas03)

---

<div align="center">

**Built with 🧠 AI and ☕ caffeine**

*If you find this project useful, please ⭐ star the repo!*

</div>
