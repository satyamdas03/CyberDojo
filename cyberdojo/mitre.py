"""
CyberDojo MITRE ATT&CK Mapping

Maps simulated in-game actions to real-world MITRE ATT&CK 
tactics and techniques for enterprise-grade reporting.
"""

from typing import Dict, Tuple

# Maps exact RedAction names to (Tactic, Technique_ID, Technique_Name)
RED_MITRE_MAP: Dict[str, Tuple[str, str, str]] = {
    "scan_network": ("Discovery", "T1046", "Network Service Discovery"),
    "scan_vulnerability": ("Discovery", "T1046", "Network Service Discovery"),  # Active Scanning is T1595 but usually pre-compromise
    "exploit": ("Execution", "T1203", "Exploitation for Client Execution"),
    "privilege_escalate": ("Privilege Escalation", "T1068", "Exploitation for Privilege Escalation"),
    "lateral_move": ("Lateral Movement", "T1210", "Exploitation of Remote Services"),
    "install_backdoor": ("Persistence", "T1505", "Server Software Component"),
    "exfiltrate_data": ("Exfiltration", "T1041", "Exfiltration Over C2 Channel"),
    "cover_tracks": ("Defense Evasion", "T1070", "Indicator Removal on Host"),
    "deploy_ransomware": ("Impact", "T1486", "Data Encrypted for Impact"),
    "phish_user": ("Initial Access", "T1566", "Phishing"),
    "ddos_service": ("Impact", "T1498", "Network Denial of Service"),
    "wait": ("Command and Control", "T1029", "Scheduled Transfer (Sleeping)"),
}

# Maps exact BlueAction names to D3FEND or custom defensive categories
BLUE_MITRE_MAP: Dict[str, Tuple[str, str, str]] = {
    "monitor_traffic": ("Detect", "D3-NTAD", "Network Traffic Analysis"),
    "analyze_alert": ("Detect", "D3-AL", "Alert Logic"),
    "isolate_node": ("Isolate", "D3-NI", "Network Isolation"),
    "patch_vulnerability": ("Evict", "D3-SP", "Software Patching"),
    "deploy_honeypot": ("Deceive", "D3-DN", "Decoy Network"),
    "restore_backup": ("Restore", "D3-SFB", "System File Backup Restoring"),
    "update_firewall": ("Isolate", "D3-FWR", "Firewall Rule Modification"),
    "forensic_analysis": ("Detect", "D3-HFA", "Host Forensic Analysis"),
    "deploy_ids_rule": ("Detect", "D3-IDSR", "IDS Rule Deployment"),
    "wait": ("Monitor", "D3-M", "Continuous Monitoring"),
}

def get_red_mitre(action_name: str) -> dict:
    """Get the MITRE ATT&CK metadata for a Red action."""
    tactic, t_code, technique = RED_MITRE_MAP.get(
        action_name, ("Unknown", "T0000", "Unknown Technique")
    )
    return {"tactic": tactic, "id": t_code, "name": technique}

def get_blue_mitre(action_name: str) -> dict:
    """Get the D3FEND metadata for a Blue action."""
    tactic, d3_code, technique = BLUE_MITRE_MAP.get(
        action_name, ("Unknown", "D3-000", "Unknown Technique")
    )
    return {"tactic": tactic, "id": d3_code, "name": technique}
