"""
CyberDojo Network Topology Simulation

Simulates a realistic network with nodes, subnets, services, and
vulnerabilities. Provides the world model for the RL environment.
"""

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import networkx as nx

from cyberdojo.config import NetworkConfig


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class AccessLevel(Enum):
    """Access levels on a compromised node."""
    NONE = 0
    USER = 1
    ROOT = 2


class NodeStatus(Enum):
    """Security status of a node."""
    CLEAN = auto()
    COMPROMISED = auto()
    DETECTED = auto()
    ISOLATED = auto()
    RESTORED = auto()


class ServiceType(Enum):
    """Types of network services."""
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    FTP = "ftp"
    SMB = "smb"
    DNS = "dns"
    SMTP = "smtp"
    MYSQL = "mysql"
    RDP = "rdp"
    TELNET = "telnet"


class VulnSeverity(Enum):
    """CVSS-inspired vulnerability severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ─────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────

@dataclass
class Vulnerability:
    """A vulnerability associated with a service."""
    vuln_id: str  # e.g., "CVE-2024-1234"
    name: str
    severity: VulnSeverity
    exploit_difficulty: float  # 0.0 (trivial) to 1.0 (very hard)
    grants_root: bool = False  # whether exploiting grants root access
    is_patched: bool = False
    is_known_to_defender: bool = False  # blue team knows about it

    @property
    def exploit_probability(self) -> float:
        """Probability of successful exploitation."""
        if self.is_patched:
            return 0.0
        # Higher severity = easier to exploit (more tooling available)
        base = 1.0 - self.exploit_difficulty
        severity_bonus = self.severity.value * 0.05
        return min(0.95, base + severity_bonus)


@dataclass
class Service:
    """A network service running on a node."""
    service_type: ServiceType
    port: int
    version: str
    is_running: bool = True
    vulnerabilities: List[Vulnerability] = field(default_factory=list)

    @property
    def has_exploitable_vulns(self) -> bool:
        return any(not v.is_patched for v in self.vulnerabilities)


@dataclass
class Node:
    """A network node (server, workstation, router)."""
    node_id: str
    name: str
    subnet_id: int
    os_type: str  # "linux", "windows", "router"
    is_critical: bool = False  # high-value target
    is_honeypot: bool = False

    # Security state
    status: NodeStatus = NodeStatus.CLEAN
    access_level: AccessLevel = AccessLevel.NONE
    has_backdoor: bool = False
    data_value: float = 1.0  # value of data on this node (1-10)
    is_encrypted: bool = False  # ransomware encrypted

    # Services
    services: List[Service] = field(default_factory=list)

    # Visibility
    is_discovered_by_red: bool = False
    scan_count: int = 0  # number of times scanned (increases detection chance)
    alert_level: float = 0.0  # how suspicious this node looks to blue (0-1)

    @property
    def is_compromised(self) -> bool:
        return self.status == NodeStatus.COMPROMISED

    @property
    def is_isolated(self) -> bool:
        return self.status == NodeStatus.ISOLATED

    @property
    def unpatched_vulns(self) -> List[Vulnerability]:
        vulns = []
        for service in self.services:
            vulns.extend(v for v in service.vulnerabilities if not v.is_patched)
        return vulns

    def reset(self) -> None:
        """Reset node to clean state."""
        self.status = NodeStatus.CLEAN
        self.access_level = AccessLevel.NONE
        self.has_backdoor = False
        self.is_encrypted = False
        self.is_discovered_by_red = False
        self.scan_count = 0
        self.alert_level = 0.0
        for service in self.services:
            service.is_running = True
            for vuln in service.vulnerabilities:
                vuln.is_patched = False
                vuln.is_known_to_defender = False


@dataclass
class FirewallRule:
    """A firewall rule between two subnets."""
    source_subnet: int
    dest_subnet: int
    allowed_ports: Set[int] = field(default_factory=set)
    is_active: bool = True

    def allows_port(self, port: int) -> bool:
        if not self.is_active:
            return True  # disabled firewall allows everything
        if not self.allowed_ports:
            return True  # empty = allow all
        return port in self.allowed_ports


class Subnet:
    """A network subnet containing multiple nodes."""

    def __init__(self, subnet_id: int, name: str, is_dmz: bool = False):
        self.subnet_id = subnet_id
        self.name = name
        self.is_dmz = is_dmz
        self.nodes: Dict[str, Node] = {}
        self.ids_enabled: bool = True
        self.ids_sensitivity: float = 0.5  # 0-1, higher = more alerts
        self.traffic_monitor_level: float = 0.0  # blue team monitoring intensity

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node

    @property
    def compromised_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.is_compromised]

    @property
    def clean_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values()
                if n.status == NodeStatus.CLEAN]


# ─────────────────────────────────────────────────────────────
# Vulnerability Database
# ─────────────────────────────────────────────────────────────

VULN_TEMPLATES = {
    ServiceType.HTTP: [
        ("CVE-2024-3094", "Remote Code Execution in Web Server",
         VulnSeverity.CRITICAL, 0.3, True),
        ("CVE-2024-2187", "SQL Injection in Web App",
         VulnSeverity.HIGH, 0.4, False),
        ("CVE-2024-5521", "Cross-Site Scripting (XSS)",
         VulnSeverity.MEDIUM, 0.5, False),
        ("CVE-2024-1100", "Directory Traversal",
         VulnSeverity.HIGH, 0.35, False),
    ],
    ServiceType.SSH: [
        ("CVE-2024-6789", "SSH Authentication Bypass",
         VulnSeverity.CRITICAL, 0.25, True),
        ("CVE-2024-4321", "SSH Buffer Overflow",
         VulnSeverity.HIGH, 0.45, True),
    ],
    ServiceType.FTP: [
        ("CVE-2024-7654", "FTP Anonymous Login",
         VulnSeverity.MEDIUM, 0.6, False),
        ("CVE-2024-8888", "FTP Buffer Overflow",
         VulnSeverity.HIGH, 0.4, True),
    ],
    ServiceType.SMB: [
        ("CVE-2024-9012", "SMB Remote Code Execution",
         VulnSeverity.CRITICAL, 0.2, True),
        ("CVE-2024-3456", "SMB Information Disclosure",
         VulnSeverity.MEDIUM, 0.5, False),
    ],
    ServiceType.DNS: [
        ("CVE-2024-1357", "DNS Cache Poisoning",
         VulnSeverity.HIGH, 0.5, False),
    ],
    ServiceType.SMTP: [
        ("CVE-2024-2468", "SMTP Open Relay",
         VulnSeverity.MEDIUM, 0.6, False),
        ("CVE-2024-1111", "SMTP Buffer Overflow",
         VulnSeverity.HIGH, 0.4, True),
    ],
    ServiceType.MYSQL: [
        ("CVE-2024-5678", "MySQL Authentication Bypass",
         VulnSeverity.CRITICAL, 0.3, True),
        ("CVE-2024-9999", "MySQL Privilege Escalation",
         VulnSeverity.HIGH, 0.35, True),
    ],
    ServiceType.RDP: [
        ("CVE-2024-7777", "RDP Remote Code Execution",
         VulnSeverity.CRITICAL, 0.25, True),
        ("CVE-2024-6666", "RDP Man-in-the-Middle",
         VulnSeverity.HIGH, 0.5, False),
    ],
    ServiceType.TELNET: [
        ("CVE-2024-4444", "Telnet Cleartext Credentials",
         VulnSeverity.HIGH, 0.7, False),
    ],
    ServiceType.HTTPS: [
        ("CVE-2024-5555", "TLS Downgrade Attack",
         VulnSeverity.HIGH, 0.45, False),
        ("CVE-2024-3333", "Certificate Validation Bypass",
         VulnSeverity.MEDIUM, 0.55, False),
    ],
}

# Realistic node name templates
NODE_NAMES = {
    "linux": [
        "web-server", "api-server", "db-server", "mail-server",
        "file-server", "app-server", "proxy-server", "log-server",
        "backup-server", "ci-server", "docker-host", "k8s-node",
    ],
    "windows": [
        "dc-primary", "workstation", "dev-machine", "admin-pc",
        "hr-desktop", "finance-pc", "exec-laptop", "print-server",
        "exchange-server", "sharepoint-server",
    ],
    "router": [
        "core-router", "edge-router", "fw-gateway", "vpn-gateway",
    ],
}

SUBNET_NAMES = [
    "DMZ", "Corporate-LAN", "Server-Farm", "Dev-Network",
    "Management", "Guest-WiFi", "IoT-Segment", "Database-Tier",
    "Backup-Network", "Research-Lab",
]


# ─────────────────────────────────────────────────────────────
# Network Builder
# ─────────────────────────────────────────────────────────────

class Network:
    """
    Full network topology with nodes, subnets, and connectivity.
    This is the world model for the CyberDojo environment.
    """

    def __init__(self, config: Optional[NetworkConfig] = None, seed: Optional[int] = None):
        self.config = config or NetworkConfig.small()
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.subnets: Dict[int, Subnet] = {}
        self.nodes: Dict[str, Node] = {}
        self.graph: nx.Graph = nx.Graph()
        self.firewall_rules: Dict[Tuple[int, int], FirewallRule] = {}
        self.critical_target_id: Optional[str] = None
        self.entry_point_id: Optional[str] = None

        self._build_network()

    def _build_network(self) -> None:
        """Construct the full network topology."""
        if getattr(self.config, "scenario_data", None) is not None:
            self._build_from_scenario(self.config.scenario_data)
            return

        node_counter = 0
        used_names = set()

        # Create subnets
        for i in range(self.config.num_subnets):
            is_dmz = (i == 0 and self.config.has_dmz)
            subnet_name = SUBNET_NAMES[i] if i < len(SUBNET_NAMES) else f"Subnet-{i}"
            subnet = Subnet(subnet_id=i, name=subnet_name, is_dmz=is_dmz)

            # Create nodes in this subnet
            n_nodes = self.config.nodes_per_subnet[i] \
                if i < len(self.config.nodes_per_subnet) else 3
            for j in range(n_nodes):
                node = self._create_node(
                    node_id=f"node-{node_counter}",
                    subnet_id=i,
                    is_dmz=is_dmz,
                    used_names=used_names,
                )
                subnet.add_node(node)
                self.nodes[node.node_id] = node
                self.graph.add_node(node.node_id, subnet=i)
                node_counter += 1

            self.subnets[i] = subnet

        # Designate entry point (first DMZ node or first node)
        # Red team discovers all DMZ/public-facing nodes initially
        dmz_nodes = list(self.subnets[0].nodes.values())
        if dmz_nodes:
            self.entry_point_id = dmz_nodes[0].node_id
            for dmz_node in dmz_nodes:
                dmz_node.is_discovered_by_red = True  # attacker can see DMZ

        # Designate critical target
        if self.config.has_critical_server and len(self.subnets) >= 2:
            last_subnet = self.subnets[self.config.num_subnets - 1]
            critical_nodes = list(last_subnet.nodes.values())
            if critical_nodes:
                target = critical_nodes[-1]
                target.is_critical = True
                target.data_value = 10.0
                target.name = "crown-jewels-db"
                self.critical_target_id = target.node_id

        # Build intra-subnet edges (full mesh within each subnet)
        for subnet in self.subnets.values():
            node_ids = list(subnet.nodes.keys())
            for i_node in range(len(node_ids)):
                for j_node in range(i_node + 1, len(node_ids)):
                    self.graph.add_edge(node_ids[i_node], node_ids[j_node],
                                        weight=1, inter_subnet=False)

        # Build inter-subnet edges (controlled connectivity)
        subnet_ids = list(self.subnets.keys())
        for i_sub in range(len(subnet_ids)):
            for j_sub in range(i_sub + 1, len(subnet_ids)):
                if self.rng.random() < self.config.connectivity or abs(i_sub - j_sub) == 1:
                    # Connect one node from each subnet
                    sub_a = list(self.subnets[subnet_ids[i_sub]].nodes.keys())
                    sub_b = list(self.subnets[subnet_ids[j_sub]].nodes.keys())
                    if sub_a and sub_b:
                        a_node = self.rng.choice(sub_a)
                        b_node = self.rng.choice(sub_b)
                        self.graph.add_edge(a_node, b_node,
                                            weight=2, inter_subnet=True)

                    # Create firewall rule
                    rule = FirewallRule(
                        source_subnet=subnet_ids[i_sub],
                        dest_subnet=subnet_ids[j_sub],
                        allowed_ports=self._default_allowed_ports(
                            subnet_ids[i_sub], subnet_ids[j_sub]),
                    )
                    self.firewall_rules[(subnet_ids[j_sub], subnet_ids[i_sub])] = rule

    def _build_from_scenario(self, scenario: dict) -> None:
        """Construct the network from a deterministic LLM-generated scenario."""
        # 1. Build Subnets
        # Because we might override config.num_subnets, we update it to match the scenario
        self.config.num_subnets = len(scenario["subnets"])
        for i, sub_data in enumerate(scenario["subnets"]):
            subnet = Subnet(subnet_id=i, name=sub_data["name"], is_dmz=sub_data["is_dmz"])
            self.subnets[i] = subnet

        # 2. Build Nodes
        for i, node_data in enumerate(scenario["nodes"]):
            node_id = f"node-{i}"
            subnet_idx = min(node_data["subnet_index"], len(self.subnets) - 1)
            
            node = Node(
                node_id=node_id,
                name=node_data["name"],
                subnet_id=subnet_idx,
                os_type=node_data.get("os_type", "linux"),
                is_critical=node_data.get("is_critical", False),
                data_value=node_data.get("data_value", 5.0),
            )
            
            # Setup specific services assigned by LLM
            # Map string name to ServiceType Enum
            svc_map = {e.value: e for e in ServiceType}
            for svc_str in node_data.get("services", []):
                svc_enum = svc_map.get(svc_str.lower())
                if svc_enum:
                    port = self._default_port(svc_enum)
                    service = Service(
                        service_type=svc_enum,
                        port=port,
                        version=f"{self.rng.randint(1, 4)}.{self.rng.randint(0, 9)}",
                    )
                    
                    # Add vulnerabilities
                    if svc_enum in VULN_TEMPLATES:
                        for vuln_template in VULN_TEMPLATES[svc_enum]:
                            if self.rng.random() < self.config.vulnerability_density:
                                vuln = Vulnerability(
                                    vuln_id=vuln_template[0],
                                    name=vuln_template[1],
                                    severity=vuln_template[2],
                                    exploit_difficulty=vuln_template[3] + self.rng.uniform(-0.1, 0.1),
                                    grants_root=vuln_template[4],
                                )
                                service.vulnerabilities.append(vuln)
                    
                    node.services.append(service)

            # Setup Critical and Entry status manually
            if node.is_critical:
                self.critical_target_id = node.node_id
                
            self.nodes[node_id] = node
            self.subnets[subnet_idx].add_node(node)
            self.graph.add_node(node_id, subnet=subnet_idx)

        # 3. Designate entry point based on DMZ exactly like random logic
        for sub_id, subnet in self.subnets.items():
            if subnet.is_dmz and subnet.nodes:
                self.entry_point_id = list(subnet.nodes.values())[0].node_id
                for dmz_node in subnet.nodes.values():
                    dmz_node.is_discovered_by_red = True
                break

        # 4. Intra-subnet full-mesh edges
        for subnet in self.subnets.values():
            node_ids = list(subnet.nodes.keys())
            for i_node in range(len(node_ids)):
                for j_node in range(i_node + 1, len(node_ids)):
                    self.graph.add_edge(node_ids[i_node], node_ids[j_node], weight=1, inter_subnet=False)

        # 5. Inter-subnet connectivity (guarantees a path)
        subnet_ids = list(self.subnets.keys())
        for i_sub in range(len(subnet_ids) - 1):
            j_sub = i_sub + 1
            sub_a = list(self.subnets[subnet_ids[i_sub]].nodes.keys())
            sub_b = list(self.subnets[subnet_ids[j_sub]].nodes.keys())
            if sub_a and sub_b:
                a_node = self.rng.choice(sub_a)
                b_node = self.rng.choice(sub_b)
                self.graph.add_edge(a_node, b_node, weight=2, inter_subnet=True)
                rule = FirewallRule(
                    source_subnet=subnet_ids[i_sub],
                    dest_subnet=subnet_ids[j_sub],
                    allowed_ports=self._default_allowed_ports(subnet_ids[i_sub], subnet_ids[j_sub]),
                )
                self.firewall_rules[(subnet_ids[i_sub], subnet_ids[j_sub])] = rule
                self.firewall_rules[(subnet_ids[j_sub], subnet_ids[i_sub])] = rule

    def _create_node(self, node_id: str, subnet_id: int, is_dmz: bool,
                     used_names: set) -> Node:
        """Create a single node with randomized services and vulnerabilities."""
        # Pick OS type
        if is_dmz:
            os_type = "linux"
        else:
            os_type = self.rng.choice(["linux", "windows", "linux"])  # bias toward linux

        # Pick name
        name_pool = [n for n in NODE_NAMES.get(os_type, NODE_NAMES["linux"])
                     if n not in used_names]
        if not name_pool:
            name = f"{os_type}-{node_id}"
        else:
            name = self.rng.choice(name_pool)
        used_names.add(name)

        # Create node
        node = Node(
            node_id=node_id,
            name=name,
            subnet_id=subnet_id,
            os_type=os_type,
            data_value=self.rng.uniform(1.0, 5.0),
        )

        # Add services
        n_services = self.rng.randint(*self.config.services_per_node)
        available_services = list(ServiceType)
        if is_dmz:
            # DMZ nodes are more likely to have web-facing services
            preferred = [ServiceType.HTTP, ServiceType.HTTPS, ServiceType.DNS]
            services_to_add = preferred[:min(n_services, len(preferred))]
            remaining = n_services - len(services_to_add)
            if remaining > 0:
                others = [s for s in available_services if s not in services_to_add]
                services_to_add.extend(self.rng.sample(others,
                                                        min(remaining, len(others))))
        else:
            services_to_add = self.rng.sample(available_services,
                                               min(n_services, len(available_services)))

        for svc_type in services_to_add:
            port = self._default_port(svc_type)
            service = Service(
                service_type=svc_type,
                port=port,
                version=f"{self.rng.randint(1, 5)}.{self.rng.randint(0, 9)}.{self.rng.randint(0, 20)}",
            )

            # Add vulnerabilities
            if svc_type in VULN_TEMPLATES:
                for vuln_template in VULN_TEMPLATES[svc_type]:
                    if self.rng.random() < self.config.vulnerability_density:
                        vuln = Vulnerability(
                            vuln_id=vuln_template[0],
                            name=vuln_template[1],
                            severity=vuln_template[2],
                            exploit_difficulty=vuln_template[3] + self.rng.uniform(-0.1, 0.1),
                            grants_root=vuln_template[4],
                        )
                        service.vulnerabilities.append(vuln)

            node.services.append(service)

        return node

    def _default_port(self, service_type: ServiceType) -> int:
        """Return the default port for a service type."""
        port_map = {
            ServiceType.HTTP: 80,
            ServiceType.HTTPS: 443,
            ServiceType.SSH: 22,
            ServiceType.FTP: 21,
            ServiceType.SMB: 445,
            ServiceType.DNS: 53,
            ServiceType.SMTP: 25,
            ServiceType.MYSQL: 3306,
            ServiceType.RDP: 3389,
            ServiceType.TELNET: 23,
        }
        return port_map.get(service_type, 8080)

    def _default_allowed_ports(self, src: int, dst: int) -> Set[int]:
        """Default firewall rules — DMZ is more restricted."""
        base_ports = {80, 443, 53}  # HTTP, HTTPS, DNS always allowed
        if src == 0 or dst == 0:  # DMZ rules
            return base_ports
        # Internal networks have more open access
        return base_ports | {22, 445, 3306, 3389}

    # ─────────────────────────────────────────────────────────
    # State Queries
    # ─────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def get_adjacent_nodes(self, node_id: str) -> List[Node]:
        """Get neighboring nodes accessible from the given node."""
        if node_id not in self.graph:
            return []
        neighbor_ids = list(self.graph.neighbors(node_id))
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_reachable_nodes(self, node_id: str) -> List[Node]:
        """Get all nodes reachable from the given node (respecting isolation)."""
        reachable = []
        for nid in nx.descendants(self.graph, node_id) | {node_id}:
            node = self.nodes.get(nid)
            if node and not node.is_isolated:
                reachable.append(node)
        return reachable

    def can_reach(self, source_id: str, target_id: str) -> bool:
        """Check if a node can reach another (considering isolation & firewalls)."""
        src_node = self.nodes.get(source_id)
        tgt_node = self.nodes.get(target_id)
        if not src_node or not tgt_node:
            return False
        if src_node.is_isolated or tgt_node.is_isolated:
            return False
        return nx.has_path(self.graph, source_id, target_id)

    def get_compromised_nodes(self) -> List[Node]:
        """Get all currently compromised nodes."""
        return [n for n in self.nodes.values() if n.is_compromised]

    def get_nodes_in_subnet(self, subnet_id: int) -> List[Node]:
        """Get all nodes in a specific subnet."""
        subnet = self.subnets.get(subnet_id)
        return list(subnet.nodes.values()) if subnet else []

    @property
    def total_data_value(self) -> float:
        """Total value of all data in the network."""
        return sum(n.data_value for n in self.nodes.values())

    @property
    def compromised_data_value(self) -> float:
        """Value of data on compromised nodes."""
        return sum(n.data_value for n in self.nodes.values() if n.is_compromised)

    # ─────────────────────────────────────────────────────────
    # State Serialization (for RL observations)
    # ─────────────────────────────────────────────────────────

    def get_red_observation(self) -> np.ndarray:
        """
        Get the attacker's observation of the network.
        Partial observability — red only sees what it has discovered.
        
        Per-node features (for discovered nodes):
        [is_discovered, subnet_id, n_services, n_vulns, access_level,
         has_backdoor, is_critical, data_value]
        """
        max_nodes = len(self.nodes)
        obs = np.zeros((max_nodes, 8), dtype=np.float32)

        for i, (nid, node) in enumerate(sorted(self.nodes.items())):
            if node.is_discovered_by_red:
                obs[i] = [
                    1.0,  # discovered
                    node.subnet_id / max(1, self.config.num_subnets - 1),  # normalized
                    len(node.services) / 4.0,  # normalized
                    len(node.unpatched_vulns) / 4.0,  # normalized
                    node.access_level.value / 2.0,  # normalized
                    float(node.has_backdoor),
                    float(node.is_critical),
                    node.data_value / 10.0,  # normalized
                ]

        return np.clip(obs.flatten(), 0.0, 1.0)

    def get_blue_observation(self) -> np.ndarray:
        """
        Get the defender's observation of the network.
        Blue sees health metrics and alerts, but not attacker's true position.
        
        Per-node features:
        [alert_level, n_services_up, n_known_vulns, is_isolated,
         traffic_anomaly, is_critical, subnet_monitor_level, scan_count_norm]
        """
        max_nodes = len(self.nodes)
        obs = np.zeros((max_nodes, 8), dtype=np.float32)

        for i, (nid, node) in enumerate(sorted(self.nodes.items())):
            subnet = self.subnets[node.subnet_id]
            known_vulns = sum(1 for v in node.unpatched_vulns
                              if v.is_known_to_defender)
            services_up = sum(1 for s in node.services if s.is_running)

            obs[i] = [
                node.alert_level,
                services_up / max(1, len(node.services)),
                known_vulns / 4.0,
                float(node.is_isolated),
                min(1.0, node.scan_count / 5.0),  # anomaly proxy
                float(node.is_critical),
                subnet.traffic_monitor_level,
                min(1.0, node.scan_count / 10.0),
            ]

        return np.clip(obs.flatten(), 0.0, 1.0)

    def get_topology_data(self) -> dict:
        """Get network topology as dict (for dashboard visualization)."""
        nodes_data = []
        for nid, node in self.nodes.items():
            nodes_data.append({
                "id": nid,
                "name": node.name,
                "subnet": node.subnet_id,
                "os": node.os_type,
                "status": node.status.name,
                "access_level": node.access_level.name,
                "is_critical": node.is_critical,
                "is_honeypot": node.is_honeypot,
                "alert_level": node.alert_level,
                "services": [s.service_type.value for s in node.services],
                "data_value": node.data_value,
            })

        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "inter_subnet": data.get("inter_subnet", False),
            })

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "subnets": [
                {"id": s.subnet_id, "name": s.name, "is_dmz": s.is_dmz}
                for s in self.subnets.values()
            ],
        }

    def reset(self) -> None:
        """Reset all nodes to clean state."""
        for node in self.nodes.values():
            node.reset()
        # Re-discover all DMZ nodes for red (same as initial build)
        dmz_nodes = list(self.subnets[0].nodes.values()) if 0 in self.subnets else []
        for dmz_node in dmz_nodes:
            dmz_node.is_discovered_by_red = True
        # Reset subnet monitoring
        for subnet in self.subnets.values():
            subnet.traffic_monitor_level = 0.0
