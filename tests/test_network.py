"""Tests for CyberDojo network topology simulation."""

import pytest
import numpy as np

from cyberdojo.network import (
    Network, Node, Subnet, Service, Vulnerability,
    AccessLevel, NodeStatus, ServiceType, VulnSeverity,
)
from cyberdojo.config import NetworkConfig


class TestVulnerability:
    def test_exploit_probability_unpatched(self):
        vuln = Vulnerability(
            vuln_id="CVE-TEST-001", name="Test Vuln",
            severity=VulnSeverity.HIGH, exploit_difficulty=0.3,
        )
        assert 0.0 < vuln.exploit_probability <= 1.0

    def test_exploit_probability_patched(self):
        vuln = Vulnerability(
            vuln_id="CVE-TEST-002", name="Patched Vuln",
            severity=VulnSeverity.CRITICAL, exploit_difficulty=0.1,
            is_patched=True,
        )
        assert vuln.exploit_probability == 0.0

    def test_critical_severity_easier(self):
        low = Vulnerability(vuln_id="LOW", name="Low", severity=VulnSeverity.LOW, exploit_difficulty=0.5)
        high = Vulnerability(vuln_id="HIGH", name="High", severity=VulnSeverity.CRITICAL, exploit_difficulty=0.5)
        assert high.exploit_probability > low.exploit_probability


class TestNode:
    def test_node_creation(self):
        node = Node(node_id="n1", name="test", subnet_id=0, os_type="linux")
        assert node.status == NodeStatus.CLEAN
        assert node.access_level == AccessLevel.NONE
        assert not node.is_compromised

    def test_node_reset(self):
        node = Node(node_id="n1", name="test", subnet_id=0, os_type="linux")
        node.status = NodeStatus.COMPROMISED
        node.access_level = AccessLevel.ROOT
        node.has_backdoor = True
        node.is_discovered_by_red = True

        node.reset()
        assert node.status == NodeStatus.CLEAN
        assert node.access_level == AccessLevel.NONE
        assert not node.has_backdoor
        assert not node.is_discovered_by_red

    def test_unpatched_vulns(self):
        service = Service(service_type=ServiceType.HTTP, port=80, version="1.0")
        service.vulnerabilities = [
            Vulnerability("V1", "Vuln1", VulnSeverity.HIGH, 0.3),
            Vulnerability("V2", "Vuln2", VulnSeverity.MEDIUM, 0.5, is_patched=True),
        ]
        node = Node(node_id="n1", name="test", subnet_id=0, os_type="linux",
                     services=[service])
        assert len(node.unpatched_vulns) == 1
        assert node.unpatched_vulns[0].vuln_id == "V1"


class TestNetwork:
    def test_small_network_creation(self):
        config = NetworkConfig.small()
        net = Network(config=config, seed=42)

        assert len(net.subnets) == 3
        assert len(net.nodes) == sum(config.nodes_per_subnet)
        assert net.entry_point_id is not None
        assert net.critical_target_id is not None

    def test_medium_network_creation(self):
        config = NetworkConfig.medium()
        net = Network(config=config, seed=42)
        assert len(net.subnets) == 5
        assert len(net.nodes) == sum(config.nodes_per_subnet)

    def test_network_connectivity(self):
        net = Network(seed=42)
        # All intra-subnet nodes should be connected
        for subnet in net.subnets.values():
            node_ids = list(subnet.nodes.keys())
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    assert net.graph.has_edge(node_ids[i], node_ids[j])

    def test_entry_point_discovered(self):
        net = Network(seed=42)
        entry = net.nodes[net.entry_point_id]
        assert entry.is_discovered_by_red

    def test_critical_target_high_value(self):
        net = Network(seed=42)
        if net.critical_target_id:
            target = net.nodes[net.critical_target_id]
            assert target.is_critical
            assert target.data_value == 10.0

    def test_get_adjacent_nodes(self):
        net = Network(seed=42)
        entry = net.entry_point_id
        adjacent = net.get_adjacent_nodes(entry)
        assert len(adjacent) > 0

    def test_red_observation_shape(self):
        net = Network(seed=42)
        obs = net.get_red_observation()
        assert obs.shape == (len(net.nodes) * 8,)
        assert obs.dtype == np.float32

    def test_blue_observation_shape(self):
        net = Network(seed=42)
        obs = net.get_blue_observation()
        assert obs.shape == (len(net.nodes) * 8,)
        assert obs.dtype == np.float32

    def test_topology_data_for_dashboard(self):
        net = Network(seed=42)
        data = net.get_topology_data()
        assert "nodes" in data
        assert "edges" in data
        assert "subnets" in data
        assert len(data["nodes"]) == len(net.nodes)

    def test_network_reset(self):
        net = Network(seed=42)
        # Compromise a node
        node = list(net.nodes.values())[1]
        node.status = NodeStatus.COMPROMISED
        node.access_level = AccessLevel.ROOT
        assert len(net.get_compromised_nodes()) > 0

        net.reset()
        assert len(net.get_compromised_nodes()) == 0

    def test_can_reach_isolated(self):
        net = Network(seed=42)
        nodes = list(net.nodes.values())
        src, tgt = nodes[0], nodes[1]
        
        # Should be reachable
        if net.can_reach(src.node_id, tgt.node_id):
            tgt.status = NodeStatus.ISOLATED
            assert not net.can_reach(src.node_id, tgt.node_id)
