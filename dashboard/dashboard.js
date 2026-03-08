/**
 * CyberDojo Dashboard — Frontend Logic
 * 
 * Real-time battle visualization using D3.js for network graph
 * rendering and WebSocket for live data streaming.
 */

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────

const state = {
    connected: false,
    socket: null,
    networkData: null,
    simulation: null,
    svg: null,
    eloHistory: { red: [], blue: [] },
    scoreHistory: { red: [], blue: [] },
    stepCount: 0,
    commanderMode: 'blue',  // 'blue' (defender) or 'red' (attacker)
    // Cumulative battle stats
    redDetections: 0,
    bluePatches: 0,
    blueDetections: 0,
    blueFalsePositives: 0,
};

// ─────────────────────────────────────────────────────────────
// Color Mapping
// ─────────────────────────────────────────────────────────────

const STATUS_COLORS = {
    CLEAN: '#10b981',
    COMPROMISED: '#ef4444',
    DETECTED: '#f59e0b',
    ISOLATED: '#475569',
    RESTORED: '#3b82f6',
};

const STATUS_GLOW = {
    CLEAN: 'rgba(16, 185, 129, 0.4)',
    COMPROMISED: 'rgba(239, 68, 68, 0.6)',
    DETECTED: 'rgba(245, 158, 11, 0.4)',
    ISOLATED: 'rgba(71, 85, 105, 0.3)',
    RESTORED: 'rgba(59, 130, 246, 0.4)',
};

const SUBNET_COLORS = [
    'rgba(6, 182, 212, 0.06)',
    'rgba(139, 92, 246, 0.06)',
    'rgba(245, 158, 11, 0.06)',
    'rgba(16, 185, 129, 0.06)',
    'rgba(236, 72, 153, 0.06)',
];

// ─────────────────────────────────────────────────────────────
// WebSocket Connection
// ─────────────────────────────────────────────────────────────

function initSocket() {
    const statusEl = document.getElementById('connection-status');

    try {
        state.socket = io({ transports: ['websocket', 'polling'] });

        state.socket.on('connect', () => {
            state.connected = true;
            statusEl.classList.add('connected');
            statusEl.querySelector('span:last-child').textContent = 'CONNECTED';
            addLogEntry('system', 'Connected to CyberDojo server');
        });

        state.socket.on('disconnect', () => {
            state.connected = false;
            statusEl.classList.remove('connected');
            statusEl.querySelector('span:last-child').textContent = 'DISCONNECTED';
            addLogEntry('alert', 'Lost connection to server');
        });

        state.socket.on('state_update', (data) => {
            handleStateUpdate(data);
        });

        state.socket.on('training_progress', (data) => {
            handleTrainingProgress(data);
        });

        // Chat Handlers
        state.socket.on('chat_broadcast', (msg) => {
            addChatMessage(msg.sender, msg.text);
        });

        // Commander Mode Handler
        state.socket.on('commander_mode', (data) => {
            state.commanderMode = data.mode || 'blue';
            applyCommanderMode(state.commanderMode);
        });

        // Hacker Chatroom Handler (APT Swarm)
        state.socket.on('hacker_chat', (msg) => {
            const panel = document.getElementById('hacker-chatroom-panel');
            if (panel) panel.style.display = 'block';

            const log = document.getElementById('hacker-log');
            const div = document.createElement('div');
            div.className = `hacker-msg ${msg.role}`;
            div.innerHTML = `
                <div class="hacker-role">${msg.icon} ${msg.role.toUpperCase()}</div>
                <div class="hacker-text">${msg.text}</div>
                <div class="hacker-action">↳ ${msg.action} → Node ${msg.target}</div>
            `;
            log.appendChild(div);
            log.scrollTop = log.scrollHeight;

            // Also add to battle log
            addLogEntry('red', `${msg.icon} APT-${msg.role.toUpperCase()}: ${msg.action} → Node ${msg.target}`);
        });

        // Enable chat UI
        document.getElementById('commander-input').disabled = false;
        document.getElementById('commander-send').disabled = false;
        document.getElementById('commander-input').placeholder = "Awaiting orders commander...";

        // Remove the 'offline' message
        const log = document.getElementById('commander-log');
        log.innerHTML = '';
        addChatMessage('system', 'Neural Link Established. Commander override enabled.');

    } catch (e) {
        console.warn('WebSocket not available, running in demo mode');
        statusEl.querySelector('span:last-child').textContent = 'DEMO MODE';
        initDemoMode();
    }
}

// ─────────────────────────────────────────────────────────────
// State Update Handler
// ─────────────────────────────────────────────────────────────

function handleStateUpdate(data) {
    if (!data) return;

    state.stepCount++;
    document.getElementById('step-counter').textContent = data.step || state.stepCount;

    // Update network graph
    if (data.network_state) {
        updateNetworkGraph(data.network_state);
    }

    // Update Red Team stats
    if (data.red) {
        document.getElementById('red-action').textContent =
            formatActionName(data.red.action);
        document.getElementById('red-target').textContent =
            `→ ${data.red.target_name || data.red.target || '—'}`;

        // Update MITRE display
        const redMitreEl = document.getElementById('red-mitre');
        if (data.red.mitre) {
            document.getElementById('red-mitre-id').textContent = data.red.mitre.id;
            document.getElementById('red-mitre-name').textContent = data.red.mitre.name;
            redMitreEl.style.display = 'flex';
        } else {
            redMitreEl.style.display = 'none';
        }

        // Animate attack beam on graph
        if (data.red.target && data.red.action !== 'wait') {
            animateAttackBeam(data.red.target, 'red');
            highlightNode(data.red.target, 'red');
        }

        // Update events
        const events = data.red.events || {};
        if (events.compromised_node) {
            addLogEntry('red', `🔴 Compromised ${data.red.target_name || data.red.target}`);
        }
        if (events.got_root) {
            addLogEntry('red', `🔴 ROOT access on ${data.red.target_name || data.red.target}`);
        }
        if (events.exfiltrated_data) {
            addLogEntry('red', `🔴 Data exfiltrated (${events.exfiltrated_data.toFixed(1)})`);
        }
        if (events.lateral_move) {
            addLogEntry('red', `🔴 Lateral movement detected`);
        }
        if (events.got_detected) {
            state.redDetections++;
            addLogEntry('alert', `⚠️ Red team detected at ${data.red.target_name || data.red.target}`);
            document.getElementById('red-detections').textContent = state.redDetections;
        }
    }

    // Update Blue Team stats
    if (data.blue) {
        document.getElementById('blue-action').textContent =
            formatActionName(data.blue.action);
        document.getElementById('blue-target').textContent =
            `→ ${data.blue.target_name || data.blue.target || '—'}`;

        // Update MITRE display
        const blueMitreEl = document.getElementById('blue-mitre');
        if (data.blue.mitre) {
            document.getElementById('blue-mitre-id').textContent = data.blue.mitre.id;
            document.getElementById('blue-mitre-name').textContent = data.blue.mitre.name;
            blueMitreEl.style.display = 'flex';
        } else {
            blueMitreEl.style.display = 'none';
        }

        if (data.blue.target) {
            animateAttackBeam(data.blue.target, 'blue');
            highlightNode(data.blue.target, 'blue');
        }

        const events = data.blue.events || {};
        if (events.detected_attacker) {
            state.blueDetections++;
            addLogEntry('blue', `🔵 Attacker detected at ${data.blue.target_name || data.blue.target}`);
            document.getElementById('blue-detections').textContent = state.blueDetections;
        }
        if (events.contained_threat) {
            addLogEntry('success', `🔵 Threat contained at ${data.blue.target_name || data.blue.target}`);
        }
        if (events.patched_vuln) {
            state.bluePatches += events.patched_vuln;
            addLogEntry('blue', `🔵 Patched ${events.patched_vuln} vulnerabilities`);
            document.getElementById('blue-patches').textContent = state.bluePatches;
        }
        if (events.false_positive) {
            state.blueFalsePositives++;
            document.getElementById('blue-false-pos').textContent = state.blueFalsePositives;
        }
        if (events.honeypot_triggered) {
            addLogEntry('success', `🍯 Honeypot triggered at ${data.blue.target_name || data.blue.target}!`);
        }
    }

    // Update aggregate stats from network state
    if (data.network_state) {
        updateAggregateStats(data.network_state);
    }
}

function handleTrainingProgress(data) {
    if (!data) return;

    if (data.red_elo) {
        document.getElementById('red-elo').textContent = Math.round(data.red_elo);
        state.eloHistory.red.push(data.red_elo);
    }
    if (data.blue_elo) {
        document.getElementById('blue-elo').textContent = Math.round(data.blue_elo);
        state.eloHistory.blue.push(data.blue_elo);
    }

    if (data.battle) {
        state.scoreHistory.red.push(data.battle.red_score);
        state.scoreHistory.blue.push(data.battle.blue_score);
        drawCharts();

        // Show winner banner
        showWinnerBanner(data.battle.winner, data.battle.red_score, data.battle.blue_score);
    }
}

// ─────────────────────────────────────────────────────────────
// Winner Banner
// ─────────────────────────────────────────────────────────────

function showWinnerBanner(winner, redScore, blueScore) {
    const overlay = document.getElementById('winner-overlay');
    const icon = document.getElementById('winner-icon');
    const text = document.getElementById('winner-text');
    const score = document.getElementById('winner-score');

    overlay.className = 'winner-overlay';
    if (winner === 'red') {
        icon.textContent = '🔴';
        text.textContent = '🔴 RED TEAM WINS!';
        overlay.classList.add('winner-red');
    } else if (winner === 'blue') {
        icon.textContent = '🔵';
        text.textContent = '🔵 BLUE TEAM WINS!';
        overlay.classList.add('winner-blue');
    } else {
        icon.textContent = '🤝';
        text.textContent = 'DRAW';
    }
    score.textContent = `Score — Red: ${redScore.toFixed(0)} | Blue: ${blueScore.toFixed(0)}`;

    overlay.style.display = 'flex';

    // Reset per-battle stats for next round
    state.redDetections = 0;
    state.blueDetections = 0;
    state.bluePatches = 0;
    state.blueFalsePositives = 0;
    state.stepCount = 0;

    // Auto-hide after 3 seconds
    setTimeout(() => { overlay.style.display = 'none'; }, 3000);
}

// ─────────────────────────────────────────────────────────────
// Attack Path Beam Animation
// ─────────────────────────────────────────────────────────────

function animateAttackBeam(targetNodeId, team) {
    if (!state.svg || !state.simulation) return;

    const nodes = state.simulation.nodes();
    const target = nodes.find(n => n.id === targetNodeId);
    if (!target || !target.x) return;

    // Find a source node (entry point or last compromised for red, random for blue)
    let source;
    if (team === 'red') {
        const compromised = nodes.filter(n => n.status === 'COMPROMISED' && n.id !== targetNodeId);
        source = compromised.length > 0
            ? compromised[compromised.length - 1]
            : nodes[0];
    } else {
        source = nodes.find(n => n.status !== 'ISOLATED') || nodes[0];
    }
    if (!source || !source.x) return;

    const color = team === 'red' ? '#ef4444' : '#3b82f6';
    const beamLayer = state.svg.select('.link-layer');

    const beam = beamLayer.append('line')
        .attr('x1', source.x)
        .attr('y1', source.y)
        .attr('x2', source.x)
        .attr('y2', source.y)
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('stroke-linecap', 'round')
        .attr('opacity', 0.9)
        .style('filter', `drop-shadow(0 0 4px ${color})`);

    // Animate beam extending to target (duration extended to 4000ms so it stays visible during LLM reasoning waits)
    beam.transition().duration(600)
        .attr('x2', target.x)
        .attr('y2', target.y)
        .transition().duration(4000)
        .attr('opacity', 0)
        .remove();
}

// ─────────────────────────────────────────────────────────────
// Cyber Commander Chat
// ─────────────────────────────────────────────────────────────

function addChatMessage(sender, text) {
    const log = document.getElementById('commander-log');
    const entry = document.createElement('div');
    entry.className = `chat-msg ${sender}`;
    entry.textContent = text;

    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function sendCommand() {
    const input = document.getElementById('commander-input');
    const msg = input.value.trim();
    if (!msg || !state.connected) return;

    // We don't append it here because the server will broadcast it back to us
    state.socket.emit('commander_chat', { message: msg });

    input.value = '';
    input.focus();
}

// Event Listeners for Chat
document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('commander-send');
    const input = document.getElementById('commander-input');

    if (sendBtn && input) {
        sendBtn.addEventListener('click', sendCommand);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendCommand();
        });
    }
});

// ─────────────────────────────────────────────────────────────
// Network Graph (D3.js)
// ─────────────────────────────────────────────────────────────

function initNetworkGraph() {
    const container = document.getElementById('network-graph');
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    state.svg = d3.select('#network-svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

    // Add defs for glows
    const defs = state.svg.append('defs');

    // Red glow filter
    const redGlow = defs.append('filter').attr('id', 'glow-red');
    redGlow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
    redGlow.append('feFlood').attr('flood-color', '#ef4444').attr('flood-opacity', '0.6');
    redGlow.append('feComposite').attr('in2', 'blur').attr('operator', 'in');
    const redMerge = redGlow.append('feMerge');
    redMerge.append('feMergeNode');
    redMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Blue glow filter
    const blueGlow = defs.append('filter').attr('id', 'glow-blue');
    blueGlow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
    blueGlow.append('feFlood').attr('flood-color', '#3b82f6').attr('flood-opacity', '0.6');
    blueGlow.append('feComposite').attr('in2', 'blur').attr('operator', 'in');
    const blueMerge = blueGlow.append('feMerge');
    blueMerge.append('feMergeNode');
    blueMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Cyan glow filter
    const cyanGlow = defs.append('filter').attr('id', 'glow-cyan');
    cyanGlow.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
    cyanGlow.append('feFlood').attr('flood-color', '#06b6d4').attr('flood-opacity', '0.4');
    cyanGlow.append('feComposite').attr('in2', 'blur').attr('operator', 'in');
    const cyanMerge = cyanGlow.append('feMerge');
    cyanMerge.append('feMergeNode');
    cyanMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Layer groups
    state.svg.append('g').attr('class', 'subnet-layer');
    state.svg.append('g').attr('class', 'link-layer');
    state.svg.append('g').attr('class', 'node-layer');
    state.svg.append('g').attr('class', 'label-layer');
}

function updateNetworkGraph(networkData) {
    if (!state.svg) return;

    const container = document.getElementById('network-graph');
    const width = container.clientWidth;
    const height = container.clientHeight || 400;

    const nodes = networkData.nodes || [];
    const edges = networkData.edges || [];

    // Update or create simulation
    if (!state.simulation || state.networkData === null) {
        state.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges).id(d => d.id).distance(60))
            .force('charge', d3.forceManyBody().strength(-200)) // Increased repulsion
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(55)) // Increased from 25 to 55 to prevent label overlap
            .force('x', d3.forceX(d => {
                const subnetSpacing = width / (Math.max(...nodes.map(n => n.subnet)) + 2);
                return (d.subnet + 1) * subnetSpacing;
            }).strength(0.3))
            .force('y', d3.forceY(height / 2).strength(0.1));

        state.simulation.on('tick', () => renderGraph(nodes, edges));
        state.networkData = networkData;
    } else {
        // Update node properties without resetting simulation
        nodes.forEach(newNode => {
            const existing = state.simulation.nodes().find(n => n.id === newNode.id);
            if (existing) {
                existing.status = newNode.status;
                existing.access_level = newNode.access_level;
                existing.is_critical = newNode.is_critical;
                existing.is_honeypot = newNode.is_honeypot;
                existing.alert_level = newNode.alert_level;
            }
        });
        renderGraph(state.simulation.nodes(), edges);
    }
}

function renderGraph(nodes, edges) {
    if (!state.svg) return;

    // ── Links ──
    const linkLayer = state.svg.select('.link-layer');
    const links = linkLayer.selectAll('.link-line')
        .data(edges, d => `${d.source.id || d.source}-${d.target.id || d.target}`);

    links.enter()
        .append('line')
        .attr('class', d => `link-line ${d.inter_subnet ? 'inter-subnet' : ''}`)
        .merge(links)
        .attr('x1', d => d.source.x || 0)
        .attr('y1', d => d.source.y || 0)
        .attr('x2', d => d.target.x || 0)
        .attr('y2', d => d.target.y || 0);

    links.exit().remove();

    // ── Nodes ──
    const nodeLayer = state.svg.select('.node-layer');
    const nodeGroups = nodeLayer.selectAll('.node-group')
        .data(nodes, d => d.id);

    const enterGroups = nodeGroups.enter()
        .append('g')
        .attr('class', 'node-group')
        .call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded))
        .on('mouseover', function (event, d) { showNodeTooltip(event, d); })
        .on('mousemove', function (event, d) { moveNodeTooltip(event); })
        .on('mouseout', function (event, d) { hideNodeTooltip(); })
        .on('click', function (event, d) { showNodeActionMenu(event, d); });

    // Outer ring (alert indicator)
    enterGroups.append('circle')
        .attr('class', 'node-alert-ring')
        .attr('r', 14)
        .attr('fill', 'none')
        .attr('stroke-width', 1.5);

    // Main circle
    enterGroups.append('circle')
        .attr('class', 'node-circle')
        .attr('r', 10);

    // Inner icon indicator
    enterGroups.append('text')
        .attr('class', 'node-icon')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .style('font-size', '8px')
        .style('pointer-events', 'none');

    // Merge
    const allGroups = enterGroups.merge(nodeGroups);

    allGroups.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`);

    // Update node colors and pulsing animations
    allGroups.select('.node-circle')
        .attr('fill', d => STATUS_COLORS[d.status] || STATUS_COLORS.CLEAN)
        .attr('stroke', d => {
            if (d.is_critical) return '#8b5cf6';
            if (d.is_honeypot) return '#06b6d4';
            return STATUS_COLORS[d.status] || '#475569';
        })
        .classed('pulse-attack', d => d.status === 'COMPROMISED')
        .classed('pulse-isolated', d => d.status === 'ISOLATED')
        .attr('filter', d => {
            if (d.status === 'COMPROMISED') return 'url(#glow-red)';
            return null;
        });

    // Alert ring
    allGroups.select('.node-alert-ring')
        .attr('stroke', d => {
            if (d.alert_level > 0.7) return 'rgba(239, 68, 68, 0.8)';
            if (d.alert_level > 0.3) return 'rgba(245, 158, 11, 0.6)';
            return 'transparent';
        })
        .attr('stroke-dasharray', d => d.alert_level > 0.5 ? 'none' : '3 3');

    // Node icons
    allGroups.select('.node-icon')
        .text(d => {
            if (d.is_critical) return '⭐';
            if (d.is_honeypot) return '🍯';
            if (d.access_level === 'ROOT') return '💀';
            if (d.status === 'COMPROMISED') return '🔓';
            if (d.status === 'ISOLATED') return '🔒';
            return '';
        });

    nodeGroups.exit().remove();

    // ── Labels ──
    const labelLayer = state.svg.select('.label-layer');
    const labels = labelLayer.selectAll('.node-label')
        .data(nodes, d => d.id);

    labels.enter()
        .append('text')
        .attr('class', 'node-label')
        .attr('dy', 22)
        .merge(labels)
        .attr('x', d => d.x || 0)
        .attr('y', d => d.y || 0)
        .text(d => d.name);

    labels.exit().remove();
}

function highlightNode(nodeId, team) {
    const nodeGroup = state.svg?.select('.node-layer')
        .selectAll('.node-group')
        .filter(d => d.id === nodeId);

    if (nodeGroup && !nodeGroup.empty()) {
        const circle = nodeGroup.select('.node-circle');
        const originalR = 10;

        circle
            .transition().duration(150)
            .attr('r', 16)
            .attr('filter', team === 'red' ? 'url(#glow-red)' : 'url(#glow-blue)')
            .transition().duration(300)
            .attr('r', originalR);
    }
}

// Drag handlers
function dragStarted(event, d) {
    if (!event.active) state.simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) state.simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// ─────────────────────────────────────────────────────────────
// Stats Updates
// ─────────────────────────────────────────────────────────────

function updateAggregateStats(networkData) {
    const nodes = networkData.nodes || [];

    // Red stats
    const compromised = nodes.filter(n => n.status === 'COMPROMISED').length;
    const backdoors = nodes.filter(n =>
        n.status === 'COMPROMISED' && n.access_level === 'ROOT'
    ).length;
    document.getElementById('red-compromised').textContent = compromised;
    document.getElementById('red-backdoors').textContent = backdoors;

    // Calculate data stolen (sum of compromised node data values)
    const dataStolen = nodes
        .filter(n => n.status === 'COMPROMISED')
        .reduce((sum, n) => sum + (n.data_value || 0), 0);
    document.getElementById('red-data-stolen').textContent = dataStolen.toFixed(1);

    // Blue stats
    const isolated = nodes.filter(n => n.status === 'ISOLATED').length;
    document.getElementById('blue-isolations').textContent = isolated;
}

// ─────────────────────────────────────────────────────────────
// Event Log
// ─────────────────────────────────────────────────────────────

function addLogEntry(type, message) {
    const log = document.getElementById('event-log');
    const now = new Date();
    const time = `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-msg">${message}</span>
    `;

    log.appendChild(entry);

    // Keep last 100 entries
    while (log.children.length > 100) {
        log.removeChild(log.firstChild);
    }

    // Auto-scroll
    log.scrollTop = log.scrollHeight;
}

// ─────────────────────────────────────────────────────────────
// Charts (Simple Canvas)
// ─────────────────────────────────────────────────────────────

function drawCharts() {
    drawMiniChart('elo-chart', state.eloHistory, 'ELO RATING');
    drawMiniChart('score-chart', state.scoreHistory, 'BATTLE SCORE');
}

function drawMiniChart(canvasId, data, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const w = canvas.width;
    const h = canvas.height;
    const padding = { top: 20, right: 10, bottom: 10, left: 40 };

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Title
    ctx.font = '10px "JetBrains Mono"';
    ctx.fillStyle = '#475569';
    ctx.fillText(title, padding.left, 12);

    const redData = data.red || [];
    const blueData = data.blue || [];

    if (redData.length < 2) return;

    const allVals = [...redData, ...blueData];
    const minVal = Math.min(...allVals) - 10;
    const maxVal = Math.max(...allVals) + 10;
    const range = maxVal - minVal || 1;

    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    const toX = (i) => padding.left + (i / Math.max(1, redData.length - 1)) * plotW;
    const toY = (v) => padding.top + plotH - ((v - minVal) / range) * plotH;

    // Grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (plotH / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
    }

    // Draw lines
    function drawLine(arr, color) {
        if (arr.length < 2) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.moveTo(toX(0), toY(arr[0]));
        for (let i = 1; i < arr.length; i++) {
            ctx.lineTo(toX(i), toY(arr[i]));
        }
        ctx.stroke();

        // Glow
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    drawLine(redData, '#ef4444');
    drawLine(blueData, '#3b82f6');

    // Latest values
    if (redData.length > 0) {
        ctx.font = '10px "JetBrains Mono"';
        ctx.fillStyle = '#ef4444';
        ctx.fillText(Math.round(redData[redData.length - 1]), w - 60, padding.top + 12);
        ctx.fillStyle = '#3b82f6';
        ctx.fillText(Math.round(blueData[blueData.length - 1]), w - 60, padding.top + 24);
    }
}

// ─────────────────────────────────────────────────────────────
// Demo Mode (when no server is connected)
// ─────────────────────────────────────────────────────────────

function initDemoMode() {
    addLogEntry('system', 'Running in demo mode — generating sample data');

    // Generate a demo network
    const demoNetwork = generateDemoNetwork();
    updateNetworkGraph(demoNetwork);

    // Simulate some activity
    let step = 0;
    const actions = [
        'scan_network', 'scan_vulnerability', 'exploit', 'lateral_move',
        'install_backdoor', 'exfiltrate_data', 'phish_user',
    ];
    const defenses = [
        'monitor_traffic', 'analyze_alert', 'isolate_node',
        'patch_vulnerability', 'deploy_honeypot', 'forensic_analysis',
    ];

    setInterval(() => {
        step++;
        const nodes = demoNetwork.nodes;
        const targetNode = nodes[Math.floor(Math.random() * nodes.length)];
        const defTarget = nodes[Math.floor(Math.random() * nodes.length)];

        // Randomly change node statuses
        if (Math.random() < 0.2) {
            const statuses = ['CLEAN', 'COMPROMISED', 'DETECTED', 'ISOLATED'];
            targetNode.status = statuses[Math.floor(Math.random() * statuses.length)];
            targetNode.alert_level = Math.random();
        }

        const redAction = actions[Math.floor(Math.random() * actions.length)];
        const blueAction = defenses[Math.floor(Math.random() * defenses.length)];

        handleStateUpdate({
            step: step,
            red: {
                action: redAction,
                target: targetNode.id,
                events: Math.random() < 0.3 ? { compromised_node: 1 } : {},
            },
            blue: {
                action: blueAction,
                target: defTarget.id,
                events: Math.random() < 0.2 ? { detected_attacker: 1 } : {},
            },
            network_state: demoNetwork,
        });

        // Demo Elo updates
        state.eloHistory.red.push(1000 + Math.sin(step / 10) * 50 + step * 0.5);
        state.eloHistory.blue.push(1000 + Math.cos(step / 10) * 50 + step * 0.3);
        state.scoreHistory.red.push(Math.random() * 50 + 20);
        state.scoreHistory.blue.push(Math.random() * 50 + 25);

        document.getElementById('red-elo').textContent =
            Math.round(state.eloHistory.red[state.eloHistory.red.length - 1]);
        document.getElementById('blue-elo').textContent =
            Math.round(state.eloHistory.blue[state.eloHistory.blue.length - 1]);

        drawCharts();
    }, 1500);
}

function generateDemoNetwork() {
    const nodes = [
        { id: 'node-0', name: 'web-server', subnet: 0, os: 'linux', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['http', 'https'], data_value: 2.5 },
        { id: 'node-1', name: 'api-server', subnet: 0, os: 'linux', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['http', 'ssh'], data_value: 3.0 },
        { id: 'node-2', name: 'proxy-server', subnet: 0, os: 'linux', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['http'], data_value: 1.5 },
        { id: 'node-3', name: 'dc-primary', subnet: 1, os: 'windows', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['rdp', 'smb'], data_value: 4.0 },
        { id: 'node-4', name: 'workstation', subnet: 1, os: 'windows', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['rdp'], data_value: 2.0 },
        { id: 'node-5', name: 'dev-machine', subnet: 1, os: 'linux', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['ssh', 'http'], data_value: 3.5 },
        { id: 'node-6', name: 'db-server', subnet: 2, os: 'linux', status: 'CLEAN', access_level: 'NONE', is_critical: false, is_honeypot: false, alert_level: 0, services: ['mysql', 'ssh'], data_value: 5.0 },
        { id: 'node-7', name: 'crown-jewels-db', subnet: 2, os: 'linux', status: 'CLEAN', access_level: 'NONE', is_critical: true, is_honeypot: false, alert_level: 0, services: ['mysql'], data_value: 10.0 },
    ];

    const edges = [
        { source: 'node-0', target: 'node-1', inter_subnet: false },
        { source: 'node-0', target: 'node-2', inter_subnet: false },
        { source: 'node-1', target: 'node-2', inter_subnet: false },
        { source: 'node-3', target: 'node-4', inter_subnet: false },
        { source: 'node-3', target: 'node-5', inter_subnet: false },
        { source: 'node-4', target: 'node-5', inter_subnet: false },
        { source: 'node-6', target: 'node-7', inter_subnet: false },
        { source: 'node-2', target: 'node-3', inter_subnet: true },
        { source: 'node-5', target: 'node-6', inter_subnet: true },
    ];

    return {
        nodes, edges, subnets: [
            { id: 0, name: 'DMZ', is_dmz: true },
            { id: 1, name: 'Corporate-LAN', is_dmz: false },
            { id: 2, name: 'Server-Farm', is_dmz: false },
        ]
    };
}

// ─────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────

function formatActionName(action) {
    if (!action) return '—';
    return action.replace(/_/g, ' ').toUpperCase();
}

// ─────────────────────────────────────────────────────────────
// Interactive Node Tooltip
// ─────────────────────────────────────────────────────────────

function showNodeTooltip(event, d) {
    const tooltip = document.getElementById('node-tooltip');
    document.getElementById('tooltip-name').textContent = d.name || d.id;

    const statusEl = document.getElementById('tooltip-status');
    const status = (d.status || 'CLEAN').toUpperCase();
    statusEl.textContent = status;
    statusEl.className = 'tooltip-status';
    if (status === 'COMPROMISED') statusEl.classList.add('compromised');
    else if (status === 'ISOLATED') statusEl.classList.add('isolated');
    else if (status === 'DETECTED') statusEl.classList.add('detected');

    document.getElementById('tooltip-os').textContent = d.os || 'linux';
    document.getElementById('tooltip-services').textContent =
        (d.services && d.services.length > 0) ? d.services.join(', ') : 'none';
    document.getElementById('tooltip-alert').textContent =
        (d.alert_level != null) ? d.alert_level.toFixed(2) : '0.00';
    document.getElementById('tooltip-access').textContent = d.access_level || 'none';
    document.getElementById('tooltip-value').textContent =
        (d.data_value != null) ? d.data_value.toFixed(1) : '0.0';
    document.getElementById('tooltip-subnet').textContent =
        (d.subnet != null) ? `Subnet ${d.subnet}` : '—';

    tooltip.style.display = 'block';
    moveNodeTooltip(event);
}

function moveNodeTooltip(event) {
    const tooltip = document.getElementById('node-tooltip');
    const x = event.clientX + 16;
    const y = event.clientY - 10;
    // Keep tooltip on screen
    const maxX = window.innerWidth - 310;
    const maxY = window.innerHeight - tooltip.offsetHeight - 10;
    tooltip.style.left = Math.min(x, maxX) + 'px';
    tooltip.style.top = Math.min(y, maxY) + 'px';
}

function hideNodeTooltip() {
    document.getElementById('node-tooltip').style.display = 'none';
}

// ─────────────────────────────────────────────────────────────
// Interactive Node Action Menu
// ─────────────────────────────────────────────────────────────

let selectedNodeForAction = null;

function showNodeActionMenu(event, d) {
    event.stopPropagation();
    hideNodeTooltip();

    selectedNodeForAction = d;
    const menu = document.getElementById('node-action-menu');
    const titlePrefix = state.commanderMode === 'red' ? '💠' : '⚡';
    document.getElementById('action-menu-title').textContent =
        `${titlePrefix} ${d.name || d.id} (Node ${d.id.replace('node-', '')})`;

    // Toggle blue/red action grids based on commander mode
    const blueGrid = document.getElementById('blue-actions');
    const redGrid = document.getElementById('red-actions');
    const pentestBtn = document.getElementById('pentest-btn');

    if (state.commanderMode === 'red') {
        blueGrid.style.display = 'none';
        redGrid.style.display = 'grid';
        pentestBtn.style.display = 'none';
        menu.classList.add('red-mode');
        menu.classList.remove('blue-mode');
    } else {
        blueGrid.style.display = 'grid';
        redGrid.style.display = 'none';
        pentestBtn.style.display = 'flex';
        menu.classList.add('blue-mode');
        menu.classList.remove('red-mode');
    }

    // Position the menu near the click
    const x = event.clientX - 140;
    const y = event.clientY + 10;
    const maxX = window.innerWidth - 300;
    const maxY = window.innerHeight - 450;
    menu.style.left = Math.max(10, Math.min(x, maxX)) + 'px';
    menu.style.top = Math.max(10, Math.min(y, maxY)) + 'px';
    menu.style.display = 'block';

    // Highlight the selected node
    d3.selectAll('.node-group').classed('selected', false);
    d3.selectAll('.node-group')
        .filter(nd => nd.id === d.id)
        .classed('selected', true);
}

function hideNodeActionMenu() {
    document.getElementById('node-action-menu').style.display = 'none';
    d3.selectAll('.node-group').classed('selected', false);
    selectedNodeForAction = null;
}

function executeNodeAction(actionName) {
    if (!selectedNodeForAction) return;

    const nodeName = selectedNodeForAction.name || selectedNodeForAction.id;
    const humanReadableAction = actionName.replace(/_/g, ' ');
    const command = `${humanReadableAction} ${nodeName}`;

    // Handle pen_test specially — it's a preview, not a real command
    if (actionName === 'pen_test') {
        const penTestCommand = `pen test ${nodeName}`;
        if (state.socket) {
            state.socket.emit('commander_chat', { message: penTestCommand });
            addLogEntry('system', `🔬 Pen Test: Previewing exploits on ${nodeName}`);
        }
        hideNodeActionMenu();
        return;
    }

    // Send via the existing Commander Chat pipeline
    if (state.socket) {
        state.socket.emit('commander_chat', { message: command });
        const logType = state.commanderMode === 'red' ? 'red' : 'blue';
        const logIcon = state.commanderMode === 'red' ? '💠' : '🎯';
        addLogEntry(logType, `${logIcon} Commander: ${humanReadableAction} → ${nodeName}`);
    }

    hideNodeActionMenu();
}

function applyCommanderMode(mode) {
    state.commanderMode = mode;
    const input = document.getElementById('commander-input');
    const chatTitle = document.querySelector('.cyber-commander-panel .panel-header h2');
    const chatSubtitle = document.querySelector('.cyber-commander-panel .panel-header span');
    const panelHeader = document.querySelector('.cyber-commander-panel .panel-header');

    if (mode === 'red') {
        if (input) input.placeholder = 'Give hacking orders...';
        if (chatTitle) {
            chatTitle.textContent = '💀 RED COMMANDER';
            chatTitle.style.color = 'var(--red-glow)';
        }
        if (chatSubtitle) {
            chatSubtitle.textContent = 'ATTACK MODE';
            chatSubtitle.style.color = 'var(--red-glow)';
        }
        if (panelHeader) {
            panelHeader.style.background = 'rgba(239, 68, 68, 0.1)';
            panelHeader.style.borderBottomColor = 'rgba(239, 68, 68, 0.2)';
        }
        addChatMessage('system', '⚔️ RED TEAM MODE: You are the attacker. Click nodes to hack them!');
        addLogEntry('system', '⚔️ RED TEAM MODE: You are the attacker. Click nodes to hack them!');
    } else {
        if (input) input.placeholder = 'Awaiting orders commander...';
        if (chatTitle) {
            chatTitle.textContent = '🧠 CYBER COMMANDER';
            chatTitle.style.color = 'var(--green-primary)';
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initNetworkGraph();
    initSocket();

    // Clear log button
    document.getElementById('clear-log').addEventListener('click', () => {
        const log = document.getElementById('event-log');
        log.innerHTML = '';
        addLogEntry('system', 'Log cleared');
    });

    // Resize handler
    window.addEventListener('resize', () => {
        const container = document.getElementById('network-graph');
        if (state.svg) {
            state.svg.attr('viewBox',
                `0 0 ${container.clientWidth} ${container.clientHeight || 400}`);
        }
    });

    // Action menu button clicks
    document.querySelectorAll('.action-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const action = btn.getAttribute('data-action');
            executeNodeAction(action);
        });
    });

    // Close action menu
    document.getElementById('action-menu-close').addEventListener('click', (e) => {
        e.stopPropagation();
        hideNodeActionMenu();
    });

    // Click anywhere else to close the menu
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('node-action-menu');
        if (menu.style.display === 'block' && !menu.contains(e.target)) {
            hideNodeActionMenu();
        }
    });
});
