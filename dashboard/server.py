"""
CyberDojo Dashboard Server

WebSocket-based server that streams real-time battle data
to the web-based visualization dashboard.
"""

import os
import json
import threading
import logging
from typing import Optional, Dict
from flask import Flask, send_from_directory
from flask_socketio import SocketIO

from cyberdojo.config import DashboardConfig

logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, static_folder=".", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global state
_latest_state: Dict = {}
_battle_history: list = []
_commander_queue: list = []
_commander_cv = threading.Condition()
_commander_mode: str = "blue"  # "blue" or "red" — which team the human controls


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard."""
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory(os.path.dirname(__file__), "styles.css")


@app.route("/dashboard.js")
def dashboard_js():
    return send_from_directory(os.path.dirname(__file__), "dashboard.js")


@app.route("/api/state")
def get_state():
    """Get current battle state."""
    return json.dumps(_latest_state)


@app.route("/api/history")
def get_history():
    """Get battle history."""
    return json.dumps(_battle_history[-100:])


@app.route("/api/mode")
def get_mode():
    """Get the current commander mode (red or blue)."""
    return json.dumps({"commander_mode": _commander_mode})


# ─────────────────────────────────────────────────────────────
# WebSocket Events
# ─────────────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    logger.info("Dashboard client connected")
    if _latest_state:
        socketio.emit("state_update", _latest_state)
    # Tell the dashboard which team the human controls
    socketio.emit("commander_mode", {"mode": _commander_mode})


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Dashboard client disconnected")


@socketio.on("request_state")
def handle_request_state():
    socketio.emit("state_update", _latest_state)


@socketio.on("commander_chat")
def handle_commander_chat(data):
    """Receive a message from the Human Commander in the UI."""
    msg = data.get("message", "").strip()
    if msg:
        logger.info(f"Human Commander Input: {msg}")
        with _commander_cv:
            _commander_queue.append(msg)
            _commander_cv.notify_all()
        
        # Broadcast the message back to update the chat log for all viewers
        socketio.emit("chat_broadcast", {"sender": "human", "text": msg})


# ─────────────────────────────────────────────────────────────
# Dashboard Bridge (used by trainer)
# ─────────────────────────────────────────────────────────────

class DashboardBridge:
    """Bridge between the trainer and the dashboard server."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._server_thread: Optional[threading.Thread] = None

    def set_commander_mode(self, mode: str) -> None:
        """Set commander mode to 'red' or 'blue'."""
        global _commander_mode
        _commander_mode = mode
        logger.info(f"Commander mode set to: {mode}")

    def push_update(self, data: Dict) -> None:
        """Push a state update to all connected dashboard clients."""
        global _latest_state
        _latest_state = data

        if data.get("step_data"):
            _battle_history.append(data["step_data"])

        socketio.emit("state_update", data)

    def push_training_progress(self, data: Dict) -> None:
        """Push training progress update."""
        socketio.emit("training_progress", data)

    def push_chat_message(self, text: str, sender: str = "system") -> None:
        """Push a message to the Cyber Commander chat log."""
        socketio.emit("chat_broadcast", {"sender": sender, "text": text})

    def wait_for_human_command(self, timeout: float = 30.0) -> Optional[str]:
        """Block and wait for the human to type a command in the UI."""
        with _commander_cv:
            if not _commander_queue:
                _commander_cv.wait(timeout=timeout)
            
            if _commander_queue:
                return _commander_queue.pop(0)
            return None

    def start_async(self) -> None:
        """Start the dashboard server in a background thread."""
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True
        )
        self._server_thread.start()

    def _run_server(self) -> None:
        """Run the Flask-SocketIO server."""
        socketio.run(
            app,
            host=self.config.host,
            port=self.config.port,
            debug=False,
            use_reloader=False,
            log_output=False,
        )


# ─────────────────────────────────────────────────────────────
# Standalone Server
# ─────────────────────────────────────────────────────────────

def start_dashboard(config: Optional[DashboardConfig] = None) -> None:
    """Start the dashboard server (blocking)."""
    cfg = config or DashboardConfig()
    socketio.run(
        app,
        host=cfg.host,
        port=cfg.port,
        debug=cfg.debug,
        use_reloader=False,
    )


if __name__ == "__main__":
    print("\n  🖥️  CyberDojo Dashboard")
    print("  📡 http://127.0.0.1:5000\n")
    start_dashboard()
