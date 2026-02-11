"""Simple HTTP server for the agent-economy dashboard."""

from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from agent_economy.ledger import HashChainedLedger
from agent_economy.state import replay_ledger


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the dashboard API and static files."""

    run_dir: Path
    static_dir: Path

    def __init__(self, *args, **kwargs) -> None:
        # Set directory to static files location
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/api/state":
            self._serve_state()
        elif self.path == "/api/events":
            self._serve_events()
        elif self.path == "/api/events/stream":
            self._serve_event_stream()
        else:
            # Serve static files, default to index.html
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def _serve_state(self) -> None:
        """Serve current derived state as JSON."""
        try:
            ledger = HashChainedLedger(self.run_dir / "ledger.jsonl")
            events = list(ledger.iter_events())
            if not events:
                self._send_json({"error": "no events"}, status=404)
                return
            state = replay_ledger(events=events)
            self._send_json(state.model_dump(mode="json"))
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _serve_events(self) -> None:
        """Serve ledger events as JSON array."""
        try:
            ledger = HashChainedLedger(self.run_dir / "ledger.jsonl")
            events = list(ledger.iter_events())
            # Return last 100 events for performance
            recent = events[-100:] if len(events) > 100 else events
            self._send_json([e.model_dump(mode="json") for e in recent])
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    def _serve_event_stream(self) -> None:
        """Serve events as Server-Sent Events stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        last_count = 0
        try:
            while True:
                ledger = HashChainedLedger(self.run_dir / "ledger.jsonl")
                events = list(ledger.iter_events())
                if len(events) > last_count:
                    # Send new events
                    for e in events[last_count:]:
                        data = json.dumps(e.model_dump(mode="json"))
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                    last_count = len(events)

                    # Also send updated state
                    state = replay_ledger(events=events)
                    state_data = json.dumps(
                        {"type": "state", "data": state.model_dump(mode="json")}
                    )
                    self.wfile.write(f"data: {state_data}\n\n".encode())
                    self.wfile.flush()

                time.sleep(2)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response."""
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass


def create_handler_class(*, run_dir: Path, static_dir: Path) -> type:
    """Create a handler class with the run_dir bound."""

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.run_dir = run_dir
    BoundHandler.static_dir = static_dir
    return BoundHandler


def run_dashboard(*, run_dir: Path, port: int = 8080, open_browser: bool = True) -> None:
    """Run the dashboard server."""
    static_dir = Path(__file__).parent / "static"
    if not static_dir.exists():
        raise FileNotFoundError(f"static directory not found: {static_dir}")

    handler_class = create_handler_class(run_dir=run_dir, static_dir=static_dir)
    server = HTTPServer(("127.0.0.1", port), handler_class)

    print(f"Dashboard running at http://localhost:{port}")
    print(f"Watching run directory: {run_dir}")
    print("Press Ctrl+C to stop\n")

    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        server.shutdown()
