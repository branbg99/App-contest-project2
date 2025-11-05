#!/usr/bin/env python3
from __future__ import annotations

"""
Launch the ProjectSearchBar desktop app:
- Starts the local HTTP API + static UI server if not already running
- Opens a desktop window embedding the UI (no external browser)

Tip: pip install pywebview for the desktop window. If missing, we fall back to default browser.
"""

import threading
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
import time
import webbrowser
import sys
import socket

try:
    import webview  # type: ignore
except Exception:
    webview = None

# Make the package importable whether we run as `python3 launch.py` or `python3 ProjectSearchBar/launch.py`
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from ProjectSearchBar import config  # type: ignore
from ProjectSearchBar.server_app.server import SearchRequestHandler  # type: ignore


def run_server():
    addr = (config.HOST, config.PORT)
    httpd = ThreadingHTTPServer(addr, SearchRequestHandler)
    print(f"Serving ProjectSearchBar at http://{config.HOST}:{config.PORT}")
    httpd.serve_forever()


def main() -> None:
    # Option A: Legacy long-prompt behavior by default
    # Disable query filtering and timeouts; raise caps for results/candidates.
    os.environ.setdefault('PROJECTSEARCHBAR_NO_FILTER', '1')
    os.environ.setdefault('PROJECTSEARCHBAR_NO_TIMEOUT', '1')
    os.environ.setdefault('PROJECTSEARCHBAR_MAX_RESULTS', '20000')
    os.environ.setdefault('PROJECTSEARCHBAR_MAX_CANDIDATES', '100000')
    os.environ.setdefault('PROJECTSEARCHBAR_SCAN_BUDGET', '5000000')
    os.environ.setdefault('PROJECTSEARCHBAR_MIN_KEEP', '3')
    # Make LLM calls more resilient on slower networks
    os.environ.setdefault('PROJECTSEARCHBAR_LLM_TIMEOUT', '45')

    # If another instance is already serving, don't start a new one
    # Cache-bust UI loads to avoid stale pywebview/browser caches
    url = f"http://{config.HOST}:{config.PORT}?v={int(time.time())}"
    server_running = False
    try:
        with socket.create_connection((config.HOST, config.PORT), timeout=0.3):
            server_running = True
    except Exception:
        server_running = False

    t: threading.Thread | None = None
    if not server_running:
        # Start API server in background
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        # Give server a moment
        time.sleep(0.5)
    else:
        print(f"Detected existing server at {url}; reusing it.")

    title = "ProjectSearchBar"
    if webview is not None:
        # Desktop window embedding the local UI; no browser is launched
        window = webview.create_window(title, url)
        webview.start()
    else:
        print("pywebview is not installed. Install with: pip install pywebview")
        print("Falling back to opening in default browser for now...")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        # Keep the main thread alive so server runs
        if t is not None:
            t.join()


if __name__ == "__main__":
    main()
