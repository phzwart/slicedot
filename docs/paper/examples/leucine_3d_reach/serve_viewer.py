#!/usr/bin/env python3
"""Export LRP assets (if needed) and serve the web viewer.

  PYTHONPATH=../../../src python serve_viewer.py
  # then open http://127.0.0.1:8765/
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER = ROOT / "viewer"


def main(port: int = 8765, sequence: str = "LRP", resolution: float = 3.0,
         open_browser: bool = True):
    os.chdir(ROOT)
    # Ensure assets exist
    from export_viewer_assets import main as export_main
    export_main(resolution=resolution, sequence=sequence)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER), **kwargs)

        def log_message(self, fmt, *args):
            print(f"[viewer] {args[0]}")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"serving {VIEWER} at {url}")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sequence", type=str, default="LRP")
    ap.add_argument("--resolution", type=float, default=3.0)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    main(
        port=args.port,
        sequence=args.sequence,
        resolution=args.resolution,
        open_browser=not args.no_browser,
    )
