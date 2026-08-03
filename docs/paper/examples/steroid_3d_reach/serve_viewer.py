#!/usr/bin/env python3
"""Export steroid viewer assets and serve them.

  uv run python serve_viewer.py --slug dexamethasone --resolution 2
  # open http://127.0.0.1:8767/
"""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER = ROOT / "viewer"


def main(
    port: int = 8767,
    slug: str = "dexamethasone",
    resolution: float = 2.0,
    open_browser: bool = True,
    skip_export: bool = False,
):
    os.chdir(ROOT)
    if not skip_export:
        from export_viewer_assets import main as export_main
        export_main(slug=slug, resolution=resolution)
    elif not (VIEWER / "data" / "structure.json").is_file():
        raise SystemExit("viewer/data/structure.json missing; run without --skip-export")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER), **kwargs)

        def log_message(self, fmt, *args):
            print(f"[viewer] {args[0]}")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"serving {VIEWER} at {url}")
        print("  Ensemble = cleaned model (tan) · True overlay = reference (blue)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--slug", type=str, default="dexamethasone")
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    main(
        port=args.port,
        slug=args.slug,
        resolution=args.resolution,
        open_browser=not args.no_browser,
        skip_export=args.skip_export,
    )
