#!/usr/bin/env python3
"""Serve the RDKit screen explorer.

  uv run python serve_rdkit_viewer.py
  # http://127.0.0.1:8770/
"""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER = ROOT / "viewer_rdkit"


def main(port: int = 8770, open_browser: bool = True, skip_export: bool = False,
         ligands: str | None = None):
    os.chdir(ROOT)
    if not skip_export:
        from export_rdkit_viewer_assets import main as export_main
        slug_list = (
            [s.strip() for s in ligands.split(",") if s.strip()]
            if ligands else None
        )
        print("exporting RDKit viewer assets …", flush=True)
        export_main(slugs=slug_list)
    elif not (VIEWER / "data" / "catalog.json").is_file():
        raise SystemExit("viewer_rdkit/data/catalog.json missing")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER), **kwargs)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store, max-age=0")
            super().end_headers()

        def log_message(self, fmt, *args):
            print(f"[rdkit-viewer] {args[0]}", flush=True)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"serving {VIEWER} at {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--ligands", type=str, default=None)
    args = ap.parse_args()
    main(port=args.port, open_browser=not args.no_browser,
         skip_export=args.skip_export, ligands=args.ligands)
