#!/usr/bin/env python3
"""Export compare-viewer assets and serve them.

  uv run python serve_compare_viewer.py
  # open http://127.0.0.1:8768/
"""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER = ROOT / "viewer_compare"


def main(
    port: int = 8768,
    open_browser: bool = True,
    skip_export: bool = False,
    resolutions: str | None = None,
    ligands: str | None = None,
):
    os.chdir(ROOT)
    if not skip_export:
        from export_compare_assets import main as export_main, DEFAULT_RESOLUTIONS
        res = (
            tuple(float(x) for x in resolutions.split(",") if x.strip())
            if resolutions else DEFAULT_RESOLUTIONS
        )
        slug_list = (
            [s.strip() for s in ligands.split(",") if s.strip()]
            if ligands else None
        )
        print("exporting compare assets …", flush=True)
        export_main(resolutions=res, slugs=slug_list)
    elif not (VIEWER / "data" / "catalog.json").is_file():
        raise SystemExit("viewer_compare/data/catalog.json missing; run without --skip-export")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER), **kwargs)

        def end_headers(self):
            # Avoid stale catalog.json after re-exports (browsers cache aggressively).
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

        def log_message(self, fmt, *args):
            print(f"[compare] {args[0]}", flush=True)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"serving {VIEWER} at {url}", flush=True)
        print("  blue = true · tan = ×1 final · pink = ×2 final · green = density", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--resolutions", type=str, default=None)
    ap.add_argument("--ligands", type=str, default=None)
    args = ap.parse_args()
    main(
        port=args.port,
        open_browser=not args.no_browser,
        skip_export=args.skip_export,
        resolutions=args.resolutions,
        ligands=args.ligands,
    )
