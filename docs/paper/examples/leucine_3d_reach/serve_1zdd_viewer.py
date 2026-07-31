#!/usr/bin/env python3
"""Export 1ZDD placement assets (if needed) and serve the web viewer.

  uv run python docs/paper/examples/leucine_3d_reach/serve_1zdd_viewer.py
  # then open http://127.0.0.1:8766/
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIEWER = ROOT / "viewer_1zdd"
OUT = ROOT / "out"


def main(port: int = 8766, seed: int = 0, resolution: float = 2.0,
         open_browser: bool = True, force_export: bool = False):
    tag = f"{float(resolution):g}".replace(".", "p")
    npz = OUT / f"1zdd_free_ot_{tag}A_seed{seed}.npz"
    if not npz.is_file():
        print(f"running free OT → {npz.name} …", flush=True)
        from run_1zdd_free_ot import main as run_ot
        # argparse-free call via subprocess-style: import and invoke carefully
        import run_1zdd_free_ot as otmod
        import sys
        sys.argv = [
            "run_1zdd_free_ot.py",
            "--resolution", str(resolution),
            "--seed", str(seed),
        ]
        otmod.main()

    meta = VIEWER / "data" / "meta.json"
    if force_export or not meta.is_file():
        from export_1zdd_viewer import main as export_main
        export_main(npz_path=npz, seed=seed, resolution=resolution)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER), **kwargs)

        def log_message(self, fmt, *args):
            print(f"[1zdd-viewer] {args[0]}")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"serving {VIEWER} at {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--force-export", action="store_true")
    args = ap.parse_args()
    main(
        port=args.port,
        seed=args.seed,
        resolution=args.resolution,
        open_browser=not args.no_browser,
        force_export=args.force_export,
    )
