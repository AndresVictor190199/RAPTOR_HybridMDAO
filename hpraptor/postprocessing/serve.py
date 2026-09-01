"""
Serve the reports directory over http, for viewing the interactive pages.

Why this exists rather than just double-clicking the file: Chromium
browsers (Edge, Chrome) apply a stricter security model to ``file://``
than to ``http://``. Pages opened from disk get an opaque origin, which
blocks Web Workers, cross-origin ``fetch``, and several storage APIs. A
vtk.js scene that fails to load under those restrictions does not raise
anything visible -- the viewer simply stays on its empty "drop a file"
landing state, which looks identical to a broken export.

Serving the same file over ``http://localhost`` gives it a real origin and
removes that whole class of failure. If a page works here but not from
disk, the file was never the problem.

Usage
-----
    python -m hpraptor.postprocessing.serve
    python -m hpraptor.postprocessing.serve --dir reports --port 8000 --open
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import functools
import webbrowser
from pathlib import Path


def serve(directory: str = "reports", port: int = 8000,
          open_browser: bool = False) -> None:
    """Serve ``directory`` on localhost until interrupted."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(root))
    # Without this, restarting quickly after Ctrl+C fails with "address
    # already in use" while the socket sits in TIME_WAIT.
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        base = f"http://localhost:{port}/"
        print(f"Serving {root}")
        print(f"  {base}")
        pages = sorted(root.glob("*_interactive.html"))
        for p in pages:
            print(f"  {base}{p.name}")
        if not pages:
            print("  (no *_interactive.html found -- generate one first)")
        print("\nCtrl+C to stop.")
        if open_browser and pages:
            webbrowser.open(f"{base}{pages[0].name}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


def main():
    p = argparse.ArgumentParser(
        description="Serve reports/ over http so the interactive 3D pages "
                    "load without file:// restrictions.")
    p.add_argument("--dir", default="reports", help="directory to serve")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true",
                   help="open the first interactive page in a browser")
    args = p.parse_args()
    serve(args.dir, args.port, args.open)


if __name__ == "__main__":
    main()
