"""Viewer server: static files from the project root, plus POST /api/steer so the viewer can steer the
moving live controller (writes runs/<stream>/control.json, which tools/live_controller.py polls).

    python tools/serve.py            # http://localhost:8765/viewer/
"""
import json, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STREAMS = {"live_dynamic", "live_feeding", "live"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_POST(self):
        if self.path.startswith("/api/steer"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                stream = body.get("stream", "live_dynamic")
                if stream not in STREAMS:
                    raise ValueError("unknown stream")
                wp = [float(body["x"]), float(body["y"])]
                out = os.path.join(ROOT, "runs", stream)
                os.makedirs(out, exist_ok=True)
                tmp = os.path.join(out, "control.json.tmp")
                with open(tmp, "w") as f:
                    json.dump({"waypoint": wp}, f)
                os.replace(tmp, os.path.join(out, "control.json"))
                self._send(200, {"ok": True, "waypoint": wp})
            except Exception as e:                      # bad request
                self._send(400, {"ok": False, "error": str(e)})
        else:
            self._send(404, {"ok": False})

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
