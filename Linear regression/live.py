"""Test de charge en direct : Python et Rust entrainent le meme modele en parallele,
l'interface 3D (live.html) suit leur debit en temps reel via Server-Sent Events.

Usage : python live.py   puis ouvrir http://127.0.0.1:8765
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
SCRIPTS = {"linear": "linear_regression.py", "sgd": "sgd.py", "mlp": "mlp.py"}
PORT = 8765
cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
exe = ".exe" if sys.platform == "win32" else ""
# Les 2 langages tournent en meme temps : 1 seul coeur chacun pour un duel equitable
ENV = {"python": {**os.environ, "OPENBLAS_NUM_THREADS": "1"}, "rust": {**os.environ, "ML_THREADS": "1"}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(200, "text/html; charset=utf-8", (ROOT / "live.html").read_bytes())
        elif url.path == "/stream":
            self.stream(parse_qs(url.query))
        else:
            self.reply(404, "text/plain", b"not found")

    def reply(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream(self, query):
        model = query.get("model", [""])[0]
        try:
            seconds = min(max(float(query.get("seconds", ["10"])[0]), 1), 60)
        except ValueError:
            seconds = None
        if model not in SCRIPTS or seconds is None:
            return self.reply(400, "text/plain", b"parametres invalides")

        cmds = {"python": [sys.executable, str(ROOT / "python" / SCRIPTS[model]), "--bench", str(seconds)],
                "rust": [str(ROOT / "rust" / "target" / "release" / (model + exe)), "--bench", str(seconds)]}
        procs = {lang: subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=ENV[lang])
                 for lang, cmd in cmds.items()}
        events = queue.Queue()

        def pump(lang, proc):
            for line in proc.stdout:
                events.put((lang, line))
            events.put((lang, None))

        for lang, proc in procs.items():
            threading.Thread(target=pump, args=(lang, proc), daemon=True).start()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        meta = {"model": model, "seconds": seconds}
        try:
            self.wfile.write(f"event: start\ndata: {json.dumps(meta)}\n\n".encode())
            alive = len(procs)
            while alive:
                lang, line = events.get()
                if line is None:
                    alive -= 1
                    continue
                self.wfile.write(f"data: {json.dumps({'lang': lang, **json.loads(line)})}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"event: end\ndata: {}\n\n")
        except (BrokenPipeError, ConnectionResetError):  # page fermee : on arrete le test
            for proc in procs.values():
                proc.kill()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    subprocess.run([cargo, "build", "--release", "--quiet"], cwd=ROOT / "rust", check=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"> Test en direct : http://127.0.0.1:{PORT}  (Ctrl+C pour arreter)")
    webbrowser.open(f"http://127.0.0.1:{PORT}")
    server.serve_forever()
