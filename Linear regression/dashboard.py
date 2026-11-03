"""Entraine les 3 modeles en Python puis en Rust et ouvre le tableau de bord comparatif dans le navigateur."""
import json
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = {"linear": "linear_regression.py", "sgd": "sgd.py", "mlp": "mlp.py"}
cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")

subprocess.run([cargo, "build", "--release", "--quiet"], cwd=ROOT / "rust", check=True)
for model, script in MODELS.items():
    subprocess.run([sys.executable, str(ROOT / "python" / script)], check=True)
    subprocess.run([cargo, "run", "--release", "--quiet", "--bin", model], cwd=ROOT / "rust", check=True)

data = {m: {lang: json.loads((RESULTS / f"{m}_{lang}.json").read_text()) for lang in ("python", "rust")}
        for m in MODELS}
template = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
out = RESULTS / "dashboard.html"
out.write_text(template.replace("__DATA__", json.dumps(data)), encoding="utf-8")
print(f"> Tableau de bord : {out}")
webbrowser.open(out.as_uri())
