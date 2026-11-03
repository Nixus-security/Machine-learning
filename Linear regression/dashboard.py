"""Lance les deux modeles (Python + Rust) puis ouvre un tableau de bord comparatif dans le navigateur."""
import json
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")

print("> Entrainement Python...")
subprocess.run([sys.executable, str(ROOT / "python" / "linear_regression.py")], check=True)
print("\n> Entrainement Rust...")
subprocess.run([cargo, "run", "--release", "--quiet"], cwd=ROOT / "rust", check=True)

data = {lang: json.loads((RESULTS / f"{lang}.json").read_text()) for lang in ("python", "rust")}
template = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
out = RESULTS / "dashboard.html"
out.write_text(template.replace("__DATA__", json.dumps(data)), encoding="utf-8")
print(f"\n> Tableau de bord : {out}")
webbrowser.open(out.as_uri())
