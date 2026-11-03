"""Briques partagees par les modeles Python : donnees, split, metriques, benchmark, export JSON."""
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Automobile_data.csv"
FEATURES = [
    "wheel-base", "length", "width", "height", "curb-weight", "engine-size",
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg",
]
TARGET = "price"
TEST_RATIO = 0.2
SEED = 42
RUNS = 5  # temps reporte = mediane de RUNS entrainements
MASK = (1 << 64) - 1


class Lcg:
    """Generateur LCG 64 bits, identique bit a bit a la version Rust."""

    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK
        return self.state

    def below(self, n):
        return (self.next() >> 33) % n

    def uniform(self):
        """Flottant dans [-1, 1)."""
        return (self.next() >> 11) / 2.0 ** 52 - 1.0

    def shuffle(self, idx):
        for i in range(len(idx) - 1, 0, -1):
            j = self.below(i + 1)
            idx[i], idx[j] = idx[j], idx[i]
        return idx


def load_data():
    """Charge le CSV et ignore les lignes avec des valeurs manquantes ('?')."""
    X, y = [], []
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            values = [row[c] for c in FEATURES + [TARGET]]
            if "?" in values:
                continue
            X.append([float(v) for v in values[:-1]])
            y.append(float(values[-1]))
    return np.array(X), np.array(y)


def should_snap(epoch, epochs):
    """~200 images par entrainement pour l'animation."""
    return epoch < 100 or epoch % (epochs // 100) == 0


def metrics(y, pred):
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(rmse), float(r2)


def run(name, fit, extra=lambda model: {}):
    """fit(X, y, snap) -> modele avec .predict(X) et .history ; snap(epoch, model) enregistre l'etat."""
    X, y = load_data()
    idx = np.array(Lcg(SEED).shuffle(list(range(len(y)))))
    n_test = int(len(y) * TEST_RATIO)
    test, train = idx[:n_test], idx[n_test:]
    # Standardisation calculee uniquement sur le jeu d'entrainement
    Xs = (X - X[train].mean(axis=0)) / X[train].std(axis=0)

    snapshots = []
    snap = lambda epoch, m: snapshots.append(
        {"epoch": epoch, "train": m.predict(Xs[train]).tolist(), "test": m.predict(Xs[test]).tolist()})
    model = fit(Xs[train], y[train], snap)

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fit(Xs[train], y[train], None)
        times.append((time.perf_counter() - start) * 1000)

    out = {"model": name, "language": "Python", "train_ms": statistics.median(times), "runs_ms": times,
           "loss": model.history, "snapshots": snapshots, **extra(model)}
    print(f"[{name}] Python")
    for part, ids in (("train", train), ("test", test)):
        pred = model.predict(Xs[ids])
        rmse, r2 = metrics(y[ids], pred)
        out[part] = {"rmse": rmse, "r2": r2, "actual": y[ids].tolist(), "predicted": pred.tolist()}
        print(f"  {part:5} RMSE = {rmse:>8,.0f} $ | R2 = {r2:.4f}")
    print(f"  temps median ({RUNS} runs) = {out['train_ms']:.2f} ms\n")
    path = ROOT / "results" / f"{name}_python.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
