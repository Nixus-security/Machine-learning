"""Regression lineaire multiple (descente de gradient) codee from scratch avec NumPy.

Objectif : predire le prix d'une voiture a partir de ses caracteristiques techniques.
Dataset : data/Automobile_data.csv
"""
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Automobile_data.csv"
RESULTS = ROOT / "results" / "python.json"
FEATURES = [
    "wheel-base", "length", "width", "height", "curb-weight", "engine-size",
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg",
]
TARGET = "price"
LEARNING_RATE = 0.01
EPOCHS = 5000
TEST_RATIO = 0.2
SEED = 42


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


def shuffle_indices(n, seed):
    """Fisher-Yates avec un generateur LCG 64 bits (identique a la version Rust)."""
    mask = (1 << 64) - 1
    state = seed
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & mask
        j = (state >> 33) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return np.array(idx)


class LinearRegression:
    def __init__(self, lr, epochs):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        self.history, self.snapshots = [], []
        for epoch in range(self.epochs):
            if epoch < 200 or epoch % 20 == 0:  # etat du modele pour l'animation
                self.snapshots.append({"epoch": epoch, "w": self.w.tolist(), "b": self.b})
            error = X @ self.w + self.b - y
            self.history.append(float(np.mean(error ** 2)))
            self.w -= self.lr * (2 / n) * (X.T @ error)
            self.b -= self.lr * (2 / n) * error.sum()
            if epoch % 1000 == 0:
                print(f"epoch {epoch:5d} | MSE train = {np.mean(error ** 2):,.0f}")
        self.snapshots.append({"epoch": self.epochs, "w": self.w.tolist(), "b": self.b})
        return self

    def predict(self, X):
        return X @ self.w + self.b


def metrics(y, pred):
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return rmse, r2


def main():
    X, y = load_data()
    idx = shuffle_indices(len(y), SEED)
    n_test = int(len(y) * TEST_RATIO)
    test, train = idx[:n_test], idx[n_test:]

    # Standardisation calculee uniquement sur le jeu d'entrainement
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    Xs = (X - mean) / std

    print(f"{len(y)} voitures | train = {len(train)} | test = {len(test)}\n")
    start = time.perf_counter()
    model = LinearRegression(LEARNING_RATE, EPOCHS).fit(Xs[train], y[train])
    train_ms = (time.perf_counter() - start) * 1000

    results = {"language": "Python", "train_ms": train_ms, "loss": model.history,
               "weights": dict(zip(FEATURES, model.w.tolist())), "bias": model.b,
               "snapshots": model.snapshots}
    for name, part in (("train", train), ("test", test)):
        pred = model.predict(Xs[part])
        rmse, r2 = metrics(y[part], pred)
        results[name] = {"rmse": rmse, "r2": r2, "actual": y[part].tolist(), "predicted": pred.tolist(),
                         "x": Xs[part].tolist()}
        print(f"\n[{name}] RMSE = {rmse:,.0f} $ | R2 = {r2:.4f}", end="")
    print(f"\nTemps d'entrainement : {train_ms:.1f} ms", end="")
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results))
    print("\n\nPoids (features standardisees) :")
    for f, w in sorted(zip(FEATURES, model.w), key=lambda t: -abs(t[1])):
        print(f"  {f:18s} {w:+10.1f}")
    print(f"  {'biais':18s} {model.b:+10.1f}")

    print("\nExemples de predictions (test) :")
    for i in test[:5]:
        print(f"  reel = {y[i]:>8,.0f} $ | predit = {model.predict(Xs[i:i + 1])[0]:>8,.0f} $")


if __name__ == "__main__":
    main()
