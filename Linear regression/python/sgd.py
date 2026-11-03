"""Regression lineaire par descente de gradient stochastique (mini-batch + momentum), from scratch avec NumPy."""
import numpy as np

from common import FEATURES, SEED, Lcg, run, should_snap

LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH = 16
EPOCHS = 300


class SgdRegression:
    def fit(self, X, y, snap):
        n, d = X.shape
        self.w, self.b, self.history = np.zeros(d), 0.0, []
        vw, vb, rng = np.zeros(d), 0.0, Lcg(SEED + 1)
        idx = list(range(n))
        for epoch in range(EPOCHS):
            if snap and should_snap(epoch, EPOCHS):
                snap(epoch, self)
            rng.shuffle(idx)  # nouvel ordre a chaque epoch
            sse = 0.0
            for s in range(0, n, BATCH):
                batch = idx[s:s + BATCH]
                xb, error = X[batch], X[batch] @ self.w + self.b - y[batch]
                sse += float(error @ error)
                vw = MOMENTUM * vw + (2 / len(batch)) * (xb.T @ error)
                vb = MOMENTUM * vb + (2 / len(batch)) * error.sum()
                self.w -= LEARNING_RATE * vw
                self.b -= LEARNING_RATE * vb
            self.history.append(sse / n)
        if snap:
            snap(EPOCHS, self)
        return self

    def predict(self, X):
        return X @ self.w + self.b


if __name__ == "__main__":
    run("sgd", lambda X, y, snap: SgdRegression().fit(X, y, snap),
        lambda m: {"weights": dict(zip(FEATURES, m.w.tolist())), "bias": m.b})
