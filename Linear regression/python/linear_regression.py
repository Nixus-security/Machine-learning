"""Regression lineaire multiple, descente de gradient batch, codee from scratch avec NumPy."""
import numpy as np

from common import FEATURES, run, should_snap

LEARNING_RATE = 0.01
EPOCHS = 5000


class LinearRegression:
    def fit(self, X, y, snap):
        n, d = X.shape
        self.w, self.b, self.history = np.zeros(d), 0.0, []
        for epoch in range(EPOCHS):
            if snap and should_snap(epoch, EPOCHS):
                snap(epoch, self)
            error = X @ self.w + self.b - y
            self.history.append(float(np.mean(error ** 2)))
            self.w -= LEARNING_RATE * (2 / n) * (X.T @ error)
            self.b -= LEARNING_RATE * (2 / n) * error.sum()
        if snap:
            snap(EPOCHS, self)
        return self

    def predict(self, X):
        return X @ self.w + self.b


if __name__ == "__main__":
    run("linear", lambda X, y, snap: LinearRegression().fit(X, y, snap),
        lambda m: {"weights": dict(zip(FEATURES, m.w.tolist())), "bias": m.b})
