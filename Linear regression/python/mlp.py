"""Deep learning : perceptron multicouche (13 -> 32 -> 16 -> 1, ReLU, Adam), from scratch avec NumPy."""
import numpy as np

from common import SEED, Lcg, run, should_snap

LAYERS = [13, 32, 16, 1]
LEARNING_RATE = 0.001
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
EPOCHS = 1000


class Mlp:
    def fit(self, X, y, snap):
        rng = Lcg(SEED + 2)
        # Init He uniforme, meme sequence aleatoire que la version Rust
        self.W = [np.array([[rng.uniform() * np.sqrt(6 / i) for _ in range(o)] for _ in range(i)])
                  for i, o in zip(LAYERS, LAYERS[1:])]
        self.B = [np.zeros(o) for o in LAYERS[1:]]
        self.y_mean, self.y_std = y.mean(), y.std()  # cible standardisee pour stabiliser l'entrainement
        ys, n = (y - self.y_mean) / self.y_std, len(y)
        params = self.W + self.B
        m, v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        self.history = []
        for epoch in range(EPOCHS):
            if snap and should_snap(epoch, EPOCHS):
                snap(epoch, self)
            acts = self.forward(X)
            delta = acts[-1][:, 0] - ys
            self.history.append(float(np.mean(delta ** 2)) * self.y_std ** 2)
            delta = (2 / n) * delta[:, None]
            gW, gB = [], []
            for layer in range(len(self.W) - 1, -1, -1):  # retropropagation
                gW.insert(0, acts[layer].T @ delta)
                gB.insert(0, delta.sum(axis=0))
                if layer:
                    delta = (delta @ self.W[layer].T) * (acts[layer] > 0)
            t = epoch + 1
            for p, g, mi, vi in zip(params, gW + gB, m, v):  # Adam, mise a jour en place
                mi[...] = BETA1 * mi + (1 - BETA1) * g
                vi[...] = BETA2 * vi + (1 - BETA2) * g * g
                p -= LEARNING_RATE * (mi / (1 - BETA1 ** t)) / (np.sqrt(vi / (1 - BETA2 ** t)) + EPS)
        if snap:
            snap(EPOCHS, self)
        return self

    def forward(self, X):
        acts = [X]
        for layer, (w, b) in enumerate(zip(self.W, self.B)):
            z = acts[-1] @ w + b
            acts.append(z if layer == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def predict(self, X):
        return self.forward(X)[-1][:, 0] * self.y_std + self.y_mean


if __name__ == "__main__":
    run("mlp", lambda X, y, snap: Mlp().fit(X, y, snap), lambda m: {"layers": LAYERS})
