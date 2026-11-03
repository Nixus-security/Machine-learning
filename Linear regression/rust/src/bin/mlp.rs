//! Deep learning : perceptron multicouche (13 -> 32 -> 16 -> 1, ReLU, Adam), from scratch en Rust.

use ml_models::{record, run, Lcg, Model, Snap, SEED};

const LAYERS: [usize; 4] = [13, 32, 16, 1];
const LEARNING_RATE: f64 = 0.001;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const EPOCHS: usize = 1000;

struct Mlp {
    w: Vec<Vec<f64>>, // couche l : matrice (in x out) a plat, ligne par ligne
    b: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
    history: Vec<f64>,
}

/// Un vecteur de zeros par couche : `sizes` = tailles voulues.
fn zeros(sizes: impl Iterator<Item = usize>) -> Vec<Vec<f64>> {
    sizes.map(|k| vec![0.0; k]).collect()
}

impl Mlp {
    /// Remplit `acts` (une entree par couche, entree incluse) sans allocation.
    fn forward(&self, x: &[f64], acts: &mut [Vec<f64>]) {
        acts[0].copy_from_slice(x);
        for l in 0..self.w.len() {
            let (prev, next) = acts.split_at_mut(l + 1);
            let z = &mut next[0];
            z.copy_from_slice(&self.b[l]);
            for (a, row) in prev[l].iter().zip(self.w[l].chunks_exact(z.len())) {
                z.iter_mut().zip(row).for_each(|(zo, wo)| *zo += a * wo);
            }
            if l < self.w.len() - 1 {
                z.iter_mut().for_each(|v| *v = v.max(0.0)); // ReLU
            }
        }
    }
}

impl Model for Mlp {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut acts = zeros(LAYERS.into_iter());
        self.forward(x, &mut acts);
        acts[LAYERS.len() - 1][0] * self.y_std + self.y_mean
    }
    fn history(&self) -> &[f64] {
        &self.history
    }
}

fn fit(x: &[Vec<f64>], y: &[f64], mut snap: Snap) -> Mlp {
    let pairs: Vec<(usize, usize)> = LAYERS.into_iter().zip(LAYERS.into_iter().skip(1)).collect();
    let mut rng = Lcg::new(SEED + 2);
    // Init He uniforme, meme sequence aleatoire que la version Python
    let w = pairs.iter().map(|&(i, o)| (0..i * o).map(|_| rng.uniform() * (6.0 / i as f64).sqrt()).collect()).collect();
    let n = y.len() as f64;
    let y_mean = y.iter().sum::<f64>() / n;
    let y_std = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n).sqrt(); // cible standardisee
    let mut model = Mlp { w, b: zeros(LAYERS[1..].iter().copied()), y_mean, y_std, history: Vec::with_capacity(EPOCHS) };

    // Buffers alloues une seule fois : gradients, moments Adam (poids puis biais), activations, deltas
    let sizes = || pairs.iter().map(|&(i, o)| i * o).chain(LAYERS[1..].iter().copied());
    let (mut grads, mut m, mut v) = (zeros(sizes()), zeros(sizes()), zeros(sizes()));
    let mut wt = zeros(pairs.iter().map(|&(i, o)| i * o)); // poids transposes (out x in)
    let mut acts = zeros(LAYERS.into_iter());
    let mut deltas = zeros(LAYERS[1..].iter().copied());
    let last = pairs.len() - 1;

    for epoch in 0..EPOCHS {
        record(&mut snap, epoch, EPOCHS, &model);
        grads.iter_mut().for_each(|g| g.fill(0.0));
        for (l, &(i_n, o_n)) in pairs.iter().enumerate() {
            for i in 0..i_n {
                for o in 0..o_n {
                    wt[l][o * i_n + i] = model.w[l][i * o_n + o];
                }
            }
        }
        let mut sse = 0.0;
        for (xi, yi) in x.iter().zip(y) {
            model.forward(xi, &mut acts);
            let err = acts[last + 1][0] - (yi - y_mean) / y_std;
            sse += err * err;
            deltas[last][0] = 2.0 / n * err;
            for l in (0..=last).rev() {
                // retropropagation : gradient de la couche l, puis delta de la couche precedente
                let (lo, hi) = deltas.split_at_mut(l);
                let (d, (gw, gb)) = (&hi[0], grads.split_at_mut(pairs.len()));
                for (a, row) in acts[l].iter().zip(gw[l].chunks_exact_mut(d.len())) {
                    row.iter_mut().zip(d).for_each(|(g, dv)| *g += a * dv);
                }
                gb[l].iter_mut().zip(d).for_each(|(g, dv)| *g += dv);
                if l > 0 {
                    let prev = &mut lo[l - 1];
                    prev.fill(0.0);
                    for (dv, col) in d.iter().zip(wt[l].chunks_exact(prev.len())) {
                        prev.iter_mut().zip(col).for_each(|(p, w)| *p += dv * w);
                    }
                    prev.iter_mut().zip(&acts[l]).for_each(|(p, a)| if *a <= 0.0 { *p = 0.0 });
                }
            }
        }
        model.history.push(sse / n * y_std * y_std);

        // Adam
        let t = (epoch + 1) as i32;
        let (c1, c2) = (1.0 - BETA1.powi(t), 1.0 - BETA2.powi(t));
        let params = model.w.iter_mut().chain(model.b.iter_mut());
        for (((p, g), m), v) in params.zip(&grads).zip(&mut m).zip(&mut v) {
            for k in 0..p.len() {
                m[k] = BETA1 * m[k] + (1.0 - BETA1) * g[k];
                v[k] = BETA2 * v[k] + (1.0 - BETA2) * g[k] * g[k];
                p[k] -= LEARNING_RATE * (m[k] / c1) / ((v[k] / c2).sqrt() + EPS);
            }
        }
    }
    record(&mut snap, EPOCHS, EPOCHS, &model);
    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("mlp", fit, |_| vec![format!(r#""layers":{LAYERS:?}"#)])
}
