//! Regression lineaire par descente de gradient stochastique (mini-batch + momentum), from scratch en Rust.

use ml_models::{record, run, weights_json, Lcg, Model, Snap, SEED};

const LEARNING_RATE: f64 = 0.01;
const MOMENTUM: f64 = 0.9;
const BATCH: usize = 16;
const EPOCHS: usize = 300;

struct SgdRegression {
    w: Vec<f64>,
    b: f64,
    history: Vec<f64>,
}

impl Model for SgdRegression {
    fn predict(&self, x: &[f64]) -> f64 {
        x.iter().zip(&self.w).map(|(xi, wi)| xi * wi).sum::<f64>() + self.b
    }
    fn history(&self) -> &[f64] {
        &self.history
    }
}

fn fit(x: &[Vec<f64>], y: &[f64], mut snap: Snap) -> SgdRegression {
    let (n, d) = (x.len(), x[0].len());
    let mut model = SgdRegression { w: vec![0.0; d], b: 0.0, history: Vec::with_capacity(EPOCHS) };
    let (mut vw, mut vb, mut rng) = (vec![0.0; d], 0.0, Lcg::new(SEED + 1));
    let mut idx: Vec<usize> = (0..n).collect();
    for epoch in 0..EPOCHS {
        record(&mut snap, epoch, EPOCHS, &model);
        rng.shuffle(&mut idx); // nouvel ordre a chaque epoch
        let mut sse = 0.0;
        for batch in idx.chunks(BATCH) {
            let (mut grad_w, mut grad_b) = (vec![0.0; d], 0.0);
            for &i in batch {
                let error = model.predict(&x[i]) - y[i];
                for (g, v) in grad_w.iter_mut().zip(&x[i]) {
                    *g += error * v;
                }
                grad_b += error;
                sse += error * error;
            }
            let scale = 2.0 / batch.len() as f64;
            for ((w, v), g) in model.w.iter_mut().zip(&mut vw).zip(&grad_w) {
                *v = MOMENTUM * *v + scale * g;
                *w -= LEARNING_RATE * *v;
            }
            vb = MOMENTUM * vb + scale * grad_b;
            model.b -= LEARNING_RATE * vb;
        }
        model.history.push(sse / n as f64);
    }
    record(&mut snap, EPOCHS, EPOCHS, &model);
    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("sgd", fit, |m| weights_json(&m.w, m.b))
}
