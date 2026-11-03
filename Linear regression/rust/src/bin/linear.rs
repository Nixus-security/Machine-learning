//! Regression lineaire multiple, descente de gradient batch, codee from scratch en Rust.
//! Le lot est stocke par variable : erreurs et gradients sont calcules par les noyaux SIMD.

use ml_models::simd::{dot, matmul, sum, to_columns};
use ml_models::{record, run, weights_json, Model, Snap};

const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 5000;

struct LinearRegression {
    w: Vec<f64>,
    b: f64,
    history: Vec<f64>,
}

impl Model for LinearRegression {
    fn predict(&self, x: &[f64]) -> f64 {
        x.iter().zip(&self.w).map(|(xi, wi)| xi * wi).sum::<f64>() + self.b
    }
    fn history(&self) -> &[f64] {
        &self.history
    }
}

fn fit(x: &[Vec<f64>], y: &[f64], mut snap: Snap) -> LinearRegression {
    let (cols, np) = to_columns(x); // np = n arrondi pour le SIMD, remplissage a 0
    let (n, scale) = (x.len(), LEARNING_RATE * 2.0 / x.len() as f64);
    let mut model = LinearRegression { w: vec![0.0; x[0].len()], b: 0.0, history: Vec::with_capacity(EPOCHS) };
    let mut error = vec![0.0; np];
    for epoch in 0..EPOCHS {
        record(&mut snap, epoch, EPOCHS, &model);
        let (w, b) = (&model.w, model.b);
        matmul(&mut error, &cols, np, |_| b, |_, j| w[j]); // error = X w + b
        error.iter_mut().zip(y).for_each(|(e, y)| *e -= y);
        error[n..].fill(0.0); // exemples de remplissage : sans effet sur les gradients
        for (w, col) in model.w.iter_mut().zip(cols.chunks_exact(np)) {
            *w -= scale * dot(col, &error);
        }
        model.b -= scale * sum(&error);
        model.history.push(dot(&error, &error) / n as f64);
    }
    record(&mut snap, EPOCHS, EPOCHS, &model);
    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("linear", fit, |m| weights_json(&m.w, m.b))
}
