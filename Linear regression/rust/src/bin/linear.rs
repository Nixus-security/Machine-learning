//! Regression lineaire multiple, descente de gradient batch, codee from scratch en Rust.

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
    let (n, d) = (x.len() as f64, x[0].len());
    let mut model = LinearRegression { w: vec![0.0; d], b: 0.0, history: Vec::with_capacity(EPOCHS) };
    for epoch in 0..EPOCHS {
        record(&mut snap, epoch, EPOCHS, &model);
        let (mut grad_w, mut grad_b, mut mse) = (vec![0.0; d], 0.0, 0.0);
        for (xi, yi) in x.iter().zip(y) {
            let error = model.predict(xi) - yi;
            for (g, v) in grad_w.iter_mut().zip(xi) {
                *g += error * v;
            }
            grad_b += error;
            mse += error * error;
        }
        for (w, g) in model.w.iter_mut().zip(&grad_w) {
            *w -= LEARNING_RATE * 2.0 / n * g;
        }
        model.b -= LEARNING_RATE * 2.0 / n * grad_b;
        model.history.push(mse / n);
    }
    record(&mut snap, EPOCHS, EPOCHS, &model);
    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("linear", fit, |m| weights_json(&m.w, m.b))
}
