//! Regression lineaire multiple (descente de gradient) codee from scratch en Rust, sans dependance.
//!
//! Objectif : predire le prix d'une voiture a partir de ses caracteristiques techniques.
//! Dataset : data/Automobile_data.csv

use std::error::Error;
use std::fs;
use std::time::Instant;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/Automobile_data.csv");
const RESULTS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../results");
const FEATURES: [&str; 13] = [
    "wheel-base", "length", "width", "height", "curb-weight", "engine-size",
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg",
];
const TARGET: &str = "price";
const LEARNING_RATE: f64 = 0.01;
const EPOCHS: usize = 5000;
const TEST_RATIO: f64 = 0.2;
const SEED: u64 = 42;

/// Charge le CSV et ignore les lignes avec des valeurs manquantes ('?').
fn load_data() -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let text = fs::read_to_string(DATA)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("CSV vide")?.split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).ok_or(format!("colonne {name} absente"));
    let cols = FEATURES.iter().map(|f| col(f)).collect::<Result<Vec<_>, _>>()?;
    let target = col(TARGET)?;

    let (mut x, mut y) = (Vec::new(), Vec::new());
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').collect();
        let row: Result<Vec<f64>, _> = cols.iter().chain([&target]).map(|&c| fields[c].parse::<f64>()).collect();
        if let Ok(mut row) = row {
            y.push(row.pop().unwrap());
            x.push(row);
        }
    }
    Ok((x, y))
}

/// Fisher-Yates avec un generateur LCG 64 bits (identique a la version Python).
fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((state >> 33) % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

struct LinearRegression {
    w: Vec<f64>,
    b: f64,
    history: Vec<f64>,
    snapshots: Vec<String>,
}

impl LinearRegression {
    /// Etat du modele (JSON) pour l'animation dans l'interface.
    fn snapshot(&mut self, epoch: usize) {
        let s = format!(r#"{{"epoch":{epoch},"w":{},"b":{}}}"#, json_array(&self.w), self.b);
        self.snapshots.push(s);
    }
}

fn json_array(values: &[f64]) -> String {
    format!("[{}]", values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
}

impl LinearRegression {
    fn predict(&self, x: &[f64]) -> f64 {
        x.iter().zip(&self.w).map(|(xi, wi)| xi * wi).sum::<f64>() + self.b
    }

    fn fit(x: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) -> Self {
        let (n, d) = (x.len() as f64, x[0].len());
        let mut model = LinearRegression { w: vec![0.0; d], b: 0.0, history: Vec::with_capacity(epochs), snapshots: Vec::new() };
        for epoch in 0..epochs {
            if epoch < 200 || epoch % 20 == 0 {
                model.snapshot(epoch);
            }
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
                *w -= lr * 2.0 / n * g;
            }
            model.b -= lr * 2.0 / n * grad_b;
            model.history.push(mse / n);
            if epoch % 1000 == 0 {
                println!("epoch {epoch:5} | MSE train = {:.0}", mse / n);
            }
        }
        model.snapshot(epochs);
        model
    }
}

fn metrics(y: &[f64], pred: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let ss_res: f64 = y.iter().zip(pred).map(|(a, p)| (a - p).powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
    ((ss_res / n).sqrt(), 1.0 - ss_res / ss_tot)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (x, y) = load_data()?;
    let idx = shuffle_indices(y.len(), SEED);
    let n_test = (y.len() as f64 * TEST_RATIO) as usize;
    let (test, train) = idx.split_at(n_test);

    // Standardisation calculee uniquement sur le jeu d'entrainement
    let d = FEATURES.len();
    let nt = train.len() as f64;
    let mean: Vec<f64> = (0..d).map(|j| train.iter().map(|&i| x[i][j]).sum::<f64>() / nt).collect();
    let std: Vec<f64> = (0..d)
        .map(|j| (train.iter().map(|&i| (x[i][j] - mean[j]).powi(2)).sum::<f64>() / nt).sqrt())
        .collect();
    let xs: Vec<Vec<f64>> = x.iter().map(|r| (0..d).map(|j| (r[j] - mean[j]) / std[j]).collect()).collect();

    println!("{} voitures | train = {} | test = {}\n", y.len(), train.len(), test.len());
    let pick = |ids: &[usize]| -> (Vec<Vec<f64>>, Vec<f64>) {
        (ids.iter().map(|&i| xs[i].clone()).collect(), ids.iter().map(|&i| y[i]).collect())
    };
    let (x_train, y_train) = pick(train);
    let start = Instant::now();
    let model = LinearRegression::fit(&x_train, &y_train, LEARNING_RATE, EPOCHS);
    let train_ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut splits = Vec::new();
    for (name, ids) in [("train", train), ("test", test)] {
        let (xp, yp) = pick(ids);
        let pred: Vec<f64> = xp.iter().map(|r| model.predict(r)).collect();
        let (rmse, r2) = metrics(&yp, &pred);
        print!("\n[{name}] RMSE = {rmse:.0} $ | R2 = {r2:.4}");
        splits.push(format!(
            r#""{name}":{{"rmse":{rmse},"r2":{r2},"actual":{},"predicted":{},"x":[{}]}}"#,
            json_array(&yp),
            json_array(&pred),
            xp.iter().map(|r| json_array(r)).collect::<Vec<_>>().join(",")
        ));
    }
    print!("\nTemps d'entrainement : {train_ms:.1} ms");

    let weights = FEATURES.iter().zip(&model.w).map(|(f, w)| format!(r#""{f}":{w}"#)).collect::<Vec<_>>().join(",");
    fs::create_dir_all(RESULTS_DIR)?;
    fs::write(
        format!("{RESULTS_DIR}/rust.json"),
        format!(
            r#"{{"language":"Rust","train_ms":{train_ms},"loss":{},"weights":{{{weights}}},"bias":{},"snapshots":[{}],{}}}"#,
            json_array(&model.history),
            model.b,
            model.snapshots.join(","),
            splits.join(",")
        ),
    )?;

    println!("\n\nPoids (features standardisees) :");
    let mut weights: Vec<(&str, f64)> = FEATURES.iter().copied().zip(model.w.iter().copied()).collect();
    weights.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    for (f, w) in weights {
        println!("  {f:18} {w:+10.1}");
    }
    println!("  {:18} {:+10.1}", "biais", model.b);

    println!("\nExemples de predictions (test) :");
    for &i in &test[..5] {
        println!("  reel = {:>8.0} $ | predit = {:>8.0} $", y[i], model.predict(&xs[i]));
    }
    Ok(())
}
