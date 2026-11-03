//! Briques partagees par les modeles Rust : donnees, split, metriques, benchmark, export JSON.
//! Aucune dependance externe : uniquement la bibliotheque standard.

pub mod simd;

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
const TEST_RATIO: f64 = 0.2;
pub const SEED: u64 = 42;
const RUNS: usize = 5; // temps reporte = mediane de RUNS entrainements

/// Generateur LCG 64 bits, identique bit a bit a la version Python.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// Flottant dans [-1, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
    /// Fisher-Yates.
    pub fn shuffle(&mut self, idx: &mut [usize]) {
        for i in (1..idx.len()).rev() {
            let j = ((self.next() >> 33) % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
    }
}

pub trait Model {
    fn predict(&self, x: &[f64]) -> f64;
    fn history(&self) -> &[f64];
}

/// Callback d'enregistrement de l'etat du modele (None pendant les runs chronometres).
pub type Snap<'a> = Option<&'a mut dyn FnMut(usize, &dyn Model)>;

/// ~200 images par entrainement pour l'animation (+ etat final).
pub fn record(snap: &mut Snap, epoch: usize, epochs: usize, model: &dyn Model) {
    if let Some(s) = snap {
        if epoch == epochs || epoch < 100 || epoch % (epochs / 100) == 0 {
            s(epoch, model);
        }
    }
}

/// Charge le CSV et ignore les lignes avec des valeurs manquantes ('?').
fn load_data() -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let text = fs::read_to_string(DATA)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("CSV vide")?.split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).ok_or(format!("colonne {name} absente"));
    let cols = FEATURES.iter().chain([&TARGET]).map(|f| col(f)).collect::<Result<Vec<_>, _>>()?;

    let (mut x, mut y) = (Vec::new(), Vec::new());
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').collect();
        let row: Result<Vec<f64>, _> = cols.iter().map(|&c| fields.get(c).ok_or("ligne courte")?.parse::<f64>().map_err(|_| "?")).collect();
        if let Ok(mut row) = row {
            y.push(row.pop().unwrap());
            x.push(row);
        }
    }
    Ok((x, y))
}

fn metrics(y: &[f64], pred: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let ss_res: f64 = y.iter().zip(pred).map(|(a, p)| (a - p).powi(2)).sum();
    let ss_tot: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
    ((ss_res / n).sqrt(), 1.0 - ss_res / ss_tot)
}

fn json_array(values: &[f64]) -> String {
    format!("[{}]", values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
}

/// Champs JSON des modeles lineaires : poids par variable + biais.
pub fn weights_json(w: &[f64], b: f64) -> Vec<String> {
    let weights = FEATURES.iter().zip(w).map(|(f, w)| format!(r#""{f}":{w}"#)).collect::<Vec<_>>().join(",");
    vec![format!(r#""weights":{{{weights}}},"bias":{b}"#)]
}

/// Test de charge : entraine en boucle pendant `secs` secondes et ecrit sur stdout une ligne JSON
/// toutes les ~50 ms (epochs cumulees, entrainements termines, MSE courante) pour l'interface en direct.
fn bench<M: Model>(secs: f64, fit: impl Fn(Snap) -> M) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let ms = || start.elapsed().as_secs_f64() * 1000.0;
    let emit = |t: f64, epochs: usize, runs: usize, loss: f64| {
        println!(r#"{{"t":{t},"epochs":{epochs},"runs":{runs},"loss":{loss}}}"#);
    };
    let (mut done, mut prev, mut runs, mut loss, mut last_emit) = (0, 0, 0, 0.0, f64::MIN);
    let mut progress = |epoch: usize, m: &dyn Model| {
        if epoch < prev {
            // nouvel entrainement : le precedent est termine
            done += prev;
            runs += 1;
        }
        prev = epoch;
        loss = m.history().last().copied().unwrap_or(loss);
        let t = ms();
        if t - last_emit >= 50.0 {
            last_emit = t;
            emit(t, done + epoch, runs, loss);
        }
    };
    while ms() < secs * 1000.0 {
        fit(Some(&mut progress));
    }
    emit(ms(), done + prev, runs + 1, loss);
    Ok(())
}

/// Entraine, chronometre (mediane de RUNS) et exporte `results/<name>_rust.json`.
/// `extra` ajoute des champs JSON propres au modele (ex. `"bias":1.0`).
pub fn run<M: Model>(
    name: &str,
    fit: impl Fn(&[Vec<f64>], &[f64], Snap) -> M,
    extra: impl Fn(&M) -> Vec<String>,
) -> Result<(), Box<dyn Error>> {
    let (x, y) = load_data()?;
    let mut idx: Vec<usize> = (0..y.len()).collect();
    Lcg::new(SEED).shuffle(&mut idx);
    let (test, train) = idx.split_at((y.len() as f64 * TEST_RATIO) as usize);

    // Standardisation calculee uniquement sur le jeu d'entrainement
    let (d, nt) = (FEATURES.len(), train.len() as f64);
    let mean: Vec<f64> = (0..d).map(|j| train.iter().map(|&i| x[i][j]).sum::<f64>() / nt).collect();
    let std: Vec<f64> = (0..d)
        .map(|j| (train.iter().map(|&i| (x[i][j] - mean[j]).powi(2)).sum::<f64>() / nt).sqrt())
        .collect();
    let xs: Vec<Vec<f64>> = x.iter().map(|r| (0..d).map(|j| (r[j] - mean[j]) / std[j]).collect()).collect();
    let pick = |ids: &[usize]| -> (Vec<Vec<f64>>, Vec<f64>) {
        (ids.iter().map(|&i| xs[i].clone()).collect(), ids.iter().map(|&i| y[i]).collect())
    };
    let (x_train, y_train) = pick(train);
    let (x_test, y_test) = pick(test);
    let predict = |m: &dyn Model, xp: &[Vec<f64>]| -> Vec<f64> { xp.iter().map(|r| m.predict(r)).collect() };

    let args: Vec<String> = std::env::args().collect();
    if let Some(secs) = args.iter().position(|a| a == "--bench").and_then(|i| args.get(i + 1)) {
        return bench(secs.parse()?, |snap| fit(&x_train, &y_train, snap));
    }

    let mut snapshots = Vec::new();
    let mut snap = |epoch: usize, m: &dyn Model| {
        snapshots.push(format!(
            r#"{{"epoch":{epoch},"train":{},"test":{}}}"#,
            json_array(&predict(m, &x_train)),
            json_array(&predict(m, &x_test))
        ));
    };
    let model = fit(&x_train, &y_train, Some(&mut snap));

    let mut times: Vec<f64> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            fit(&x_train, &y_train, None);
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    let runs_ms = json_array(&times);
    times.sort_by(f64::total_cmp);
    let train_ms = times[RUNS / 2];

    let mut fields = vec![
        format!(r#""model":"{name}","language":"Rust","train_ms":{train_ms},"runs_ms":{runs_ms}"#),
        format!(r#""loss":{},"snapshots":[{}]"#, json_array(model.history()), snapshots.join(",")),
    ];
    println!("[{name}] Rust");
    for (part, xp, yp) in [("train", &x_train, &y_train), ("test", &x_test, &y_test)] {
        let pred = predict(&model, xp);
        let (rmse, r2) = metrics(yp, &pred);
        println!("  {part:5} RMSE = {rmse:>8.0} $ | R2 = {r2:.4}");
        fields.push(format!(
            r#""{part}":{{"rmse":{rmse},"r2":{r2},"actual":{},"predicted":{}}}"#,
            json_array(yp),
            json_array(&pred)
        ));
    }
    println!("  temps median ({RUNS} runs) = {train_ms:.2} ms\n");
    fields.extend(extra(&model));
    fs::create_dir_all(RESULTS_DIR)?;
    fs::write(format!("{RESULTS_DIR}/{name}_rust.json"), format!("{{{}}}", fields.join(",")))?;
    Ok(())
}
