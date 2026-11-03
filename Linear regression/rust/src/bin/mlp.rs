//! Deep learning : perceptron multicouche (13 -> 32 -> 16 -> 1, ReLU, Adam), from scratch en Rust.
//!
//! Optimisations :
//! - le lot est stocke "par variable" et calcule par les noyaux SIMD a tuiles de `ml_models::simd` ;
//! - le lot est decoupe en paquets traites en parallele par des threads persistants (1 par coeur),
//!   synchronises par 2 barrieres par epoch ; les gradients partiels sont sommes avant Adam.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};
use std::thread;

use ml_models::simd::{grad, matmul, sum, to_columns, LANES};
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
    /// Propagation avant d'un lot de n exemples ; acts[0] contient deja l'entree.
    fn forward(&self, n: usize, acts: &mut [Vec<f64>]) {
        for l in 0..self.w.len() {
            let (prev, next) = acts.split_at_mut(l + 1);
            let (w, b, out) = (&self.w[l], &self.b[l], self.b[l].len());
            matmul(&mut next[0], &prev[l], n, |o| b[o], |o, i| w[i * out + o]);
            if l < self.w.len() - 1 {
                next[0].iter_mut().for_each(|v| *v = v.max(0.0)); // ReLU
            }
        }
    }
}

impl Model for Mlp {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut acts = zeros(LAYERS.iter().map(|k| k * LANES)); // 1 exemple + remplissage SIMD
        x.iter().enumerate().for_each(|(j, v)| acts[0][j * LANES] = *v);
        self.forward(LANES, &mut acts);
        acts[LAYERS.len() - 1][0] * self.y_std + self.y_mean
    }
    fn history(&self) -> &[f64] {
        &self.history
    }
}

/// Barriere a attente active : latence de l'ordre de la microseconde, la ou `std::sync::Barrier`
/// (endormissement / reveil des threads) couterait plus qu'une epoch entiere.
/// Si l'attente se prolonge (CPU sature), cede le coeur pour ne pas bloquer les autres threads.
struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    fn new(n: usize) -> Self {
        SpinBarrier { n, count: AtomicUsize::new(0), generation: AtomicUsize::new(0) }
    }
    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.generation.load(Ordering::Acquire) == generation {
                spins += 1;
                if spins < 20_000 {
                    std::hint::spin_loop();
                } else {
                    thread::yield_now();
                }
            }
        }
    }
}

/// Paquet d'exemples traite par un thread, avec ses buffers alloues une seule fois.
struct Shard {
    n: usize, // nombre d'exemples arrondi au multiple de LANES (remplissage : delta nul)
    ys: Vec<f64>,
    acts: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
    grads: Vec<Vec<f64>>, // poids puis biais, meme ordre que les moments Adam
    sse: f64,
}

impl Shard {
    fn new(x: &[Vec<f64>], ys: &[f64], grad_sizes: impl Iterator<Item = usize>) -> Self {
        let (cols, n) = to_columns(x);
        let mut acts = zeros(LAYERS.iter().map(|k| k * n));
        acts[0] = cols;
        let deltas = zeros(LAYERS[1..].iter().map(|k| k * n));
        Shard { n, ys: ys.to_vec(), acts, deltas, grads: zeros(grad_sizes), sse: 0.0 }
    }

    /// Gradients partiels de la MSE (normalisee par n_total) sur ce paquet.
    fn backprop(&mut self, model: &Mlp, n_total: f64) {
        let (n, last) = (self.n, model.w.len() - 1);
        model.forward(n, &mut self.acts);
        self.sse = 0.0;
        for ((d, out), t) in self.deltas[last].iter_mut().zip(&self.acts[last + 1]).zip(&self.ys) {
            let err = out - t;
            self.sse += err * err;
            *d = 2.0 / n_total * err;
        }
        let (gw, gb) = self.grads.split_at_mut(model.w.len());
        for l in (0..=last).rev() {
            // retropropagation : gradients de la couche l, puis delta de la couche precedente
            let (lo, hi) = self.deltas.split_at_mut(l);
            let (d, w, out) = (&hi[0], &model.w[l], model.b[l].len());
            grad(&mut gw[l], &self.acts[l], d, n);
            for (g, dv) in gb[l].iter_mut().zip(d.chunks_exact(n)) {
                *g = sum(dv);
            }
            if l > 0 {
                matmul(&mut lo[l - 1], d, n, |_| 0.0, |i, o| w[i * out + o]);
                lo[l - 1].iter_mut().zip(&self.acts[l]).for_each(|(p, a)| if *a <= 0.0 { *p = 0.0 });
            }
        }
    }
}

fn fit(x: &[Vec<f64>], y: &[f64], mut snap: Snap) -> Mlp {
    let pairs: Vec<(usize, usize)> = LAYERS.into_iter().zip(LAYERS.into_iter().skip(1)).collect();
    let mut rng = Lcg::new(SEED + 2);
    // Init He uniforme, meme sequence aleatoire que la version Python
    let w = pairs.iter().map(|&(i, o)| (0..i * o).map(|_| rng.uniform() * (6.0 / i as f64).sqrt()).collect()).collect();
    let nf = y.len() as f64;
    let y_mean = y.iter().sum::<f64>() / nf;
    let y_std = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / nf).sqrt(); // cible standardisee
    let ys: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_std).collect();
    let model = RwLock::new(Mlp { w, b: zeros(LAYERS[1..].iter().copied()), y_mean, y_std, history: Vec::with_capacity(EPOCHS) });

    // 1 paquet par coeur (taille multiple de LANES pour le SIMD)
    let sizes = || pairs.iter().map(|&(i, o)| i * o).chain(LAYERS[1..].iter().copied());
    let cores = std::env::var("ML_THREADS").ok().and_then(|c| c.parse().ok()) // limite optionnelle (test en direct)
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, |c| c.get()))
        .max(1);
    let chunk = y.len().div_ceil(cores).next_multiple_of(LANES);
    let shards: Vec<Mutex<Shard>> =
        x.chunks(chunk).zip(ys.chunks(chunk)).map(|(xc, yc)| Mutex::new(Shard::new(xc, yc, sizes()))).collect();
    let (start, done) = (SpinBarrier::new(shards.len()), SpinBarrier::new(shards.len()));
    let (mut grads, mut m, mut v) = (zeros(sizes()), zeros(sizes()), zeros(sizes()));

    thread::scope(|scope| {
        for shard in &shards[1..] {
            let (model, start, done) = (&model, &start, &done);
            scope.spawn(move || {
                for _ in 0..EPOCHS {
                    start.wait();
                    shard.lock().unwrap().backprop(&model.read().unwrap(), nf);
                    done.wait();
                }
            });
        }
        for epoch in 0..EPOCHS {
            record(&mut snap, epoch, EPOCHS, &*model.read().unwrap());
            start.wait();
            shards[0].lock().unwrap().backprop(&model.read().unwrap(), nf);
            done.wait();

            let mut model = model.write().unwrap();
            grads.iter_mut().for_each(|g| g.fill(0.0));
            let mut sse = 0.0;
            for shard in &shards {
                let shard = shard.lock().unwrap();
                sse += shard.sse;
                for (g, sg) in grads.iter_mut().zip(&shard.grads) {
                    g.iter_mut().zip(sg).for_each(|(g, s)| *g += s);
                }
            }
            model.history.push(sse / nf * y_std * y_std);

            // Adam
            let t = (epoch + 1) as i32;
            let (step, inv_c2) = (LEARNING_RATE / (1.0 - BETA1.powi(t)), 1.0 / (1.0 - BETA2.powi(t)));
            let Mlp { w, b, .. } = &mut *model;
            for (((p, g), m), v) in w.iter_mut().chain(b.iter_mut()).zip(&grads).zip(&mut m).zip(&mut v) {
                for (((p, g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
                    *m = BETA1 * *m + (1.0 - BETA1) * g;
                    *v = BETA2 * *v + (1.0 - BETA2) * g * g;
                    *p -= step * *m / ((*v * inv_c2).sqrt() + EPS);
                }
            }
        }
    });
    let model = model.into_inner().unwrap();
    record(&mut snap, EPOCHS, EPOCHS, &model);
    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("mlp", fit, |_| vec![format!(r#""layers":{LAYERS:?}"#)])
}
