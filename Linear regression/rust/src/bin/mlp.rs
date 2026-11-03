//! Deep learning : perceptron multicouche (13 -> 32 -> 16 -> 1, ReLU, Adam), from scratch en Rust.
//!
//! Optimisations :
//! - le lot est stocke "par variable" et calcule par les noyaux SIMD a tuiles de `ml_models::simd` ;
//! - le lot est decoupe en paquets traites en parallele par des threads persistants (1 par coeur) ;
//!   chaque thread garde sa copie des poids, somme les gradients publies par tous les paquets et
//!   applique Adam lui-meme (calcul identique => poids identiques) : 1 seule barriere par epoch,
//!   aucune phase sequentielle ou les autres threads attendraient.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use std::thread;

use ml_models::simd::{grad, matmul, sum, to_columns, LANES};
use ml_models::{json_array, record, run, Lcg, Model, Snap, SEED};

const LAYERS: [usize; 4] = [13, 32, 16, 1];
const LEARNING_RATE: f64 = 0.001;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const EPOCHS: usize = 1000;
const FRAME_EVERY: usize = 20; // etat des poids exporte pour la vue 3D du reseau

#[derive(Clone)]
struct Mlp {
    w: Vec<Vec<f64>>, // couche l : matrice (in x out) a plat, ligne par ligne
    b: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
    history: Vec<f64>,
    frames: Vec<String>, // poids et biais (JSON) au fil des epochs, pour la vue 3D
}

/// Un vecteur de zeros par couche : `sizes` = tailles voulues.
fn zeros(sizes: impl Iterator<Item = usize>) -> Vec<Vec<f64>> {
    sizes.map(|k| vec![0.0; k]).collect()
}

impl Mlp {
    fn frame(&mut self, epoch: usize) {
        let layers = |m: &[Vec<f64>]| m.iter().map(|v| json_array(v)).collect::<Vec<_>>().join(",");
        self.frames.push(format!(r#"{{"epoch":{epoch},"w":[{}],"b":[{}]}}"#, layers(&self.w), layers(&self.b)));
    }

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
}

impl Shard {
    fn new(x: &[Vec<f64>], ys: &[f64]) -> Self {
        let (cols, n) = to_columns(x);
        let mut acts = zeros(LAYERS.iter().map(|k| k * n));
        acts[0] = cols;
        let deltas = zeros(LAYERS[1..].iter().map(|k| k * n));
        Shard { n, ys: ys.to_vec(), acts, deltas }
    }

    /// Ecrit dans `grads` (poids puis biais) les gradients partiels de la MSE (normalisee par n_total)
    /// sur ce paquet ; renvoie la somme des erreurs au carre.
    fn backprop(&mut self, model: &Mlp, n_total: f64, grads: &mut [Vec<f64>]) -> f64 {
        let (n, last) = (self.n, model.w.len() - 1);
        model.forward(n, &mut self.acts);
        let mut sse = 0.0;
        for ((d, out), t) in self.deltas[last].iter_mut().zip(&self.acts[last + 1]).zip(&self.ys) {
            let err = out - t;
            sse += err * err;
            *d = 2.0 / n_total * err;
        }
        let (gw, gb) = grads.split_at_mut(model.w.len());
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
        sse
    }
}

/// Mise a jour Adam des poids puis biais (memes ordres que `grads`, `m`, `v`).
fn adam(model: &mut Mlp, grads: &[Vec<f64>], m: &mut [Vec<f64>], v: &mut [Vec<f64>], epoch: usize) {
    let t = (epoch + 1) as i32;
    let (step, inv_c2) = (LEARNING_RATE / (1.0 - BETA1.powi(t)), 1.0 / (1.0 - BETA2.powi(t)));
    let params = model.w.iter_mut().chain(model.b.iter_mut());
    for (((p, g), m), v) in params.zip(grads).zip(m).zip(v) {
        for (((p, g), m), v) in p.iter_mut().zip(g).zip(m.iter_mut()).zip(v.iter_mut()) {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            *p -= step * *m / ((*v * inv_c2).sqrt() + EPS);
        }
    }
}

/// Gradients publies par un paquet pour une epoch, avec sa somme d'erreurs au carre.
type Published = RwLock<(Vec<Vec<f64>>, f64)>;

fn fit(x: &[Vec<f64>], y: &[f64], mut snap: Snap) -> Mlp {
    let pairs: Vec<(usize, usize)> = LAYERS.into_iter().zip(LAYERS.into_iter().skip(1)).collect();
    let mut rng = Lcg::new(SEED + 2);
    // Init He uniforme, meme sequence aleatoire que la version Python
    let w = pairs.iter().map(|&(i, o)| (0..i * o).map(|_| rng.uniform() * (6.0 / i as f64).sqrt()).collect()).collect();
    let nf = y.len() as f64;
    let y_mean = y.iter().sum::<f64>() / nf;
    let y_std = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / nf).sqrt(); // cible standardisee
    let ys: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_std).collect();
    let model = Mlp { w, b: zeros(LAYERS[1..].iter().copied()), y_mean, y_std, history: Vec::with_capacity(EPOCHS), frames: Vec::new() };

    // 1 paquet par coeur (taille multiple de LANES pour le SIMD)
    let sizes = || pairs.iter().map(|&(i, o)| i * o).chain(LAYERS[1..].iter().copied());
    let cores = std::env::var("ML_THREADS").ok().and_then(|c| c.parse().ok()) // limite optionnelle (test en direct)
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, |c| c.get()))
        .max(1);
    let chunk = y.len().div_ceil(cores).next_multiple_of(LANES);
    let shards: Vec<Shard> = x.chunks(chunk).zip(ys.chunks(chunk)).map(|(xc, yc)| Shard::new(xc, yc)).collect();
    // Double tampon : l'epoch e publie dans le tampon e % 2 pendant que les retardataires lisent
    // encore l'autre ; un paquet ne reecrit un tampon qu'apres la barriere suivante => 1 barriere suffit.
    let published: Vec<[Published; 2]> =
        shards.iter().map(|_| [(); 2].map(|_| RwLock::new((zeros(sizes()), 0.0)))).collect();
    let barrier = SpinBarrier::new(shards.len());

    let train = |t: usize, mut shard: Shard, mut model: Mlp, snap: &mut Snap| -> Mlp {
        let (mut total, mut m, mut v) = (zeros(sizes()), zeros(sizes()), zeros(sizes()));
        for epoch in 0..EPOCHS {
            record(snap, epoch, EPOCHS, &model);
            if snap.is_some() && epoch % FRAME_EVERY == 0 {
                model.frame(epoch);
            }
            let buf = epoch % 2;
            {
                let mut out = published[t][buf].write().unwrap();
                let (grads, sse) = &mut *out;
                *sse = shard.backprop(&model, nf, grads);
            }
            barrier.wait();
            total.iter_mut().for_each(|g| g.fill(0.0));
            let mut sse = 0.0;
            for shard_out in &published {
                let shard_out = shard_out[buf].read().unwrap();
                sse += shard_out.1;
                for (g, sg) in total.iter_mut().zip(&shard_out.0) {
                    g.iter_mut().zip(sg).for_each(|(g, s)| *g += s);
                }
            }
            model.history.push(sse / nf * y_std * y_std);
            adam(&mut model, &total, &mut m, &mut v, epoch);
        }
        model
    };

    let model = thread::scope(|scope| {
        let mut shards = shards.into_iter();
        let first = shards.next().unwrap();
        for (t, shard) in shards.enumerate() {
            let (train, model) = (&train, model.clone());
            scope.spawn(move || train(t + 1, shard, model, &mut None));
        }
        train(0, first, model, &mut snap)
    });
    let mut model = model;
    if snap.is_some() {
        model.frame(EPOCHS);
    }
    record(&mut snap, EPOCHS, EPOCHS, &model);
    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run("mlp", fit, |m| {
        let net = format!(r#""net":{{"y_mean":{},"y_std":{},"frames":[{}]}}"#, m.y_mean, m.y_std, m.frames.join(","));
        vec![format!(r#""layers":{LAYERS:?}"#), net]
    })
}
