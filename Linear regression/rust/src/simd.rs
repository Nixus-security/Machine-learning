//! Noyaux de calcul vectorises (SIMD) sans dependance.
//!
//! Les lots sont stockes "par variable" : `m[j * n + s]` = variable j de l'exemple s, avec n multiple
//! de `LANES`. Chaque boucle interne traite LANES exemples contigus, que le compilateur traduit en une
//! instruction SIMD (AVX-512 : 8 x f64). Les tuiles de sortie restent en registres pendant tout le
//! calcul, ce qui reduit les acces memoire par multiplication-addition.

use std::array::from_fn;

/// Exemples traites par instruction SIMD (8 x f64 = 512 bits).
pub const LANES: usize = 8;
const TILE: usize = 4;

/// Convertit un lot "par exemple" (x[s][j]) en "par variable", n arrondi au multiple de LANES
/// superieur ; les exemples de remplissage valent 0. Renvoie (matrice, n arrondi).
pub fn to_columns(x: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let n = x.len().next_multiple_of(LANES);
    let mut out = vec![0.0; x.first().map_or(0, Vec::len) * n];
    for (s, row) in x.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[j * n + s] = *v;
        }
    }
    (out, n)
}

fn lanes(m: &[f64], row: usize, n: usize, s: usize) -> &[f64; LANES] {
    m[row * n + s..][..LANES].try_into().unwrap()
}

/// Produit scalaire (longueur multiple de LANES), LANES accumulateurs independants.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0; LANES];
    for (x, y) in a.chunks_exact(LANES).zip(b.chunks_exact(LANES)) {
        for t in 0..LANES {
            acc[t] += x[t] * y[t];
        }
    }
    acc.iter().sum()
}

/// Somme (longueur multiple de LANES), LANES accumulateurs : vectorisee, contrairement a `iter().sum()`.
pub fn sum(a: &[f64]) -> f64 {
    let mut acc = [0.0; LANES];
    for x in a.chunks_exact(LANES) {
        for t in 0..LANES {
            acc[t] += x[t];
        }
    }
    acc.iter().sum()
}

/// y[r][s] = init(r) + somme_k m(r, k) * a[k][s]   (y : lignes de n exemples).
pub fn matmul(y: &mut [f64], a: &[f64], n: usize, init: impl Fn(usize) -> f64, m: impl Fn(usize, usize) -> f64) {
    let rows = y.len() / n;
    let full = rows - rows % TILE;
    // bloc d'exemples a l'exterieur : la colonne de `a` lue (k x LANES valeurs) reste en cache L1
    for s in (0..n).step_by(LANES) {
        for r0 in (0..full).step_by(TILE) {
            matmul_tile::<TILE>(y, a, n, s, r0, &init, &m);
        }
        for r0 in full..rows {
            matmul_tile::<1>(y, a, n, s, r0, &init, &m);
        }
    }
}

fn matmul_tile<const R: usize>(y: &mut [f64], a: &[f64], n: usize, s: usize, r0: usize, init: &impl Fn(usize) -> f64, m: &impl Fn(usize, usize) -> f64) {
    let mut acc: [[f64; LANES]; R] = from_fn(|r| [init(r0 + r); LANES]);
    for k in 0..a.len() / n {
        let ak = lanes(a, k, n, s);
        for (r, acc) in acc.iter_mut().enumerate() {
            let w = m(r0 + r, k);
            for t in 0..LANES {
                acc[t] += w * ak[t];
            }
        }
    }
    for (r, acc) in acc.iter().enumerate() {
        y[(r0 + r) * n + s..][..LANES].copy_from_slice(acc);
    }
}

/// g[i * outs + o] = somme_s a[i][s] * d[o][s]   (gradient des poids d'une couche).
pub fn grad(g: &mut [f64], a: &[f64], d: &[f64], n: usize) {
    let (ins, outs) = (a.len() / n, d.len() / n);
    let full = ins - ins % TILE;
    // tuile de `a` a l'exterieur : ses TILE lignes restent en cache pendant le parcours des sorties
    for i0 in (0..full).step_by(TILE) {
        (0..outs).for_each(|o| grad_tile::<TILE>(g, a, d, n, i0, o));
    }
    for i in full..ins {
        (0..outs).for_each(|o| grad_tile::<1>(g, a, d, n, i, o));
    }
}

/// Tuile de R entrees x 1 sortie : R accumulateurs SIMD (tuiles plus grandes : moins bien vectorisees).
fn grad_tile<const R: usize>(g: &mut [f64], a: &[f64], d: &[f64], n: usize, i0: usize, o: usize) {
    let outs = d.len() / n;
    let mut acc = [[0.0; LANES]; R];
    for s in (0..n).step_by(LANES) {
        let dv = lanes(d, o, n, s);
        for (x, acc) in acc.iter_mut().enumerate() {
            let av = lanes(a, i0 + x, n, s);
            for t in 0..LANES {
                acc[t] += av[t] * dv[t];
            }
        }
    }
    for (x, acc) in acc.iter().enumerate() {
        g[(i0 + x) * outs + o] = acc.iter().sum();
    }
}
