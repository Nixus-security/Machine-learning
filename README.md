# Machine learning : Python vs Rust

Ce projet veut montrer que Rust est un candidat sérieux pour les environnements d'IA. Trois modèles d'apprentissage sont codés **from scratch** en Python (NumPy) et en Rust (bibliothèque standard, aucune dépendance), puis comparés sur le même problème : prédire le prix d'une voiture à partir de ses caractéristiques techniques.

Les deux implémentations utilisent les mêmes données, le même découpage, la même initialisation et les mêmes hyperparamètres. Leurs prédictions sont identiques ; seul le temps de calcul change.

## Résultats

Temps d'entraînement médian sur 5 exécutions, sur une VM cloud à 4 vCPU (Xeon 2,1 GHz). Les mesures varient de ±30 % d'un lancement à l'autre sur cette machine.

| Modèle | Python | Rust | Rust plus rapide | R² test |
|---|---|---|---|---|
| Régression linéaire | 66 ms | 5–8 ms | ×8 à ×13 | 0,634 |
| Descente de gradient stochastique | 74 ms | 0,6 ms | ×120 | 0,607 |
| Réseau de neurones | 182 ms | 52 ms | ×3,5 | 0,831 |

L'écart est le plus grand quand Python exécute beaucoup de petites opérations (mini-batchs de 16 exemples) : chaque appel NumPy a un coût fixe que Rust n'a pas. Sur le réseau de neurones, NumPy délègue les produits matriciels à OpenBLAS, une bibliothèque C/Fortran très optimisée, d'où un écart plus faible.

## Les modèles

| Modèle | Fichiers | Détails |
|---|---|---|
| Régression linéaire | `python/linear_regression.py`, `rust/src/bin/linear.rs` | Descente de gradient batch, 5 000 epochs, taux 0,01 |
| Gradient stochastique (SGD) | `python/sgd.py`, `rust/src/bin/sgd.rs` | Mini-batchs de 16, momentum 0,9, 300 epochs, taux 0,01 |
| Réseau de neurones (MLP) | `python/mlp.py`, `rust/src/bin/mlp.rs` | 13 → 32 → 16 → 1, ReLU, Adam, 1 000 epochs, taux 0,001 |

Les données viennent de `data/Automobile_data.csv` : 159 voitures complètes, 13 variables numériques (cylindrée, puissance, poids…) et le prix comme cible. Le split est 80 % entraînement / 20 % test. Les variables sont standardisées avec les statistiques du jeu d'entraînement.

Une **epoch** est un passage complet sur les données d'entraînement. L'unité **ep/s** (epochs par seconde) mesure donc la vitesse d'apprentissage.

## Lancer le projet

Prérequis : Python 3 avec NumPy, et Rust (`cargo`).

```bash
cd "Linear regression"
```

Pour entraîner les 6 modèles et ouvrir le tableau de bord comparatif (`results/dashboard.html`) :

```bash
python dashboard.py
```

Pour lancer le test de charge en direct dans une interface 3D (http://127.0.0.1:8765) :

```bash
python live.py
```

Pour lancer un seul modèle :

```bash
python python/mlp.py
```

```bash
cd rust && cargo run --release --bin mlp
```

## Interfaces

**Tableau de bord** (`dashboard.py`). Il contient :

- une synthèse des accélérations et un tableau comparatif ;
- la relecture de l'entraînement epoch par epoch (prix réel vs prédit, courbe d'erreur) ;
- les temps de chaque exécution, et les poids appris ou l'architecture du réseau.

Le thème clair ou sombre suit celui du système.

**Test en direct 3D** (`live.py` + `live.html`). Python et Rust entraînent le modèle choisi en boucle, en même temps, pendant 5 à 30 s, sur 1 cœur chacun pour un duel équitable. Un petit serveur (bibliothèque standard Python) relaie leur progression en temps réel (Server-Sent Events) vers une scène three.js :

- les rubans montrent le débit instantané en ep/s, sur une échelle logarithmique ;
- les colonnes montrent les epochs cumulées.

## Comment Rust va plus vite

- **Compilation native** : `--release` et `target-cpu=native` (`rust/.cargo/config.toml`) permettent d'utiliser les instructions SIMD du processeur (AVX2 / AVX-512).
- **Données « par variable »** : les lots sont stockés par colonne (`rust/src/simd.rs`), si bien que chaque boucle interne traite 8 exemples contigus en une instruction.
- **Noyaux à tuiles** : les produits matriciels et les gradients gardent leurs accumulateurs en registres et parcourent la mémoire dans l'ordre favorable au cache.
- **Zéro allocation dans la boucle d'entraînement** : tous les buffers sont alloués une seule fois.
- **Multithreading** (réseau de neurones) : le lot est réparti sur les cœurs. Les threads persistants sont synchronisés par une barrière à attente active. Les gradients partiels sont sommés avant la mise à jour Adam. `ML_THREADS` limite le nombre de threads.

## Méthodologie

- Le générateur pseudo-aléatoire (LCG 64 bits) est identique bit à bit dans les deux langages. Le mélange des données et l'initialisation des poids sont donc les mêmes.
- Le temps mesuré exclut le chargement des données et l'enregistrement des états servant à l'animation.
- Python profite de NumPy vectorisé. Comparer à du Python pur serait plus favorable à Rust, mais moins réaliste.

## Structure

```
Linear regression/
├── data/Automobile_data.csv
├── python/            common.py + 1 script par modèle
├── rust/
│   ├── src/lib.rs     données, split, métriques, benchmark, export JSON
│   ├── src/simd.rs    noyaux de calcul vectorisés
│   └── src/bin/       1 binaire par modèle
├── dashboard.py       entraîne tout et génère results/dashboard.html
├── live.py, live.html test de charge en direct (3D)
└── results/           dashboard.html (+ JSON intermédiaires ignorés par git)
```
