# Data card: realistic-plant track (groups planted into real point clouds)

Datasheet format after Gebru et al. (CACM 2021).

## Motivation
Planted isotropic groups made methods look 5x better than on real rare groups (single-cell). This track plants groups
into the real KDD 99 and 10X PBMC clouds with the recipe whose shape descriptors are closest to real rare groups.

## How the recipe was chosen
- Descriptors per group (spread, anisotropy, kNN purity, isolation), standardised on the real groups.
- Distance = exact Wasserstein-1 between the planted and real descriptor distributions.
- 19 recipes compared (isotropic, local-covariance Gaussian, Student t, absolute and relative offset bootstrap, each of
  the last three also with the centre shifted off the target point by 0.5 / 1 / 2 local radii).

| Recipe | W1 single-cell [95% CI] | W1 KDD [95% CI] |
|---|---|---|
| iso10 (stage-7 default) | 6.54 [6.22, 6.79] | 4.19 [4.03, 4.37] |
| iso50 | 5.58 [5.38, 5.99] | 4.45 [4.20, 4.67] |
| t3loc1.0 | 5.51 [5.18, 5.78] | 1.60 [1.38, 1.88] |
| boot (absolute offsets) | 3.36 [3.31, 3.98] | 300 [6.9, 681] |
| bootrel | 3.97 [3.57, 4.36] | 1.32 [1.29, 1.46] |
| **bootrel@0.5 (recommended)** | **2.06 [1.67, 2.41]** | 1.43 [1.23, 2.08] |
| bootrel@1 | 5.29 [4.86, 5.72] | 1.30 [1.15, 1.59] |

- bootrel@0.5 = offsets of a real rare group of another class, rescaled by the ratio of local 10-NN radii, centred
  half a local radius off an existing point along the local manifold.
- It is the smallest W1 on single-cell in 100 of 100 bootstrap resamples and on both held-out halves. On KDD it is
  within noise of the best (bootrel@1, 1.30). Smallest mean W1 relative to iso10 over both domains (0.33).

## What it does and does not fix
- **Level, single-cell: fixed.** Planted/real recall ratio (mean over 5 methods) goes from 5.1x (iso10) to 0.6x;
  mean absolute recall gap from 0.41 to 0.05.
- **Level, KDD: not fixed.** Isotropic plants were already at the right level on KDD (ratio 1.0x); bootrel@0.5 makes
  plants too hard (0.5x).
- **Rankings: not identified.** Only 3 of 16 real single-cell groups (seed 0) are found by any method, so the real
  ranking is mostly ties; Spearman CIs span 0 to 0.9 for most recipes.

## Use
Report real-group tracks first; the realistic-plant track second; the synthetic tiers third. Never report a
synthetic number alone.

## Composition, licence
Host clouds: see `kdd99.md`, `pbmc10x.md`. 10 plants per scene, sizes from the host's real group sizes. Ground truth =
planted ids. Licence: generated scenes CC BY 4.0, code MIT (see LICENSE); the host clouds keep the upstream terms stated in `kdd99.md` and `pbmc10x.md`.
