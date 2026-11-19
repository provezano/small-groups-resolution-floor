# tinybench

A benchmark for methods that find **tiny groups** (2–15 points) in unlabelled data.

- A deterministic scene generator with ground truth: 9 tiers (placement × contrast), d = 2 / 5 / 10, and four group
  shapes (isotropic, anisotropic, heavy-tailed, real-shape bootstrap).
- One score: recall at **matched false-group rates** (≤ 0.25 and ≤ 0.5 false small groups per 100 points), with
  bootstrap confidence intervals and a paired test for "A beats B on tier T".
- A per-tier **ceiling**: the oracle local scan that knows the true background density.
- **Real-group tracks** (KDD Cup 99 rare attack types, 10X PBMC downsampled classes) and a **realistic-plant track**
  (groups planted into real point clouds with the recipe closest to real groups).
- Baselines behind one signature, `fit(X, param) -> labels`.

Licence: code MIT; generated benchmark data and offset library CC BY 4.0; data derived from KDD Cup 99 / UCI and 10X Genomics PBMC keeps its upstream terms (see `LICENSE` and the data cards).

## Install

```
pip install -e .            # core: numpy, scipy, scikit-learn, joblib, hdbscan
pip install -e ".[real]"    # + KDD Cup 99 track (pandas)
pip install -e ".[singlecell]"   # + 10X PBMC track (h5py, anndata, scanpy)
pip install -e ".[test]" && pytest tests
```

Tested: Python 3.11, fresh `venv`, macOS arm64.

## Quickstart

```python
import numpy as np
import tinybench as tb

scene = tb.generate("touching", contrast="mid", d=5, seed=500)       # X (n x 20), kind, gid
labels = my_method(scene["X"])                                        # -1 = noise
print(tb.score(labels, scene["kind"], scene["gid"]))                 # recall, false_per100, ...

# frontier over a parameter grid, on the frozen test seeds, with 95% CIs
def fit(X, threshold):
    return my_method(X, threshold)
res = tb.frontier(fit, grid=[0.01, 0.05, 0.3], seeds=tb.seeds.seeds("test", "1.0"))
print(res["touching/mid"])        # {'recall@0.25': ..., 'ci@0.25': [lo, hi], 'recall@0.5': ..., ...}

ceil = tb.ceiling(seeds=tb.seeds.seeds("test", "1.0"))               # oracle scan, per tier

fit_gc, grid = tb.baselines.get("gapclust_k30")                       # a baseline, same signature
real = tb.load_real("kdd99", seed=0)                                  # X, y, groups (real rare types)
print(tb.score_real(fit_gc(real["X"], 2.0), real["groups"]))
```

Command line:

```
tinybench generate --placement inside --contrast low --d 10 --seed 500 --shape boot --out scene.npz
tinybench frontier --baseline lof_group --seeds 500-504
tinybench frontier --method mypkg.mymod:fit --param-grid "0.01,0.05,0.3" --seeds 500-504 --out mine.json
tinybench ceiling --seeds 500-504
tinybench leaderboard results/leaderboard_v1.0_s500_cap0.25.json --out LEADERBOARD.md
```

## Rules for a leaderboard entry
1. Tune only on dev seeds (`seeds("dev")`). Report on test seeds, once.
2. Report the whole frontier (both caps), not one operating point.
3. Report CIs. Claim "A beats B on tier T" only when the paired test survives Holm correction over tiers.
4. Report the oracle ceiling next to your numbers.
5. Report at least one real-group track next to any synthetic number. Planted isotropic groups inflated recall
   5–7× on single-cell.

## What is in the box

| Module | What |
|---|---|
| `generator` | `make`, `generate`; shapes iso / aniso / t3 / boot |
| `scoring` | `score` (Jaccard ≥ 0.5 recall, false small groups per 100 points, mass pieces, noise) |
| `core` | `frontier`, `ceiling` (with bootstrap CIs) |
| `oracle` | the oracle local scan (true background density) |
| `stats` | bootstrap CIs, exact paired permutation test, Holm |
| `baselines` | HDBSCAN, LOF/IF/GLOSH + grouping, GapClust core port, McCatch reimplementation; optional scCAD, CTS |
| `real` | `load_real("kdd99" / "pbmc10x" / "pbmc10x_counts")`, `score_real` |
| `tracks` | planting recipes for real clouds; `RECOMMENDED` = the recipe closest to real groups |
| `seeds` | frozen, versioned seed lists |
| `leaderboard` | Markdown table from a results JSON |

Data cards (datasheet format): `datacards/`.

## Provenance and honesty notes
- The generator, scoring, oracle and baseline wrappers are copies of the research code used for the paper (same
  project); a test pins `shape="iso"` to it bit for bit.
- **McCatch is our reimplementation** from the paper; the official code could not be fetched. Not validated.
- **GapClust is not included.** The upstream R package (github.com/fabotao/GapClust) is GPL-3, so a port cannot be distributed under
  this package's MIT licence. The baselines `gapclust_*` are stubs that raise an explanatory error; set `GAPCLUST_PY` to your own
  Python port (it must define `gapclust_raw(Z, k, squared)` and `gapclust_groups(skew, top_ave, v1k, idx, thr)`). The
  leaderboard rows `gapclust_port*` in `results/` were computed with the removed port; they are kept as our measurements but
  cannot be regenerated from this package alone (a licensing constraint, not a missing result).
- **scCAD** is not vendored; the wrapper calls the authors' own `scCAD.py` (set `SCCAD_PY`). Its official code scores
  only sub-clusters of ≥ 10 cells, so it cannot report groups of 2–9 by design.
- **CTS** (the method of the companion paper) is a contestant, not part of the benchmark; its wrapper imports it only
  if installed.
