# Data card: 10X PBMC downsampled-class track (real groups)

Datasheet format after Gebru et al. (CACM 2021). Documents tinybench's use of a public data set.

## Motivation
Real rare cell populations in single-cell RNA-seq. No class of this set is rare (smallest 135 cells), so one class is
downsampled to a handful of real cells (semi-synthetic but every point is a real cell; common practice in rare-cell
papers).

## Composition
- **Source.** 10X Genomics PBMC, as packaged by the scDeepCluster repository (file `10X_PBMC.h5`: X = counts,
  4,271 cells × 16,653 genes; Y = 8 cell-type labels with 1,292 / 702 / 606 / 459 / 450 / 332 / 295 / 135 cells).
- **Scene.** All cells of 7 classes + m real cells of the chosen class (m = 8 or 16; seeds 0–2; all 8 classes).
- **Groups.** The downsampled class. Labels are the packaged annotations (cluster-derived, so partly circular).
- **Known issues.** Downsampled classes are real cells but their density is artificially low; real rare populations
  may be more or less compact. Annotations come from clustering. Real rare cells mix with neighbours (median kNN purity
  0.15), which is why planted groups looked far easier.

## Preprocessing
Normalise total to 1e4, log1p, 2,000 highly variable genes (scanpy "seurat" flavour), scale (max 10), PCA 50.
`pbmc10x_counts` gives the raw count scene for count-based methods (scCAD, the full GapClust pipeline).

## Uses
Real-group track; host for the realistic-plant track; donor of 8 shapes to the boot shape library.

## Distribution
Not redistributed: the user downloads the file and passes its path (`TINYBENCH_PBMC10X`). Original data: 10x Genomics
public data sets (CC BY 4.0). Our generated scenes and offsets derived from it: CC BY 4.0 (see LICENSE).
