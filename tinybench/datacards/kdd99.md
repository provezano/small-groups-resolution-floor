# Data card: KDD Cup 99 rare-type track (real groups)

Datasheet format after Gebru et al. (CACM 2021). This card documents how tinybench USES the public data set; the
original documentation is the UCI KDD archive page.

## Motivation
Real small groups beside real near-duplicate masses: the 12 rare attack types sit next to smurf / neptune floods and
normal traffic, which is the touching / inside setting of the synthetic tiers, on real records.

## Composition
- **Source.** KDD Cup 1999, 10% file (494,021 records, 41 features, 23 classes), via `sklearn.datasets.fetch_kddcup99`.
- **Scene per seed.** All records of the rare types (spy 2, perl 3, phf 4, multihop 7, ftp_write 8, loadmodule 9,
  rootkit 10, imap 12, warezmaster 20, land 21, buffer_overflow 30, guess_passwd 53) + normal 1,500, smurf 600,
  neptune 600, 300 mid-size attack records in proportion; about 3,180 points.
- **Groups.** The rare types with 2–60 records (12 groups). Labels are the original attack labels.
- **Known issues.** Many exact duplicate records (dithered with 1e-3 × median non-zero 1-NN distance; stated). The data
  set is old (1998 DARPA simulation) and its flaws are documented (McHugh 2000; Tavallaee et al. 2009). Several
  "false" groups found by methods are pure subtypes of normal traffic, so report group purity next to rare-label
  precision.

## Preprocessing
One-hot protocol / service / flag, signed log1p, standardise, PCA 20 fit on the scene, dither.

## Uses
Real-group track (recall of the rare types, false small groups per 100 points, purity). Also a host for the
realistic-plant track (plants into the real cloud).

## Distribution
Not redistributed: downloaded by scikit-learn from the original host on first use. Original licence: UCI ML Repository
(CC BY 4.0). Seeds: 0–4. Our generated scenes and offsets derived from it: CC BY 4.0 (see LICENSE).
