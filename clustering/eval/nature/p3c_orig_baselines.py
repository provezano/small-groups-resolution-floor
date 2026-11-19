"""
Original baseline code against our ports, scene by scene, on tiny-bench (register item 3.4).

GapClust: the official R core (p3c_gapclust_core.R: fabotao/GapClust R/GapClust.R sourced unchanged, rflann from
the GitHub repository the GapClust README names) against our Python port of the official form
(p3_gapclust_full.gapclust_raw_knn, squared distances), at k = 200 and 30 and the six thresholds used everywhere.

McCatch: the authors' Java/ELKI code (github.com/BraulioSanchez/McCatch, code/vectorial, compiled unchanged against
ELKI 0.8.0) against our reimplementation (tiny_baselines.mccatch). Both go through the same post-processing as
tiny_baselines.mccatch_sweep: non-singleton microclusters of 2..ceil(0.1 n) points, ordered by McCatch's own score,
top share 1 / 0.75 / 0.5 / 0.25 at b = 0.1, plus all of them at b = 0.05 and 0.2, written over HDBSCAN with groups
of at most 15 points.

For every scene the record holds both outputs, the agreement (identical groups, skewness difference) and the
tiny-bench score dicts of both, so the frontier and the frozen per-scene numbers can be compared directly.

  python -m clustering.eval.nature.p3c_orig_baselines --seeds 620-639 --out-dir <dir> --tools <dir>

Tools dir layout (built by the branch, outside prototype/): Rlib/ (rflann 1.5, e1071, jsonlite), gapclust/GapClust.R,
jdk/bin/java, mccatch_build/ (compiled classes), jars/ (ELKI 0.8.0 modules and dependencies).
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .tiny_bench import TIERS, make, score
from .tiny_baselines import gapclust_groups, groups_over_base, mccatch
from .tiny_methods import hdb
from .p3_gapclust_full import gapclust_raw_knn

CONTRASTS = ("high", "mid", "low")
DIMS = (2, 5, 10)
THR = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
KS = (200, 30)
FRACS = (1.0, 0.75, 0.5, 0.25)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- McCatch, official Java code ----------------
def mccatch_official_raw(Z, tools, a=15, b=0.1, c=0.1, workdir=None):
    """Runs the authors' McCatch (output mode 1: microclusters with their scores). Returns (groups, scores, secs),
    groups as 0-based index arrays, singletons included."""
    java = os.path.join(tools, "jdk", "bin", "java")
    cp = ":".join([os.path.join(tools, "mccatch_build")] + sorted(glob.glob(os.path.join(tools, "jars", "*.jar"))))
    wd = tempfile.mkdtemp(dir=workdir)
    try:
        f = os.path.join(wd, "x.csv")
        np.savetxt(f, np.c_[Z, np.zeros(len(Z))], delimiter=",", fmt="%.17g")
        t = time.time()
        r = subprocess.run([java, "-Xmx1g", "-XX:ActiveProcessorCount=1", "-cp", cp,
                            "elki.application.KDDCLIApplication", "-algorithm", "elki.outlier.distance.McCatch",
                            "-dbc.in", f, "-parser.labelIndices", str(Z.shape[1]), "-mccatch.a", str(a),
                            "-mccatch.b", str(b), "-mccatch.c", str(c), "-mccatch.output", "1"],
                           capture_output=True, text=True, cwd=wd)
        secs = time.time() - t
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    mem, sc = {}, {}
    for line in r.stdout.splitlines():
        if line.startswith("mj"):
            j, i, s = line.split(",")
            mem.setdefault(j, []).append(int(i)); sc[j] = float(s)
    if r.returncode != 0 and not mem:
        raise RuntimeError(r.stderr[-2000:])
    keys = sorted(mem, key=lambda j: int(j[2:]))
    return [np.array(sorted(mem[j])) for j in keys], np.array([sc[j] for j in keys]), secs


def _sweep_from(groups, scores, n, base, extra):
    """mccatch_sweep's post-processing, shared by both implementations."""
    c = int(math.ceil(0.1 * n))
    keep = [i for i, g in enumerate(groups) if 2 <= len(g) <= c]
    groups = [groups[i] for i in keep]; scores = np.asarray(scores)[keep] if len(keep) else np.array([])
    order = np.argsort(-scores) if len(scores) else np.array([], int)
    out = []
    for f in FRACS:
        k = int(math.ceil(f * len(order)))
        out.append(groups_over_base(None, [groups[i] for i in order[:k]], base, 15))
    for g in extra:
        g = [x for x in g if 2 <= len(x) <= c]
        out.append(groups_over_base(None, g, base, 15))
    return out


def mccatch_official_sweep(Z, tools, base=None, workdir=None):
    base = hdb(Z) if base is None else base
    g, s, t0 = mccatch_official_raw(Z, tools, workdir=workdir)
    extra, secs = [], [t0]
    for bb in (0.05, 0.2):
        gb, _, tb = mccatch_official_raw(Z, tools, b=bb, workdir=workdir)
        extra.append(gb); secs.append(tb)
    return _sweep_from(g, s, len(Z), base, extra), {"raw": [g] + extra, "secs": secs}


def mccatch_reimpl_sweep(Z, base=None):
    base = hdb(Z) if base is None else base
    t = time.time()
    g, s, _ = mccatch(Z, return_all=True)
    extra = [mccatch(Z, b=bb) for bb in (0.05, 0.2)]
    return _sweep_from(g, s, len(Z), base, extra), {"raw": [g] + extra, "secs": time.time() - t}


# ---------------- GapClust, official R core ----------------
def gapclust_official_batch(Xs, tools, workdir=None, dump_knn=False):
    """Xs: {scene id: array}. Returns {scene id: {k: {"skew", "groups" (per THR), "secs"}}} from the official R core."""
    wd = tempfile.mkdtemp(dir=workdir)
    try:
        man = {"k": list(KS), "thr": list(THR), "dump_knn": int(dump_knn), "scenes": []}
        for sid, X in Xs.items():
            f = os.path.join(wd, f"{len(man['scenes'])}.bin")
            np.ascontiguousarray(X, dtype="<f8").tofile(f)
            man["scenes"].append({"id": sid, "bin": f, "n": int(X.shape[0]), "p": int(X.shape[1])})
        mf, of = os.path.join(wd, "man.json"), os.path.join(wd, "out.json")
        json.dump(man, open(mf, "w"))
        r = subprocess.run(["Rscript", "--vanilla", os.path.join(HERE, "p3c_gapclust_core.R"),
                            os.path.join(tools, "gapclust", "GapClust.R"), os.path.join(tools, "Rlib"), mf, of],
                           capture_output=True, text=True, cwd=wd)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-3000:])
        res = json.load(open(of))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    out = {}
    for sc in res["scenes"]:
        out[sc["id"]] = {}
        for kr in sc["k"]:
            groups = [[np.array(g if isinstance(g, list) else [g], int) for g in gl] for gl in kr["groups"]]
            out[sc["id"]][kr["k"]] = {"skew": np.array(kr["skew"], float), "groups": groups,
                                      "secs": kr["secs_core"] + kr["secs_select"],
                                      **({"knn_d": np.array(kr["knn_d"]), "knn_i": np.array(kr["knn_i"])}
                                         if "knn_d" in kr else {})}
    return out, res.get("rflann")


def _same(ga, gb):
    return sorted(tuple(sorted(map(int, g))) for g in ga) == sorted(tuple(sorted(map(int, g))) for g in gb)


def one_seed(seed, tools, workdir=None):
    scenes = [(p, c, d) for p in TIERS for c in CONTRASTS for d in DIMS]
    data = {f"{p}/{c}/{d}/{seed}": make(p, d, seed, contrast=c) for p, c, d in scenes}
    t = time.time()
    rg, rflann_ver = gapclust_official_batch({k: v[0] for k, v in data.items()}, tools, workdir)
    r_secs = time.time() - t
    rows = []
    for sid, (X, kind, gid) in data.items():
        p, c, d, s = sid.split("/")
        base = hdb(X)
        row = {"placement": p, "contrast": c, "d": int(d), "seed": int(s), "gap": {}, "mcc": {}}
        for k in KS:
            kk = min(k, len(X) - 1)
            t0 = time.time()
            raw = gapclust_raw_knn(X, kk, squared=True)
            py_groups = [gapclust_groups(*raw, thr=th) for th in THR]
            py_secs = time.time() - t0
            R = rg[sid][kk]
            same = [_same(a, b) for a, b in zip(py_groups, R["groups"])]
            row["gap"][f"k{k}"] = {
                "skew_maxabs": float(np.nanmax(np.abs(raw[0] - R["skew"]))) if len(R["skew"]) == len(raw[0]) else None,
                "same_groups": same, "n_groups_R": [len(g) for g in R["groups"]],
                "score_R": [score(groups_over_base(X, g, base), kind, gid) for g in R["groups"]],
                "score_py": [score(groups_over_base(X, g, base), kind, gid) for g in py_groups],
                "secs_R": R["secs"], "secs_py": py_secs}
        lo, info_o = mccatch_official_sweep(X, tools, base, workdir)
        lr, info_r = mccatch_reimpl_sweep(X, base)
        row["mcc"] = {
            "score_official": [score(l, kind, gid) for l in lo], "score_reimpl": [score(l, kind, gid) for l in lr],
            "same_raw_nonsingleton": [_same([g for g in a if len(g) >= 2], [g for g in b if len(g) >= 2])
                                      for a, b in zip(info_o["raw"], info_r["raw"])],
            "n_mc_official": [sum(len(g) >= 2 for g in a) for a in info_o["raw"]],
            "n_mc_reimpl": [sum(len(g) >= 2 for g in a) for a in info_r["raw"]],
            "n_outliers_official": [int(sum(len(g) for g in a)) for a in info_o["raw"]],
            "secs_official_wall": info_o["secs"], "secs_reimpl": info_r["secs"]}
        rows.append(row)
    return {"seed": seed, "rflann": rflann_ver, "secs_R_batch": r_secs, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="lo-hi or comma list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tools", required=True)
    a = ap.parse_args(argv)
    seeds = (list(range(int(a.seeds.split("-")[0]), int(a.seeds.split("-")[1]) + 1)) if "-" in a.seeds
             else [int(s) for s in a.seeds.split(",")])
    os.makedirs(a.out_dir, exist_ok=True)
    wd = os.path.join(a.out_dir, "tmp"); os.makedirs(wd, exist_ok=True)
    for s in seeds:
        f = os.path.join(a.out_dir, f"seed{s}.json")
        if os.path.exists(f):
            continue
        t = time.time()
        res = one_seed(s, a.tools, wd)
        json.dump(res, open(f + ".part", "w")); os.replace(f + ".part", f); os.chmod(f, 0o600)
        with open(os.path.join(a.out_dir, "STATUS.md"), "a") as st:
            st.write(f"seed {s} done in {time.time() - t:.0f} s\n")


if __name__ == "__main__":
    main()
