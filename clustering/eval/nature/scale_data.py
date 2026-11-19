"""
Public data loaders for the stage-9 scale runs. Every file is read as data (CSV / MatrixMarket / ARFF text); nothing
downloaded is executed. Sources:
- KDD Cup 99, 10% file (494,021 records): scikit-learn `fetch_kddcup99(percent10=True)`.
- UNSW-NB15 (54,296-record sample, OpenML id 46301, a public mirror; the official UNSW page links a SharePoint folder
  that could not be fetched non-interactively). Rare attack categories: Worms 24, Backdoor 67, Analysis 192, Shellcode 192.
- Baron et al. 2016 human pancreas (GEO GSE84133, 4 donors, inDrop UMI counts, author cell-type labels). Rare types per
  donor (2-23 cells): epsilon, macrophage, mast, schwann, t_cell, and per donor some endothelial / acinar / stellate.
- 10X Genomics fresh 68k PBMC donor A (official 10X page), labels from the paper's public repository (Zheng et al.
  2017, 68k_pbmc_barcodes_annotation.tsv, 11 types).
- MNIST and Fashion-MNIST (OpenML `mnist_784`, `Fashion-MNIST`), 20 Newsgroups (scikit-learn).
"""
import gzip
import io
import os

import numpy as np

try:
    from . import _tgpaths as _TGP
except ImportError:
    import _tgpaths as _TGP
ROOT = os.environ.get("SCALE_DATA", _TGP.DATA + "/scale")


def kdd_full(seed=0, dims=20, dither=True):
    """All 494,021 records, run_kdd features (one-hot, signed log1p, standardised, PCA 20). dither=True: dithered like
    run_kdd. dither=False: returns (Z, y, key), key = id of each record's raw feature row among the unique rows, for
    scale_cts.cts_scale_dedup."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from .run_kdd import load
    df, y = load()
    X = pd.get_dummies(df, columns=["protocol_type", "service", "flag"]).astype(np.float32).to_numpy()
    key = np.unique(X, axis=0, return_inverse=True)[1].ravel()
    X = StandardScaler().fit_transform(np.log1p(np.abs(X)) * np.sign(X))
    Z = PCA(dims, random_state=seed).fit_transform(X)
    if not dither:
        return Z, y, key
    rng = np.random.default_rng(seed)
    from .scale_cts import knn_exact
    pick = rng.choice(len(Z), 20000, replace=False)
    d1 = knn_exact(Z, 1, query=Z[pick], ids=pick)[0][:, 1]
    Z = Z + rng.normal(0, 1e-3 * np.median(d1[d1 > 0]), Z.shape)
    return Z, y


def unsw():
    import pandas as pd
    path = os.path.join(ROOT, "unsw/UNSW_NB15.arff")
    txt = open(path, encoding="utf-8", errors="replace").read()
    head, body = txt.split("@DATA", 1) if "@DATA" in txt else txt.split("@data", 1)
    names = [l.split()[1].strip('"') if not l.split()[1].startswith('"') else l.split('"')[1]
             for l in head.splitlines() if l.upper().startswith("@ATTRIBUTE")]
    df = pd.read_csv(io.StringIO(body.strip()), header=None, names=names, quotechar="'", low_memory=False)
    y = df.pop("attack_cat").astype(str).str.strip().to_numpy()
    df = df.drop(columns=["Unnamed: 0", "id", "label"])
    return df, y


def unsw_features(df, seed, dims=20):
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = pd.get_dummies(df, columns=["proto", "service", "state"]).astype(float).to_numpy()
    X = StandardScaler().fit_transform(np.log1p(np.abs(X)) * np.sign(X))
    return PCA(dims, random_state=seed).fit_transform(X)


def baron(donor):
    """Counts (cells x genes) and labels for human donor 1-4."""
    import pandas as pd
    f = [x for x in os.listdir(os.path.join(ROOT, "baron")) if f"human{donor}_" in x][0]
    df = pd.read_csv(os.path.join(ROOT, "baron", f), index_col=0)
    y = df.pop("assigned_cluster").astype(str).to_numpy()
    df = df.drop(columns=["barcode"])
    return df.to_numpy(np.float32), y


def pbmc68k():
    """Sparse counts (cells x genes, CSR) and the 11 published labels."""
    import scipy.io
    base = os.path.join(ROOT, "pbmc68k/filtered_matrices_mex/hg19")
    X = scipy.io.mmread(os.path.join(base, "matrix.mtx")).T.tocsr().astype(np.float32)
    bc = [l.strip() for l in open(os.path.join(base, "barcodes.tsv"))]
    lab = {}
    for i, l in enumerate(open(os.path.join(ROOT, "pbmc68k/annot.tsv"))):
        if i:
            p = l.rstrip("\n").split("\t")
            lab[p[2]] = p[3]
    y = np.array([lab.get(b, "NA") for b in bc])
    return X, y


def openml_images(name):
    from sklearn.datasets import fetch_openml
    b = fetch_openml(name, version=1, as_frame=False, parser="liac-arff", data_home=os.path.join(ROOT, "openml"))
    return b.data.astype(np.float32), np.asarray(b.target).astype(str)


def news20():
    from sklearn.datasets import fetch_20newsgroups
    b = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), data_home=os.path.join(ROOT, "news20"))
    return b.data, np.array(b.target_names)[b.target]
