"""The shared HDBSCAN base every group method writes its small groups over (vendored from
clustering.eval.nature.tiny_methods.hdb / methods.hdbscan_labels): min_cluster_size = max(8, n // 250),
min_samples = max(3, mcs // 2), excess-of-mass selection."""
import numpy as np


def _mcs(n):
    return max(8, n // 250)


def hdbscan_fit(Z, mcs, ms):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="eom",
                           cluster_selection_epsilon=0.0, allow_single_cluster=False).fit(np.asarray(Z, np.float64))


def hdb(Z):
    n = len(Z)
    return hdbscan_fit(Z, _mcs(n), max(3, _mcs(n) // 2)).labels_


def groups_over_base(Z, groups, base=None, max_size=None):
    """Write each group (index array, >= 2 points) as a new cluster over the base labels."""
    lab = (hdb(Z) if base is None else base).copy()
    nxt = lab.max() + 1
    for g in groups:
        g = np.asarray(g)
        if len(g) >= 2 and (max_size is None or len(g) <= max_size):
            lab[g] = nxt; nxt += 1
    return lab
