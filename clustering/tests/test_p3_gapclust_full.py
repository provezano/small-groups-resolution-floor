"""Checks for the GapClust full pipeline (p3_gapclust_full) against reference values printed by R 4.6.2 /
Seurat 6.0.0 (v3 assay) on small deterministic inputs."""
import numpy as np
import pytest

from clustering.eval.nature import p3_gapclust_full as G
from clustering.eval.nature.tiny_baselines import gapclust_available

pytestmark = pytest.mark.skipif(not gapclust_available(), reason="GapClust port not shipped (GPL-3 upstream); set GAPCLUST_PY to your own port")


def test_r_density_matches_r():
    # R: d <- density(c(0.1, 0.4, 0.45, 0.5, 0.9, 1.3, 1.31, 2.2, 3.5, 0.05, 0.6, 0.62)); d$bw; d$y[c(1,100,256,400,512)]
    x = np.array([0.1, 0.4, 0.45, 0.5, 0.9, 1.3, 1.31, 2.2, 3.5, 0.05, 0.6, 0.62])
    xg, yg, bw = G.r_density(x)
    assert len(xg) == 512
    assert bw == pytest.approx(0.353441522510344, rel=1e-12)
    ref = [0.00179272130766322, 0.410197076417798, 0.129880514177594, 0.0853209419924125, 0.00104738673233014]
    assert yg[[0, 99, 255, 399, 511]] == pytest.approx(ref, rel=1e-9)


def test_find_elbow_simple():
    x = np.linspace(0, 1, 101)
    y = np.where(x < 0.3, 1 - x / 0.3 * 0.9, 0.1 - (x - 0.3) * 0.1)      # sharp knee at x = 0.3
    assert G.find_elbow(x, y) == pytest.approx(0.3, abs=0.011)


def test_seurat_vst_matches_r():
    pytest.importorskip("skmisc")
    # R: M <- outer(0:199, 0:39, function(i, j) ((i*7 + j*13 + (i*j) %% 5) %% 11) * ((i %% 6) + (j %% 3)) %/% 4 *
    #    ((i %% 4) != 0 | (j %% 2) == 0)); FindVariableFeatures(CreateSeuratObject(counts = M), "vst", nfeatures = 200)
    i, j = np.meshgrid(np.arange(200), np.arange(40), indexing="ij")
    M = ((i * 7 + j * 13 + (i * j) % 5) % 11) * (((i % 6) + (j % 3)) // 4) * (((i % 4) != 0) | ((j % 2) == 0))
    assert M.sum() == 17359
    v = G.seurat_vst(M)["variance_standardized"]
    ref = [0, 0, 1.040954573, 0, 0.9700735028, 0.8867384746, 0]
    assert v[[0, 1, 2, 49, 100, 149, 199]] == pytest.approx(ref, abs=1e-8)
    assert v.sum() == pytest.approx(132.5760936, rel=1e-8)


def test_squared_knn_changes_statistic():
    rng = np.random.default_rng(0)
    P = np.r_[rng.normal(0, 1, (300, 5)), rng.normal(6, 0.05, (6, 5))]
    s_sq = G.gapclust_raw_knn(P, 50, squared=True)[0]
    s_eu = G.gapclust_raw_knn(P, 50, squared=False)[0]
    assert s_sq.shape == s_eu.shape == (47,)                                    # k - 3 columns, as R
    assert not np.allclose(s_sq, s_eu)
