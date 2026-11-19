"""
GapClust full pipeline: NOT REDISTRIBUTED (licensing).

The module that was here ported more of the official GapClust R code (github.com/fabotao/GapClust: the elbow rule,
the statistics on the kNN result) and of R's own `density`, with Seurat's variable-feature step. The upstream GapClust
package is GPL-3 with no LICENSE file, so these ports cannot be redistributed under this deposit's MIT licence and were
removed. Names are kept so that importers still work; each function delegates to your own implementation if the
environment variable GAPCLUST_PY names a Python file that defines it, and otherwise raises GapClustUnavailable
(see tiny_baselines.py for the contract). Stored results produced with the removed code are kept in results/.
"""
from .tiny_baselines import GapClustUnavailable, _gapclust_user, gapclust_available   # noqa: F401


def _delegate(name):
    def f(*args, **kwargs):
        return getattr(_gapclust_user(), name)(*args, **kwargs)
    f.__name__ = name
    f.__doc__ = f"Stub for {name}: delegates to the user's port (GAPCLUST_PY) or raises GapClustUnavailable."
    return f


seurat_vst = _delegate("seurat_vst")
bw_nrd0 = _delegate("bw_nrd0")
r_density = _delegate("r_density")
find_elbow = _delegate("find_elbow")
select_genes = _delegate("select_genes")
embed = _delegate("embed")
gapclust_raw_knn = _delegate("gapclust_raw_knn")
gapclust_full = _delegate("gapclust_full")
