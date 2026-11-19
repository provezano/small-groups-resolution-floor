"""Compare the 18 hand-typed cells of Figure 4 in make_figures.py with results/p1math_null.json.
Run from the deposit root:  python figures/check_fig4_cells.py     (prints one line per cell and a summary)"""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
N_TESTS = 22400                      # tests per scene: predicted share 1.00e-3 of sphere d=2 corresponds to 22.4 rejections
d = json.load(open(HERE.parent / "results" / "p1math_null.json"))
src = (HERE / "make_figures.py").read_text()
cells = eval(re.search(r"cells = (\[.*?\n\])", src, re.S).group(1))
geo = {"torus": "torus", "sphere": "sphere", "mixture": "gm"}
bad = 0
for g, dim, t, m, p in cells:
    k = f"{geo[g]}/d{dim}"; s = "true" if t == "true" else "twonn"
    jm = d[k]["measured_mean_rejections"][s]["0.001"] / N_TESTS * 1000
    jp = d[k]["predicted_mean_rejections"][s]["0.001"] / N_TESTS * 1000
    diff = max(abs(jm - m), abs(jp - p)); bad += diff > 0.006
    print(f"{g:8s} d={dim:2d} {t:6s} typed {m:5.2f} {p:5.2f}  file {jm:6.3f} {jp:6.3f}  max diff {diff:.3f}")
print("cells differing by more than 0.006:", bad, "of", len(cells))
