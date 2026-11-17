"""Summarize gate1 rows: bias / RMSE / correlation of each estimator vs truth."""
import json
import sys
import numpy as np

EST = ["in", "split", "eb", "eblog"]
for path in sys.argv[1:]:
    rows = json.load(open(path))
    print(f"\n=== {path}  ({len(rows)} studies)")
    groups = {}
    for r in rows:
        solver = r["file"].split("_")[1]
        knob = r["file"].split("_")[0][3:]
        groups.setdefault((knob, solver, r["kind"]), []).append(r)
    print(f"{'knob':5}{'solver':8}{'kind':7}{'n':>4}{'true':>8}" +
          "".join(f"{e+' bias':>12}{e+' rmse':>11}" for e in EST))
    for key in sorted(groups):
        g = groups[key]
        t = np.array([r["true"] for r in g])
        line = f"{key[0]:5}{key[1]:8}{key[2]:7}{len(g):4d}{t.mean():8.2f}"
        for e in EST:
            x = np.array([r[e] for r in g])
            line += f"{(x - t).mean():12.2f}{np.sqrt(((x - t) ** 2).mean()):11.2f}"
        print(line)
    t = np.array([r["true"] for r in rows])
    print("pooled: " + "  ".join(
        f"{e}: bias {np.mean([r[e] for r in rows]) - t.mean():+.2f}, "
        f"rmse {np.sqrt(np.mean([(r[e] - r['true']) ** 2 for r in rows])):.2f}, "
        f"corr {np.corrcoef([r[e] for r in rows], t)[0, 1]:.3f}" for e in EST))
    # does truth track spread or level? (rank corr, windowed sets only)
    w = [r for r in rows if r["kind"] == "win"]
    def rk(a):
        return np.argsort(np.argsort(a))
    for solver in sorted({r["file"] for r in w}):
        ws = [r for r in w if r["file"] == solver]
        t_, s_, l_ = (np.array([r[k] for r in ws]) for k in ("true", "spread", "level"))
        print(f"{solver:14} spearman(true,spread)={np.corrcoef(rk(t_), rk(s_))[0,1]:+.2f} "
              f"spearman(true,level)={np.corrcoef(rk(t_), rk(l_))[0,1]:+.2f}")
