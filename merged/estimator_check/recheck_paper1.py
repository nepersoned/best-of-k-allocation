"""Re-check paper1's in-distribution audit (uniform TSP-100, N=50, S/N=100).

Axis B is a deterministic pool: the inference protocol draws k trajectories
without replacement from a FIXED pool of 800. The exact curve of that protocol
is the U-statistic over the full pool -- there is no estimation noise. So the
full-pool plug-in gain is the TRUE headroom of the Axis B protocol.
Axis A (sampling) is re-checked the same way for comparison."""
import json
import numpy as np
import gate1_estimators as G

G.SN, G.CAP = 100, 400
rng = np.random.default_rng(11)
for solver in ["POMO", "AM", "SYMNCO"]:
    for ax in ["axisB", "axisA"]:
        d = np.load(f"../../data/expand_{solver}_{ax}.npz")
        meta = json.loads(str(d["meta"]))
        idx = [i for i, m in enumerate(meta) if m["dist"] == "uniform"][:50]
        gaps = [np.sort(100 * (d[f"c{i}"] / meta[i]["ref"] - 1)) for i in idx]
        K = len(gaps[0])
        W = G.ustat_weights(K, G.CAP)
        f = np.stack([W @ g for g in gaps])
        d_full = G.gain(f, G.greedy(f))
        Wh = G.ustat_weights(K // 2, G.CAP)
        sp = []
        for _ in range(20):
            perm = [rng.permutation(g) for g in gaps]
            fa = np.stack([Wh @ np.sort(p[: K // 2]) for p in perm])
            fb = np.stack([Wh @ np.sort(p[K // 2:]) for p in perm])
            sp.append(G.gain(fb, G.greedy(fa)))
        ndistinct = np.mean([len(np.unique(np.round(g, 6))) for g in gaps])
        print(f"{solver:7}{ax}: K={K} distinct/inst={ndistinct:6.1f}  "
              f"full-pool plug-in={d_full:6.3f}  d_split(mean of 20)={np.mean(sp):6.3f}")
