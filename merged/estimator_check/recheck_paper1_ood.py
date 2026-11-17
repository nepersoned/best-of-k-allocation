"""Re-check paper1's confirmatory OOD result (50 uniform + 50 clustered, S/N=100).
Axis B: exact full-pool curves = true headroom of the protocol.
Axis A: full-sample U-statistic plug-in (bias about +0.1..+0.3 pt per gate 1) and d_split."""
import json
import numpy as np
import gate1_estimators as G

rng = np.random.default_rng(5)
G.SN = 100
for solver in ["AM", "SYMNCO", "POMO"]:
    for ax in ["axisA", "axisB"]:
        d = np.load(f"../../data/confirm_{solver}_{ax}.npz")
        meta = json.loads(str(d["meta"]))
        gaps = [np.sort(100 * (d[f"c{i}"] / m["ref"] - 1)) for i, m in enumerate(meta)]
        K = len(gaps[0])
        G.CAP = K // 2
        f = np.stack([G.ustat_weights(K, G.CAP) @ g for g in gaps]) if False else None
        W = G.ustat_weights(K, G.CAP)
        f = np.stack([W @ g for g in gaps])
        full = G.gain(f, G.greedy(f))
        Wh = G.ustat_weights(K // 2, G.CAP)
        sp = []
        for _ in range(10):
            perm = [rng.permutation(g) for g in gaps]
            fa = np.stack([Wh @ np.sort(p[: K // 2]) for p in perm])
            fb = np.stack([Wh @ np.sort(p[K // 2:]) for p in perm])
            sp.append(G.gain(fb, G.greedy(fa)))
        ndist = {m["dist"] for m in meta}
        print(f"{solver:7}{ax}: N={len(gaps)} dists={sorted(ndist)} K={K} "
              f"exact-curve plug-in={full:6.2f}  d_split(mean of 10)={np.mean(sp):6.2f}", flush=True)
