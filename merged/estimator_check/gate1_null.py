"""Exact null: all N instances share ONE population (true spread 0, true gain 0).
Measures the phantom (d_in) and the estimators' behaviour when there is nothing to find."""
import json
import os
import numpy as np
import gate1_estimators as G

rng = np.random.default_rng(7)
G.rng = rng
REPS = 6
out = {}
W_m, W_h = G.ustat_weights(G.M, G.CAP), G.ustat_weights(G.M // 2, G.CAP)
Wp = G.pop_weights(1000, G.CAP)
for fn in ["rq1lv_AM", "rq1lv_SYMNCO", "rq1lv_POMO"]:
    d = np.load(G.DATA + fn + ".npz")
    meta = json.loads(str(d["meta"]))
    res = {k: [] for k in ("true", "in", "split", "eb", "eblog")}
    for src in rng.choice(40, 8, replace=False):          # in-distribution sources
        pop = np.sort(100 * (d[f"c{src}"] / meta[src]["ref"] - 1))
        for _ in range(REPS):
            o = G.study([pop] * G.NSET, W_m, W_h, Wp)
            for k in res:
                res[k].append(o[k])
    out[fn] = {k: [float(np.mean(v)), float(np.percentile(v, 95))] for k, v in res.items()}
    print(fn, {k: f"mean {v[0]:+.2f} p95 {v[1]:+.2f}" for k, v in out[fn].items()}, flush=True)
json.dump(out, open(f"gate1_null_m{G.M}_sn{G.SN}.json", "w"), indent=1)
