"""Estimator accuracy with known truth, at the main analysis settings (m=800, u=100, cap 400).

Each instance's 1,000 stored costs are treated as the population; the true curve is
the expected minimum of k i.i.d. draws from it (closed form). A simulated study draws
m costs per instance with replacement and estimates the headroom of a 40-instance
workload by
  in    : exact U-statistic curves on all m draws, allocate and evaluate in sample
  split : exact curves on two halves; allocate on one, evaluate on the other
Accuracy: C2A (cluster) and C2B (size), all three solvers; per file the 40 reference
instances plus 24 windowed workloads, 8 studies each.
Null: 40 copies of one reference instance's population (true headroom exactly 0),
8 source instances x 6 studies per file, C2A and C2B.
"""
import json
import numpy as np
import reanalysis as R

M, SN, CAP, NSET, NWIN, REPS = 800, 100, 400, 40, 24, 8
rng = np.random.default_rng(20261117)


def pop_curve(sorted_pop, kmax):
    n = len(sorted_pop); j = np.arange(n)
    ks = np.arange(1, kmax + 1)[:, None]
    w = ((n - j) / n)[None, :] ** ks - ((n - j - 1) / n)[None, :] ** ks
    return w @ sorted_pop


def study(pops, ptrue):
    d_true = R.headroom(ptrue, SN)
    samp = [rng.choice(p, M, replace=True) for p in pops]
    F = np.stack([R.exact(s, CAP) for s in samp])
    d_in = R.headroom(F, SN)
    perm = [rng.permutation(s) for s in samp]
    Fa = np.stack([R.exact(s[: M // 2], CAP) for s in perm])
    Fb = np.stack([R.exact(s[M // 2:], CAP) for s in perm])
    d_split = R.gain(Fb, R.greedy(Fa, SN), SN)
    return d_true, d_in, d_split


acc = {"in": [], "split": []}; null = {}
for pre in ["rq1lv", "rq1sz"]:
    for s in ["AM", "SYMNCO", "POMO"]:
        inst = R.load2(R.P2 + f"{pre}_{s}.npz")
        pops = [np.sort(100 * (d["c"] / d["ref"] - 1)) for d in inst]
        T = np.stack([pop_curve(p, CAP) for p in pops])
        sev = np.array([p.mean() for p in pops]); order = np.argsort(sev)
        ref_ix = [i for i, d in enumerate(inst) if d["sigma"] < 0]
        sets = [np.array(ref_ix)]
        for _ in range(NWIN):
            width = rng.integers(NSET, len(pops) + 1)
            lo = rng.integers(0, len(pops) - width + 1)
            sets.append(rng.choice(order[lo:lo + width], NSET, replace=False))
        for idx in sets:
            for _ in range(REPS):
                t, i, sp = study([pops[j] for j in idx], T[idx])
                acc["in"].append(i - t); acc["split"].append(sp - t)
        nl = []
        for src in rng.choice(ref_ix, 8, replace=False):
            for _ in range(6):
                t, i, _ = study([pops[src]] * NSET, np.stack([T[src]] * NSET))
                nl.append(i)
        null[f"{pre}/{s}"] = [float(np.mean(nl)), float(np.percentile(nl, 95))]
        print(pre, s, "null mean/p95", null[f"{pre}/{s}"], flush=True)

out = {k: dict(bias=float(np.mean(v)), rmse=float(np.sqrt(np.mean(np.square(v)))), n=len(v))
       for k, v in acc.items()}
out["null"] = null
print(json.dumps(out, indent=1))
json.dump(out, open("results/estimator_sim.json", "w"), indent=1)
