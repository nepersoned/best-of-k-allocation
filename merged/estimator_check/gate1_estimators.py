"""Gate 1: does a noise-corrected spread estimator beat d_split?

Design (truth known exactly):
  population  = the 1000 stored costs of each instance (paper2 arrays)
  true f_i(k) = E[min of k iid draws from the population]   (closed form)
  a "study"   = m iid draws per instance from the population
  target      = oracle headroom d = (U - O) / U on the TRUE curves, in percent

Estimators, all computed from the study samples only:
  d_in    allocate and evaluate on the same m samples          (paper1 phantom)
  d_split allocate on half, evaluate on the other half         (paper1 fix)
  d_eb    empirical-Bayes: shrink each instance's marginal-gain curve toward
          the set mean by (true spread)/(observed spread), where
          true spread = observed across-instance variance - mean bootstrap
          noise variance; then plug in                          (proposed)
  d_eblog same, on log marginals (spread in log units, as in the theory)
"""
import json
import sys
import numpy as np

DATA = "../../paper2/data/"
import os
M = int(os.environ.get("G_M", 400))      # study samples per instance
SN = int(os.environ.get("G_SN", 25))    # budget per instance
CAP = int(os.environ.get("G_CAP", 100)) # per-instance cap
NSET = 40        # instances per set
B = 30           # bootstrap reps for noise variance
R = 8            # study replicates per set
NWIN = 24        # windowed sets per file
rng = np.random.default_rng(20261001)


def logC(n, k):
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def ustat_weights(m, kmax):
    """W[k-1, j] = P(min of a random k-subset of m sorted values is the j-th)."""
    from math import lgamma
    W = np.zeros((kmax, m))
    for k in range(1, kmax + 1):
        lc = logC(m, k)
        for j in range(m - k + 1):          # 0-indexed j, needs m-1-j >= k-1
            W[k - 1, j] = np.exp(logC(m - 1 - j, k - 1) - lc)
    return W


def pop_weights(n, kmax):
    """iid draws with replacement from n sorted values."""
    j = np.arange(n)
    hi = ((n - j) / n)[None, :] ** np.arange(1, kmax + 1)[:, None]
    lo = ((n - j - 1) / n)[None, :] ** np.arange(1, kmax + 1)[:, None]
    return hi - lo


def greedy(f):
    """f: (N, CAP) curves, f[:,k-1] = f(k). Optimal allocation for sum = SN*N."""
    N = f.shape[0]
    d = f[:, :-1] - f[:, 1:]                       # marginal of step k -> k+1
    d = np.minimum.accumulate(d, axis=1)           # enforce non-increasing
    extra = SN * N - N
    flat = np.sort(d.ravel())[::-1]
    thr = flat[extra - 1]
    take = (d > thr).sum(1)
    short = extra - take.sum()
    ties = np.argwhere(d == thr)
    for i, _ in ties[:short]:
        take[i] += 1
    return 1 + take                                  # k_i


def gain(f_eval, k):
    N = f_eval.shape[0]
    U = f_eval[:, SN - 1].sum()
    O = f_eval[np.arange(N), k - 1].sum()
    return 100 * (U - O) / U


def curves_from_marginals(f1, d):
    return np.concatenate([f1[:, None], f1[:, None] - np.cumsum(d, 1)], 1)


def eb_shrink(f, fboot, log=False):
    d = f[:, :-1] - f[:, 1:]
    db = fboot[:, :, :-1] - fboot[:, :, 1:]
    if log:
        eps = 1e-6
        d_, db_ = np.log(np.maximum(d, eps)), np.log(np.maximum(db, eps))
    else:
        d_, db_ = d, db
    v = db_.var(1)                                  # (N, CAP-1) noise var
    mu = d_.mean(0)
    V = d_.var(0)
    s2 = np.maximum(V - v.mean(0), 0)
    lam = s2 / (s2 + v + 1e-12)
    ds = mu + lam * (d_ - mu)
    if log:
        ds = np.exp(ds)
    return curves_from_marginals(f[:, 0], ds)


def study(pop_sorted_gap, W_m, W_h, Wpop_true):
    N = len(pop_sorted_gap)
    true = np.stack([Wpop_true @ p for p in pop_sorted_gap])
    d_true = gain(true, greedy(true))
    samp = [rng.choice(p, M, replace=True) for p in pop_sorted_gap]
    f = np.stack([W_m @ np.sort(s) for s in samp])
    d_in = gain(f, greedy(f))
    perm = [rng.permutation(s) for s in samp]
    fa = np.stack([W_h @ np.sort(s[: M // 2]) for s in perm])
    fb = np.stack([W_h @ np.sort(s[M // 2:]) for s in perm])
    d_split = gain(fb, greedy(fa))
    fboot = np.stack([np.stack([W_m @ np.sort(rng.choice(s, M, replace=True))
                                for _ in range(B)]) for s in samp])
    out = {"true": d_true, "in": d_in, "split": d_split}
    for name, lg in (("eb", False), ("eblog", True)):
        g = eb_shrink(f, fboot, log=lg)
        out[name] = gain(g, greedy(g))
    return out


def main():
    W_m = ustat_weights(M, CAP)
    W_h = ustat_weights(M // 2, CAP)
    Wp = pop_weights(1000, CAP)
    rows = []
    for fn in sys.argv[1:]:
        d = np.load(DATA + fn + ".npz")
        meta = json.loads(str(d["meta"]))
        gaps = [np.sort(100 * (d[f"c{i}"] / m["ref"] - 1)) for i, m in enumerate(meta)]
        sev = np.array([g.mean() for g in gaps])
        order = np.argsort(sev)
        sets = [("homog", np.arange(40))]
        for w in range(NWIN):
            width = rng.integers(NSET, len(gaps) + 1)
            lo = rng.integers(0, len(gaps) - width + 1)
            sets.append(("win", rng.choice(order[lo:lo + width], NSET, replace=False)))
        for kind, idx in sets:
            logsev = np.log(np.maximum(sev[idx], 1e-3))
            for r in range(R):
                o = study([gaps[i] for i in idx], W_m, W_h, Wp)
                o.update(file=fn, kind=kind, spread=float(logsev.std()),
                         level=float(np.median(sev[idx])))
                rows.append(o)
            print(fn, kind, {k: round(v, 3) for k, v in o.items()
                             if isinstance(v, float)}, flush=True)
    with open(f"gate1_rows_m{M}_sn{SN}.json", "w") as fh:
        json.dump(rows, fh)


if __name__ == "__main__":
    main()

