"""Model-based allocation baselines (policy collection, rq3), reviewer request.

Same accounting as the probe rules: a probe of m samples per instance from slice
[0:200] is charged and kept; the remaining budget is allocated; instance i is
scored at F_i(k_i), the exact curve on the disjoint pool [200:1000]; cap 400.
No reference tour is used: curves are in relative cost, cost / (probe minimum).

fit_greedy  : fit f(k) = a + b k^-alpha (alpha on a grid, a,b by least squares,
              b >= 0) to the exact best-of-k curve of each instance's probe
              (k = 1..m), extrapolate, and allocate the rest greedily by the
              fitted marginal reductions (optimal for the fitted curves).
two_stage   : fit_greedy with half of the remaining budget, then refit each
              instance on all its observed samples (at most 200, the decision
              slice) and allocate the other half greedily.
oracle_shape: greedy on the true exact curves (uncharged) -- the ceiling.
"""
import json
import numpy as np
import reanalysis as R
import reanalysis2 as R2

U, CAP, OBS = 100, 400, 200
ALPHAS = np.linspace(0.05, 2.5, 50)


def fit_curve(costs_obs, kmax=CAP):
    m = len(costs_obs)
    f = R.exact(costs_obs, m)                       # exact curve on the observed samples, k=1..m
    f = f / costs_obs.min()                         # relative cost
    ks = np.arange(1, m + 1, dtype=float)
    best = None
    for al in ALPHAS:
        X = np.stack([np.ones(m), ks ** -al], 1)
        coef, *_ = np.linalg.lstsq(X, f, rcond=None)
        a, b = coef
        if b < 0:
            a, b = f.mean(), 0.0
        sse = ((a + b * ks ** -al - f) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, a, b, al)
    _, a, b, al = best
    K = np.arange(1, kmax + 1, dtype=float)
    return a + b * K ** -al


def greedy_extra(curves, k0, extra):
    """Add `extra` samples greedily on fitted curves starting from counts k0."""
    k = k0.copy()
    N = len(k)
    # marginal gains for the next step of each instance
    def mg(i):
        return curves[i][k[i] - 1] - curves[i][k[i]] if k[i] < CAP else -np.inf
    import heapq
    h = [(-mg(i), i) for i in range(N)]
    heapq.heapify(h)
    for _ in range(extra):
        g, i = heapq.heappop(h)
        if -g == -np.inf:
            break
        k[i] += 1
        heapq.heappush(h, (-mg(i), i))
    return k


def policies(costs, F, ix, m):
    N = len(ix); S = U * N
    k0 = np.full(N, m)
    rest = S - m * N
    curves = [fit_curve(costs[j][:m]) for j in ix]
    k1 = greedy_extra(curves, k0, rest)
    out = {"fit_greedy": R.gain(F[ix], k1, U)}
    half = rest // 2
    ka = greedy_extra(curves, k0, half)
    curves2 = [fit_curve(costs[j][:min(ka[t], OBS)]) for t, j in enumerate(ix)]
    kb = greedy_extra(curves2, ka, rest - half)
    out["two_stage"] = R.gain(F[ix], kb, U)
    probe = np.stack([costs[j][:m] for j in ix]); med = np.median(probe, 1)
    out["probe_medmin"] = R2.charged(F[ix], (med - probe.min(1)) / med, m)
    return out


def run(solver, m=20, n_boot=200):
    inst = R.load1(R.P2 + f"rq3_{solver}_axisA.npz")
    costs = [c[R.X] for _, c in inst]
    F = np.stack([R.exact(100 * (c[R.POOL] / mm["ref"] - 1), CAP) for mm, c in inst])
    N = len(F)
    pt = policies(costs, F, np.arange(N), m)
    pt["oracle"] = R.headroom(F, U)
    rb = np.random.default_rng(33)
    B = {k: [] for k in pt}; D = {k: [] for k in ("fit_greedy", "two_stage")}
    for _ in range(n_boot):
        ix = rb.choice(N, N, replace=True)
        e = policies(costs, F, ix, m); e["oracle"] = R.headroom(F[ix], U)
        for k in pt: B[k].append(e[k])
        for k in D: D[k].append(e[k] - e["probe_medmin"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {k: dict(point=float(pt[k]), ci=q(B[k])) for k in pt}
    res["minus_probe"] = {k: dict(point=float(pt[k] - pt["probe_medmin"]), ci=q(v)) for k, v in D.items()}
    return res


if __name__ == "__main__":
    import sys
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    O = {}
    for s in ["AM", "SYMNCO", "POMO"]:
        O[s] = run(s, m)
        print(s, m, json.dumps(O[s]), flush=True)
    json.dump(O, open(f"results/fit_baselines_m{m}.json", "w"), indent=1)
