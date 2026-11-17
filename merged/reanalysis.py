"""Re-analysis for the merged NCO paper (2026-11-17).

Replaces paper2's estimator (200-permutation Monte Carlo curves + d_split on a
400-sample slice) with exact U-statistic curves, following APIN Appendix B and
the Gate-1 check (estimator_check/GATE1_RESULTS.md):

  headroom d = 100 * (U - O) / U, where U = sum_i F_i(S/N) and O is the greedy
  optimum, both on F_i = exact expected best-of-k over instance i's pool.

Independence of axis and outcome (paper2 §2.3) is kept: the x-axis statistics
(severity, spread, level, power-law b) come from slice [0:200]; the headroom
comes from the disjoint pool [200:1000] (800 samples).

Sets, seeds, partial correlations and the instance bootstrap replicate
paper2/code/analysis_confirm2.py exactly; only the outcome changes.
"""
import json
import sys
import numpy as np
from math import lgamma
from scipy.stats import spearmanr, rankdata

P1 = "../data/"
P2 = "../paper2/data/"
X, POOL = slice(0, 200), slice(200, 1000)
SN, CAP = 100, 400
NSET, NSAMP, N_BOOT = 40, 200, 400
RHO_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
RHO_NSET, RHO_REP = 30, 25


# ---------- exact curves ----------
_W = {}
def W(m, kmax):
    """W[k-1, j] = P(min of a random k-subset of m sorted values is the j-th)."""
    if (m, kmax) not in _W:
        lc = lambda n, k: lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
        M = np.zeros((kmax, m))
        for k in range(1, kmax + 1):
            l = lc(m, k)
            j = np.arange(m - k + 1)
            M[k - 1, : m - k + 1] = np.exp([lc(m - 1 - jj, k - 1) - l for jj in j])
        _W[(m, kmax)] = M
    return _W[(m, kmax)]


def exact(gaps, kmax):
    g = np.sort(gaps)
    return W(len(g), kmax) @ g


def greedy(F, sn=SN):
    N = F.shape[0]
    d = np.minimum.accumulate(F[:, :-1] - F[:, 1:], axis=1)
    extra = sn * N - N
    flat = np.sort(d.ravel())[::-1]
    thr = flat[extra - 1]
    take = (d > thr).sum(1)
    short = extra - take.sum()
    for i, _ in np.argwhere(d == thr)[:short]:
        take[i] += 1
    return 1 + take


def gain(F, k, sn=SN):
    U = F[:, sn - 1].sum()
    return 100 * (U - F[np.arange(len(F)), k - 1].sum()) / U


def headroom(F, sn=SN):
    return gain(F, greedy(F, sn), sn)


# ---------- power-law fit (theory variable b) ----------
KF = 200
KGRID = np.unique(np.round(np.geomspace(1, KF, 30)).astype(int))
ALPHAS = np.linspace(0.05, 3.0, 119)
def fit_b(f):
    y = f[KGRID - 1]
    best = None
    for a_ in ALPHAS:
        A = np.stack([np.ones_like(KGRID, float), KGRID ** -a_], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = ((A @ coef - y) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, coef[1], a_)
    return max(best[1], 1e-9), best[2]


# ---------- paper2 machinery ----------
def presid(a, *covs):
    ra = rankdata(a)
    Cm = np.column_stack([rankdata(c) for c in covs] + [np.ones(len(ra))])
    beta, *_ = np.linalg.lstsq(Cm, ra, rcond=None)
    return ra - Cm @ beta


def load2(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    inst = [dict(d) for d in meta]
    for d in inst:
        d["c"] = z[f"c{d['id']}"]
    return inst


def prep(path):
    inst = load2(path)
    base = [d for d in inst if d["sigma"] < 0]
    shift = [d for d in inst if d["sigma"] > 0]
    b1 = np.mean([100 * (d["c"][X].mean() / d["ref"] - 1) for d in base])
    for d in base + shift:
        gx = 100 * (d["c"][X] / d["ref"] - 1)
        d["r"] = gx.mean() / b1
        d["b"], d["alpha"] = fit_b(exact(gx, KF))
        d["F"] = exact(100 * (d["c"][POOL] / d["ref"] - 1), CAP)
    return base, shift


def build_sets(shift, seed=5):
    P = sorted(shift, key=lambda d: d["r"]); n = len(P)
    rng = np.random.default_rng(seed); out = []
    for _ in range(NSAMP):
        lo = rng.integers(0, n - NSET)
        w = rng.integers(NSET, n - lo + 1)
        idx = sorted(rng.choice(np.arange(lo, lo + w), NSET, replace=False))
        g = [P[i] for i in idx]
        out.append(dict(ix=np.array(idx),
                        spread=float(np.std(np.log([d["r"] for d in g]))),
                        spread_b=float(np.std(np.log([d["b"] for d in g]))),
                        level=float(np.median([d["r"] for d in g])),
                        reflen=float(np.median([d["ref"] for d in g])),
                        d=headroom(np.stack([d["F"] for d in g]))))
    return out, n


def pc(y, x, covs):
    return spearmanr(presid(x, *covs), presid(y, *covs))[0]


def boot(sets, n, fn, seed=9):
    rb = np.random.default_rng(seed); v = []
    for _ in range(N_BOOT):
        keep = set(rb.choice(n, n, replace=True).tolist())
        sub = [s for s in sets if len(set(s["ix"].tolist()) & keep) >= NSET * 0.6]
        if len(sub) < 30:
            continue
        v.append(fn(sub))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def col(sets, k):
    return np.array([s[k] for s in sets])


def arm(path):
    base, shift = prep(path)
    sets, n = build_sets(shift)
    D = col(sets, "d")
    tests = {
        "spread|level": lambda S: pc(col(S, "d"), col(S, "spread"), [col(S, "level")]),
        "level|spread": lambda S: pc(col(S, "d"), col(S, "level"), [col(S, "spread")]),
        "spread_b|level": lambda S: pc(col(S, "d"), col(S, "spread_b"), [col(S, "level")]),
        "level|spread_b": lambda S: pc(col(S, "d"), col(S, "level"), [col(S, "spread_b")]),
        "reflen|spread": lambda S: pc(col(S, "d"), col(S, "reflen"), [col(S, "spread")]),
    }
    out = dict(d_range=[float(D.min()), float(D.max())], d_mean=float(D.mean()),
               spread_range=[float(col(sets, "spread").min()), float(col(sets, "spread").max())])
    top = col(sets, "spread") >= np.percentile(col(sets, "spread"), 75)
    out["d_top_quartile_spread"] = float(D[top].mean())
    for k, f in tests.items():
        out[k] = dict(point=float(f(sets)), ci=boot(sets, n, f))
    return out, base, shift


def three_way(base, shift):
    rows = []
    for rho in RHO_LEVELS:
        for rep in range(RHO_REP):
            rng = np.random.default_rng(3000 + rep)
            nO = int(round(rho * RHO_NSET)); nU = RHO_NSET - nO
            if nU > len(base):
                continue
            ib = list(rng.choice(len(base), nU, replace=False))
            is_ = list(rng.choice(len(shift), nO, replace=False))
            g = [base[i] for i in ib] + [shift[i] for i in is_]
            rows.append(dict(rho=rho,
                             spread=float(np.std(np.log([d["r"] for d in g]))),
                             level=float(np.median([d["r"] for d in g])),
                             d=headroom(np.stack([d["F"] for d in g])),
                             mem=[i for i in ib] + [len(base) + j for j in is_]))
    P, H, L, D = (np.array([r[k] for r in rows]) for k in ("rho", "spread", "level", "d"))
    npool = len(base) + len(shift)
    def stat(ix, covs):
        return spearmanr(presid(P[ix], *[c[ix] for c in covs]), presid(D[ix], *[c[ix] for c in covs]))[0]
    allix = np.arange(len(rows))
    res = {}
    for name, covs in (("rho|spread", [H]), ("spread|rho", None), ("rho|spread,level", [H, L])):
        if name == "spread|rho":
            f = lambda ix: spearmanr(presid(H[ix], P[ix]), presid(D[ix], P[ix]))[0]
        else:
            f = (lambda cv: (lambda ix: stat(ix, cv)))(covs)
        rb = np.random.default_rng(4); v = []
        for _ in range(N_BOOT):
            keep = set(rb.choice(npool, npool, replace=True).tolist())
            ix = np.array([i for i, r in enumerate(rows) if sum(m in keep for m in r["mem"]) >= RHO_NSET * 0.6])
            if len(ix) < 40:
                continue
            v.append(f(ix))
        res[name] = dict(point=float(f(allix)), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    res["by_rho"] = {str(rho): dict(spread=float(H[P == rho].mean()), d=float(D[P == rho].mean()))
                     for rho in RHO_LEVELS}
    return res


# ---------- paper1 ----------
def load1(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return [(m, z[f"c{m['id']}"]) for m in meta]


def p1_headroom(path, sel=None, n_boot=400, seed=33):
    inst = load1(path)
    if sel:
        inst = [x for x in inst if x[0]["dist"] == sel][:50]
    K = len(inst[0][1])
    F = np.stack([exact(100 * (c / m["ref"] - 1), CAP) for m, c in inst])
    pt = headroom(F)
    rb = np.random.default_rng(seed); v = []
    for _ in range(n_boot):
        ix = rb.choice(len(F), len(F), replace=True)
        v.append(headroom(F[ix]))
    return dict(N=len(F), K=K, point=float(pt), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


# ---------- rq3: label rule on exact curves ----------
def rq3(path, ratio):
    inst = load1(path)
    F = np.stack([exact(100 * (c[POOL] / m["ref"] - 1), CAP) for m, c in inst])
    lab = np.array([m["dist"] == "clustered" for m, _ in inst])
    N = len(F); S = SN * N
    w = np.where(lab, ratio, 1.0)
    k = np.maximum(1, np.floor(S * w / w.sum()).astype(int))
    k[np.argsort(-w)[: S - k.sum()]] += 1
    k = np.minimum(k, CAP)
    rb = np.random.default_rng(33); v_o, v_l = [], []
    for _ in range(N_BOOT):
        ix = rb.choice(N, N, replace=True)
        v_o.append(headroom(F[ix]))
        w_ = w[ix]; k_ = np.maximum(1, np.floor(S * w_ / w_.sum()).astype(int))
        k_[np.argsort(-w_)[: S - k_.sum()]] += 1
        v_l.append(gain(F[ix], np.minimum(k_, CAP)))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(oracle=float(headroom(F)), oracle_ci=q(v_o),
                label=float(gain(F, k)), label_ci=q(v_l))


if __name__ == "__main__":
    R = {}
    which = sys.argv[1:] or ["p1", "rq1", "rq3"]
    if "p1" in which:
        R["p1"] = {}
        for s in ["AM", "SYMNCO", "POMO"]:
            R["p1"][s] = dict(
                indist_axisB=p1_headroom(P1 + f"expand_{s}_axisB.npz", "uniform"),
                indist_axisA=p1_headroom(P1 + f"expand_{s}_axisA.npz", "uniform"),
                ood_axisA=p1_headroom(P1 + f"confirm_{s}_axisA.npz"))
            print("p1", s, json.dumps(R["p1"][s]), flush=True)
    if "rq1" in which:
        R["rq1"] = {}
        for tag, pre in [("C1_cluster", "rq1conf"), ("C2A_cluster", "rq1lv"), ("C2B_size", "rq1sz")]:
            for s in ["AM", "SYMNCO", "POMO"]:
                r, base, shift = arm(P2 + f"{pre}_{s}.npz")
                if tag == "C2A_cluster" and s != "POMO":
                    r["composition"] = three_way(base, shift)
                R["rq1"][f"{tag}/{s}"] = r
                print(tag, s, json.dumps({k: v for k, v in r.items() if k != "composition"}), flush=True)
    if "rq3" in which:
        R["rq3"] = {}
        for s, ratio in [("AM", 20.0), ("SYMNCO", 20.0), ("POMO", 20.0)]:
            R["rq3"][s] = rq3(P2 + f"rq3_{s}_axisA.npz", ratio)
            print("rq3", s, json.dumps(R["rq3"][s]), flush=True)
    out = "reanalysis_" + "_".join(which) + ".json"
    json.dump(R, open("results/" + out, "w"), indent=1)
    print("saved", out)
