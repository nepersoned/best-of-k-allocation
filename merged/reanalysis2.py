"""Second batch of the exact-curve re-analysis.

(1) Test U: the scale reading recomputed in absolute length units.
    d_abs = mean_i(F_i(u) * ref_i / 100) - mean_i(F_i(k_i) * ref_i / 100),
    where k is the greedy allocation on the gap curves. No length-bearing
    denominator enters. Partial Spearman of d_abs with set-median tour length,
    given spread; same sets and bootstrap as reanalysis.py.
(2) Simple probe rules on the policy collection (rq3), all charged:
    a probe of m samples per instance (taken from slice [0:200], so it is
    independent of the evaluation pool [200:1000]) is counted against the
    budget and kept as candidates; the remaining budget is split in
    proportion to a probe statistic; instance i is scored at F_i(k_i) with
    k_i = m + extra_i. Statistics: coefficient of variation (paper 1) and
    (median - min) / median (the feature paper 2 found carried the signal).
    Also: the uncharged oracle and label 20:1, for reference.
"""
import json
import numpy as np
import reanalysis as R

def test_u(prefix, solver):
    base, shift = R.prep(R.P2 + f"{prefix}_{solver}.npz")
    P = sorted(shift, key=lambda d: d["r"]); n = len(P)
    rng = np.random.default_rng(5); sets = []
    for _ in range(R.NSAMP):
        lo = rng.integers(0, n - R.NSET)
        w = rng.integers(R.NSET, n - lo + 1)
        idx = sorted(rng.choice(np.arange(lo, lo + w), R.NSET, replace=False))
        g = [P[i] for i in idx]
        F = np.stack([d["F"] for d in g]); ref = np.array([d["ref"] for d in g])
        k = R.greedy(F)
        dabs = np.mean(F[:, R.SN - 1] * ref / 100) - np.mean(F[np.arange(len(F)), k - 1] * ref / 100)
        sets.append(dict(ix=np.array(idx),
                         spread=float(np.std(np.log([d["r"] for d in g]))),
                         reflen=float(np.median(ref)), d=float(dabs)))
    f = lambda S: R.pc(R.col(S, "d"), R.col(S, "reflen"), [R.col(S, "spread")])
    return dict(point=float(f(sets)), ci=R.boot(sets, n, f))


def charged(F, stat, m, sn=R.SN):
    N = len(F); S = sn * N
    rem = S - m * N
    w = np.maximum(stat, 1e-12)
    extra = np.floor(rem * w / w.sum()).astype(int)
    extra[np.argsort(-w)[: rem - extra.sum()]] += 1
    k = np.minimum(m + extra, R.CAP)
    return R.gain(F, k, sn)


def rq3_rules(solver, m=20):
    inst = R.load1(R.P2 + f"rq3_{solver}_axisA.npz")
    F = np.stack([R.exact(100 * (c[R.POOL] / mm["ref"] - 1), R.CAP) for mm, c in inst])
    # probe statistics use raw tour costs only: no reference tour at deployment time
    probe = np.stack([c[R.X][:m] for mm, c in inst])
    cv = probe.std(1) / np.maximum(probe.mean(1), 1e-9)
    med = np.median(probe, 1)
    mm_ = (med - probe.min(1)) / np.maximum(med, 1e-9)
    lab = np.array([mm["dist"] == "clustered" for mm, _ in inst])
    rules = {"oracle": None, "probe_cv": cv, "probe_medmin": mm_}
    out = {}
    N = len(F)
    rb = np.random.default_rng(33)
    boots = {k: [] for k in rules}
    pairs = {"medmin-cv": []}
    def evalr(ix):
        r = {}
        for k, s in rules.items():
            r[k] = R.headroom(F[ix]) if s is None else charged(F[ix], s[ix], m)
        return r
    pt = evalr(np.arange(N))
    for _ in range(R.N_BOOT):
        ix = rb.choice(N, N, replace=True)
        r = evalr(ix)
        for k in rules:
            boots[k].append(r[k])
        pairs["medmin-cv"].append(r["probe_medmin"] - r["probe_cv"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for k in rules:
        out[k] = dict(point=float(pt[k]), ci=q(boots[k]))
    out["medmin-cv"] = dict(point=float(pt["probe_medmin"] - pt["probe_cv"]), ci=q(pairs["medmin-cv"]))
    out["recovery_medmin"] = float(pt["probe_medmin"] / pt["oracle"]) if pt["oracle"] > 0.5 else None
    out["recovery_cv"] = float(pt["probe_cv"] / pt["oracle"]) if pt["oracle"] > 0.5 else None
    return out


if __name__ == "__main__":
    O = {"test_u": {}, "rq3_rules": {}}
    for tag, pre in [("cluster", "rq1lv"), ("size", "rq1sz")]:
        for s in ["AM", "SYMNCO"]:
            O["test_u"][f"{tag}/{s}"] = test_u(pre, s)
            print("testU", tag, s, O["test_u"][f"{tag}/{s}"], flush=True)
    for s in ["AM", "SYMNCO", "POMO"]:
        for m in (20, 40):
            O["rq3_rules"][f"{s}/m{m}"] = rq3_rules(s, m)
            print("rq3", s, m, json.dumps(O["rq3_rules"][f"{s}/m{m}"]), flush=True)
    json.dump(O, open("results/reanalysis2.json", "w"), indent=1)
    print("saved reanalysis2.json")
