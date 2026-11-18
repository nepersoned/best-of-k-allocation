"""Reviewer-driven extras (2026-11-18), all on stored arrays with exact curves.

(1) lengths : headroom translated into mean tour-length reduction (%), and the
              uniform gap it starts from, for the in-distribution, mixed and policy
              collections (oracle) and for the charged probe rule (policy collection).
(2) budget  : spread|level and level|spread at u = 25, 50, 100, 200 (cap 400),
              collections C2A (cluster) and C2B (size), same sets and bootstrap
              as reanalysis.py.
(3) disjoint: independence check. Each replicate partitions the 240 heterogeneous
              instances into 12 disjoint workloads of 20: the severity-sorted pool is
              cut into 24 blocks of 10 and blocks are paired at random, so spread
              varies across workloads. Rank-residualized partial Spearman of spread
              with headroom given level over the 12 independent workloads; 1000
              replicates; report the median correlation, the share of positive
              replicates, and the share with permutation p < 0.05 (2000 permutations
              of headroom across workloads).
"""
import json
import numpy as np
from scipy.stats import spearmanr
import reanalysis as R
import reanalysis2 as R2


def length_reduction(F, ref, k, u):
    L_u = (ref * (1 + F[:, u - 1] / 100)).sum()
    L_k = (ref * (1 + F[np.arange(len(F)), k - 1] / 100)).sum()
    return 100 * (L_u - L_k) / L_u


def lengths():
    out = {}
    for s in ["AM", "SYMNCO", "POMO"]:
        row = {}
        for tag, path, sel, pool in [("indist", R.P1 + f"expand_{s}_axisA.npz", "uniform", slice(None)),
                                     ("mixed", R.P1 + f"confirm_{s}_axisA.npz", None, slice(None)),
                                     ("policy", R.P2 + f"rq3_{s}_axisA.npz", None, R.POOL)]:
            inst = R.load1(path)
            if sel:
                inst = [x for x in inst if x[0]["dist"] == sel][:50]
            ref = np.array([m["ref"] for m, _ in inst])
            F = np.stack([R.exact(100 * (c[pool] / m["ref"] - 1), R.CAP) for m, c in inst])
            k = R.greedy(F)
            row[tag] = dict(uniform_gap=float(F[:, R.SN - 1].mean()), headroom=float(R.gain(F, k)),
                            length_reduction=float(length_reduction(F, ref, k, R.SN)))
            if tag == "policy":
                probe = np.stack([c[R.X][:20] for _, c in inst]); med = np.median(probe, 1)
                st = (med - probe.min(1)) / med
                N = len(F); S = R.SN * N; rem = S - 20 * N
                extra = np.floor(rem * st / st.sum()).astype(int)
                extra[np.argsort(-st)[: rem - extra.sum()]] += 1
                kp = np.minimum(20 + extra, R.CAP)
                row[tag]["probe_headroom"] = float(R.gain(F, kp))
                row[tag]["probe_length_reduction"] = float(length_reduction(F, ref, kp, R.SN))
        out[s] = row
        print("lengths", s, json.dumps(row), flush=True)
    return out


def budget():
    out = {}
    for tag, pre in [("cluster", "rq1lv"), ("size", "rq1sz")]:
        for s in ["AM", "SYMNCO", "POMO"]:
            base, shift = R.prep(R.P2 + f"{pre}_{s}.npz")
            P = sorted(shift, key=lambda d: d["r"]); n = len(P)
            res = {}
            for u in (25, 50, 100, 200):
                rng = np.random.default_rng(5); sets = []
                for _ in range(R.NSAMP):
                    lo = rng.integers(0, n - R.NSET)
                    w = rng.integers(R.NSET, n - lo + 1)
                    idx = sorted(rng.choice(np.arange(lo, lo + w), R.NSET, replace=False))
                    g = [P[i] for i in idx]
                    sets.append(dict(ix=np.array(idx),
                                     spread=float(np.std(np.log([d["r"] for d in g]))),
                                     level=float(np.median([d["r"] for d in g])),
                                     d=R.headroom(np.stack([d["F"] for d in g]), u)))
                fs = lambda S: R.pc(R.col(S, "d"), R.col(S, "spread"), [R.col(S, "level")])
                fl = lambda S: R.pc(R.col(S, "d"), R.col(S, "level"), [R.col(S, "spread")])
                res[u] = {"d_mean": float(R.col(sets, "d").mean()),
                          "spread|level": dict(point=float(fs(sets)), ci=R.boot(sets, n, fs)),
                          "level|spread": dict(point=float(fl(sets)), ci=R.boot(sets, n, fl))}
            out[f"{tag}/{s}"] = res
            print("budget", tag, s, json.dumps(res), flush=True)
    return out


def disjoint(reps=1000, nperm=2000):
    out = {}
    for tag, pre in [("C1_cluster", "rq1conf"), ("C2A_cluster", "rq1lv"), ("C2B_size", "rq1sz")]:
        for s in ["AM", "SYMNCO", "POMO"]:
            base, shift = R.prep(R.P2 + f"{pre}_{s}.npz")
            P = sorted(shift, key=lambda d: d["r"])
            blocks = [P[i:i + 10] for i in range(0, 240, 10)]
            rng = np.random.default_rng(20261118)
            rs, ps = [], []
            for _ in range(reps):
                perm = rng.permutation(24)
                H, L, D = [], [], []
                for j in range(12):
                    g = blocks[perm[2 * j]] + blocks[perm[2 * j + 1]]
                    H.append(np.std(np.log([d["r"] for d in g])))
                    L.append(np.median([d["r"] for d in g]))
                    D.append(R.headroom(np.stack([d["F"] for d in g])))
                H, L, D = map(np.array, (H, L, D))
                x = R.presid(H, L); y = R.presid(D, L)
                rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
                rx -= rx.mean(); ry -= ry.mean()
                den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
                r0 = float(rx @ ry / den)
                Pm = np.argsort(rng.random((nperm, len(ry))), 1)
                null = (ry[Pm] @ rx) / den
                rs.append(r0); ps.append(float((null >= r0).mean()))
            rs, ps = np.array(rs), np.array(ps)
            out[f"{tag}/{s}"] = dict(median_r=float(np.median(rs)), share_positive=float((rs > 0).mean()),
                                     share_p05=float((ps < 0.05).mean()), iqr=[float(np.percentile(rs, 25)), float(np.percentile(rs, 75))])
            print("disjoint", tag, s, out[f"{tag}/{s}"], flush=True)
    return out


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["lengths", "budget", "disjoint"]
    O = {}
    for w in which:
        O[w] = globals()[w]()
    json.dump(O, open("results/extras_" + "_".join(which) + ".json", "w"), indent=1)
