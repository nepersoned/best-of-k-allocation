# 바닥 '정의' 스윕 — max 가 원천 수에 따라 발산함을 파일로 기록 (§3.2 근거)
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
FLOOR_REP, FLOOR_PERM, S_PER = 6, 60, 100

def curve(c, ref, kmax=None, n_perm=FLOOR_PERM, seed=22):
    K = len(c); kmax = kmax or K
    rng = np.random.default_rng(seed)
    idx = rng.random((n_perm, K)).argsort(1)[:, :kmax]
    return 100.0*(np.minimum.accumulate(np.asarray(c)[idx], 1).mean(0)/ref - 1.0)
def cvx(f):
    n = len(f); xs = np.arange(n, dtype=float); h = []
    for i in range(n):
        while len(h) >= 2:
            (x1,y1),(x2,y2) = h[-2], h[-1]
            if (y2-y1)*(xs[i]-x1) >= (f[i]-y1)*(x2-x1): h.pop()
            else: break
        h.append((xs[i], f[i]))
    return np.interp(xs, np.array([p[0] for p in h]), np.array([p[1] for p in h]))
def d_in_null(cs, s_per):
    N = len(cs); mg = [(-np.diff(cvx(c))) for c in cs]
    caps = np.array([len(c) for c in cs]); S = min(N*s_per, int(caps.sum()))
    M = np.concatenate([np.stack([mg[i], np.full(len(mg[i]), i)], 1) for i in range(N)])
    take = M[np.argsort(-M[:,0])[:max(0,S-N)],1].astype(int)
    ko = np.minimum(np.ones(N,int)+np.bincount(take,minlength=N), caps)
    ku = np.minimum(np.full(N, S//N), caps)
    b = float(np.mean([cs[i][ku[i]-1] for i in range(N)]))
    return 100*(b - float(np.mean([cs[i][ko[i]-1] for i in range(N)])))/b

OUT = {"note": "per-source p95 of the instance-wise null; the max statistic "
                "diverges with the number of sources examined, which is why the "
                "paper uses median (primary) and p90 (conservative).",
       "config": {"floor_rep": FLOOR_REP, "floor_perm": FLOOR_PERM, "S_per_N": S_PER},
       "cells": {}}
for s in ["POMO", "AM", "SYMNCO"]:
    for ax in ["A", "B"]:
        z = np.load(f"expand_{s}_axis{ax}.npz", allow_pickle=True)
        meta = json.loads(str(z["meta"])); inst = [dict(m) for m in meta]
        for d, m in zip(inst, meta): d[ax] = z[f"c{m['id']}"]
        uni = [d for d in inst if d["dist"] == "uniform"]; N = len(uni)
        kmax = min(len(d[ax]) for d in uni); s_per = min(S_PER, kmax//4)
        rng = np.random.default_rng(40); p95 = []
        for d in uni:
            zz = d[ax][:kmax]/d["ref"]; ds = []
            for _ in range(FLOOR_REP):
                cs = [curve(rng.choice(zz, kmax, True), 1.0) for _ in range(N)]
                ds.append(d_in_null(cs, s_per))
            p95.append(float(np.percentile(ds, 95)))
        p95 = np.array(p95)
        maxes = {str(m): float(np.mean([p95[rng.choice(N, m, replace=False)].max()
                                        for _ in range(200)])) for m in (8, 12, 25, 50)}
        OUT["cells"][f"{s}_axis{ax}"] = {
            "N": N, "K": int(kmax), "S_per_N": int(s_per),
            "median": float(np.median(p95)), "p90": float(np.percentile(p95, 90)),
            "mean": float(p95.mean()), "max_by_n_sources": maxes}
        print(f"{s:>7} {ax}  median={np.median(p95):6.3f}  p90={np.percentile(p95,90):6.3f}  "
              f"max@8/12/25/50 = " + "/".join(f"{maxes[k]:.2f}" for k in ("8","12","25","50")),
              flush=True)
json.dump(OUT, open("floor_definition_sweep.json", "w"), indent=1)
print("\nsaved floor_definition_sweep.json")
