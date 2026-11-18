"""Successive-halving baseline for instance-wise allocation (policy collection, rq3).

Same accounting as the probe rules in reanalysis2.py: every sample spent while
deciding is charged to the total budget S = u N and kept as a candidate; decisions
use only samples from slice [0:200]; each instance is scored at F_i(k_i), the exact
best-of-k curve on the disjoint pool [200:1000]; per-instance cap 400, and samples
above the cap are lost (as for the probe rules).

Successive halving (Karnin et al. 2013 style, adapted from arms to instances):
optionally every instance first receives `floor` samples (charged); the halving
phase then uses half of the remaining budget, split equally over R rounds; in each
round every surviving instance receives an equal share, then the half with the
lowest score is dropped (they keep what they already have). Halving stops early if
a further round would need more than 200 observed samples per survivor. The rest
of the budget is split equally over the final survivors.

Scores (computed on raw costs of the samples observed so far; no reference tour):
  medmin : (median - min) / median            (the probe statistic of Sect. 7)
  improve: relative drop of the running minimum during the last round
"""
import json
import numpy as np
import reanalysis as R

U, CAP, OBS = 100, 400, 200


def score(obs, kind, prev_min):
    if kind == "medmin":
        med = np.median(obs)
        return (med - obs.min()) / med
    return (prev_min - obs.min()) / prev_min


def successive_halving(costs, F, kind, phase, ix, floor=0):
    N = len(ix); S = U * N
    rounds = int(np.ceil(np.log2(N)))
    k = np.full(N, floor, int)
    alive = np.arange(N)
    spent = floor * N
    per_round = phase * (S - spent) / rounds
    prev_min = np.full(N, np.inf)
    dropped = []                      # most recently dropped first
    for r in range(rounds):
        if len(alive) <= 1:
            break
        add = int(per_round // len(alive))
        if add < 1 or (k[alive] + add).max() > OBS:
            break
        rest_after = S - spent - add * len(alive)
        if (len(alive) // 2) * CAP < rest_after:      # survivors could not absorb the rest
            break
        before = np.array([costs[ix[i]][:k[i]].min() if k[i] else np.inf for i in alive])
        k[alive] += add; spent += add * len(alive)
        sc = np.array([score(costs[ix[i]][:k[i]], kind,
                             before[j] if np.isfinite(before[j]) else costs[ix[i]][:k[i]].max())
                       for j, i in enumerate(alive)])
        order = np.argsort(-sc)
        keep, drop = order[: max(1, len(alive) // 2)], order[max(1, len(alive) // 2):]
        dropped = list(alive[drop][::-1]) + dropped
        alive = alive[keep]
    # remaining budget: survivors first (up to the cap), overflow to the most recently
    # dropped instances, tier by tier, so no sample is lost while capacity remains
    rest = S - spent
    tiers = [alive] + [np.array([d]) for d in dropped]
    for t in tiers:
        if rest <= 0:
            break
        room = CAP - k[t]
        while rest > 0 and room.sum() > 0:
            share = max(1, rest // max(1, (room > 0).sum()))
            give = np.minimum(room, share) * (room > 0)
            if give.sum() > rest:
                give = np.zeros_like(give); give[np.argmax(room)] = rest
            k[t] += give; rest -= int(give.sum()); room = CAP - k[t]
    k = np.maximum(1, np.minimum(k, CAP))
    return R.gain(F[ix], k, U), int(spent), len(alive)


def run(solver, n_boot=R.N_BOOT):
    inst = R.load1(R.P2 + f"rq3_{solver}_axisA.npz")
    costs = [c[R.X] for m, c in inst]
    F = np.stack([R.exact(100 * (c[R.POOL] / m["ref"] - 1), CAP) for m, c in inst])
    probe = np.stack([c[:20] for c in costs]); med = np.median(probe, 1)
    st = (med - probe.min(1)) / med
    import reanalysis2 as R2
    def evaluate(ix):
        out = {"probe_medmin": R2.charged(F[ix], st[ix], 20)}
        for kind in ("medmin", "improve"):
            for fl in (0, 20, 50):
                out[f"sh_{kind}_f{fl}"] = successive_halving(costs, F, kind, 0.5, ix, fl)[0]
        return out
    N = len(F); full = np.arange(N)
    pt = evaluate(full)
    diag = {f"sh_{kind}_f{fl}": successive_halving(costs, F, kind, 0.5, full, fl)[1:]
            for kind in ("medmin", "improve") for fl in (0, 20, 50)}
    rb = np.random.default_rng(33); B = {k: [] for k in pt}; D = {k: [] for k in pt if k != "probe_medmin"}
    for _ in range(n_boot):
        ix = rb.choice(N, N, replace=True)
        e = evaluate(ix)
        for k in pt:
            B[k].append(e[k])
        for k in D:
            D[k].append(e[k] - e["probe_medmin"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {k: dict(point=float(pt[k]), ci=q(B[k])) for k in pt}
    res["minus_probe"] = {k: dict(point=float(pt[k] - pt["probe_medmin"]), ci=q(v)) for k, v in D.items()}
    res["diag_spent_survivors"] = diag
    return res


if __name__ == "__main__":
    O = {}
    for s in ["AM", "SYMNCO", "POMO"]:
        O[s] = run(s)
        print(s, json.dumps(O[s]), flush=True)
    json.dump(O, open("results/sh_baseline.json", "w"), indent=1)
