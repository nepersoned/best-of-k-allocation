"""Reallocation versus, and combined with, 2-opt local search (mixed collection).

Uses the re-collected samples (ls_collect.py): 400 stochastic samples with tours per
instance, 50 uniform + 50 clustered TSP-100 instances. Budget S = 100 N samples.
Policies (realized, not exact curves, since tours are needed):
  uniform   : the first 100 samples of every instance
  realloc   : the two-stage fitted-curve rule of fit_baselines.py (probe m = 20, charged);
              decisions use only the first min(k, 200) samples; cap 400
Each is reported as the mean gap of the best sampled tour, and after 2-opt applied to
that best tour (first-improvement 2-opt to a local optimum). Paired instance bootstrap.
Wall-clock: mean seconds per instance for 2-opt, against the solver's sampling time
for 100 samples (from ls_collect.log).
"""
import json, sys, time
import numpy as np
import fit_baselines as FB

D = sys.argv[1] if len(sys.argv) > 1 else "ls_tours"
U, CAP, OBS, M = 100, 400, 200, 20


def tour_len(xy, t):
    q = xy[t]; return float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum())


def two_opt(xy, t):
    """Best-improvement 2-opt to a local optimum; each step evaluates all moves at once."""
    t = np.array(t, dtype=int); n = len(t)
    xy = np.asarray(xy, dtype=np.float64)
    Dm = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    I, J = np.triu_indices(n, 2)
    keep = ~((I == 0) & (J == n - 1)); I, J = I[keep], J[keep]
    while True:
        a, b = t[I], t[I + 1]; c, d = t[J], t[(J + 1) % n]
        delta = Dm[a, c] + Dm[b, d] - Dm[a, b] - Dm[c, d]
        k = np.argmin(delta)
        if delta[k] >= -1e-9:
            return t
        i, j = I[k], J[k]
        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]


def realloc_counts(costs):
    N = len(costs); S = U * N
    k0 = np.full(N, M); rest = S - M * N
    curves = [FB.fit_curve(costs[i][:M]) for i in range(N)]
    half = rest // 2
    ka = FB.greedy_extra(curves, k0, half)
    curves2 = [FB.fit_curve(costs[i][:min(ka[i], OBS)]) for i in range(N)]
    return FB.greedy_extra(curves2, ka, rest - half)


def main():
    I = np.load(f"{D}/instances.npz"); xy, ref, dist = I["coords"], I["ref"], I["dist"]
    out = {}
    for s in ["AM", "SYMNCO", "POMO"]:
        try:
            Z = np.load(f"{D}/{s}.npz")
        except FileNotFoundError:
            continue
        costs, tours = Z["costs"], Z["tours"]
        N = len(costs)
        kr = realloc_counts(costs)
        rows = {}
        t_ls = []
        for name, k in [("uniform", np.full(N, U)), ("realloc", kr)]:
            g, g2 = [], []
            for i in range(N):
                j = int(np.argmin(costs[i][:k[i]]))
                g.append(100 * (costs[i][j] / ref[i] - 1))
                t0 = time.time(); t2 = two_opt(xy[i], tours[i][j]); t_ls.append(time.time() - t0)
                g2.append(100 * (tour_len(xy[i], t2) / ref[i] - 1))
            rows[name] = np.array(g); rows[name + "+2opt"] = np.array(g2)
        rb = np.random.default_rng(33); keys = list(rows)
        boot = {k: [] for k in keys}; pairs = {p: [] for p in ["realloc-uniform", "uniform+2opt-uniform", "realloc+2opt-uniform+2opt", "realloc-uniform+2opt"]}
        for _ in range(2000):
            ix = rb.choice(N, N, replace=True)
            m = {k: rows[k][ix].mean() for k in keys}
            for k in keys: boot[k].append(m[k])
            pairs["realloc-uniform"].append(m["realloc"] - m["uniform"])
            pairs["uniform+2opt-uniform"].append(m["uniform+2opt"] - m["uniform"])
            pairs["realloc+2opt-uniform+2opt"].append(m["realloc+2opt"] - m["uniform+2opt"])
            pairs["realloc-uniform+2opt"].append(m["realloc"] - m["uniform+2opt"])
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res = {k: dict(mean_gap=float(rows[k].mean()), ci=q(boot[k])) for k in keys}
        res["paired_gap_differences"] = {
            "realloc-uniform": dict(point=float(rows["realloc"].mean() - rows["uniform"].mean()), ci=q(pairs["realloc-uniform"])),
            "uniform+2opt-uniform": dict(point=float(rows["uniform+2opt"].mean() - rows["uniform"].mean()), ci=q(pairs["uniform+2opt-uniform"])),
            "realloc+2opt-uniform+2opt": dict(point=float(rows["realloc+2opt"].mean() - rows["uniform+2opt"].mean()), ci=q(pairs["realloc+2opt-uniform+2opt"])),
            "realloc-uniform+2opt": dict(point=float(rows["realloc"].mean() - rows["uniform+2opt"].mean()), ci=q(pairs["realloc-uniform+2opt"])),
        }
        res["headroom_vs_uniform_pct"] = float(100 * (rows["uniform"].mean() - rows["realloc"].mean()) / rows["uniform"].mean())
        res["headroom_after_2opt_pct"] = float(100 * (rows["uniform+2opt"].mean() - rows["realloc+2opt"].mean()) / rows["uniform+2opt"].mean())
        res["two_opt_seconds_per_instance"] = float(np.mean(t_ls))
        res["by_dist"] = {dd: {k: float(rows[k][dist == dd].mean()) for k in keys} for dd in ("uniform", "clustered")}
        out[s] = res
        print(s, json.dumps(res), flush=True)
    json.dump(out, open("results/ls_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
