"""Does reallocation still pay when every sampled tour is improved by 2-opt?

For the mixed collection (re-collected tours, 400 samples per instance), apply
2-opt to every sampled tour, giving post-search costs. Then, exactly as for raw
samples, build exact best-of-k curves (k <= 400, on all 400 post-search costs)
and compute the oracle headroom of reallocation at u = 100 (in-sample plug-in,
as in Section 5), plus the same for the raw costs for comparison. Instance
bootstrap B = 400.
"""
import json
import numpy as np
import reanalysis as R
import ls_analysis as L

U, CAP = 100, 400


def main(solvers):
    import sys, time
    I = np.load("ls_tours/instances.npz"); xy, ref = I["coords"], I["ref"]
    out = {}
    for s in solvers:
        t0 = time.time()
        Z = np.load(f"ls_tours/{s}.npz"); costs, tours = Z["costs"], Z["tours"]
        N = len(costs)
        post = np.zeros_like(costs)
        for i in range(N):
            for j in range(costs.shape[1]):
                post[i, j] = L.tour_len(xy[i], L.two_opt(xy[i], tours[i][j]))
            if (i + 1) % 10 == 0: print(s, i + 1, "/", N, round(time.time() - t0), "s", flush=True)
        np.save(f"ls_tours/{s}_post2opt.npy", post)
        Fr = np.stack([R.exact(100 * (costs[i] / ref[i] - 1), CAP) for i in range(N)])
        Fp = np.stack([R.exact(100 * (post[i] / ref[i] - 1), CAP) for i in range(N)])
        res = {}
        for name, F in (("raw", Fr), ("post2opt", Fp)):
            rb = np.random.default_rng(33); v = []
            for _ in range(400):
                ix = rb.choice(N, N, replace=True); v.append(R.headroom(F[ix], U))
            k = R.greedy(F, U)
            L_u = (ref * (1 + F[:, U - 1] / 100)).sum(); L_k = (ref * (1 + F[np.arange(N), k - 1] / 100)).sum()
            res[name] = dict(uniform_gap=float(F[:, U - 1].mean()), headroom=float(R.headroom(F, U)),
                             ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                             length_reduction=float(100 * (L_u - L_k) / L_u))
        out[s] = res
        print(s, json.dumps(res), flush=True)
    json.dump(out, open("results/ls_curves_" + "_".join(solvers) + ".json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(sys.argv[1].split(","))
