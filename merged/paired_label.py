"""Paired bootstrap: charged probe (median-min) rule minus label 20:1, rq3, exact curves."""
import json, numpy as np, reanalysis as R, reanalysis2 as R2
out = {}
for s in ["AM", "SYMNCO", "POMO"]:
    inst = R.load1(R.P2 + f"rq3_{s}_axisA.npz")
    F = np.stack([R.exact(100 * (c[R.POOL] / m["ref"] - 1), R.CAP) for m, c in inst])
    probe = np.stack([c[R.X][:20] for m, c in inst]); med = np.median(probe, 1)
    st = (med - probe.min(1)) / med
    w0 = np.array([20.0 if m["dist"] == "clustered" else 1.0 for m, _ in inst])
    def lab(ix):
        w = w0[ix]; S = R.SN * len(ix); k = np.maximum(1, np.floor(S * w / w.sum()).astype(int))
        k[np.argsort(-w)[: S - k.sum()]] += 1
        return R.gain(F[ix], np.minimum(k, R.CAP))
    f = lambda ix: R2.charged(F[ix], st[ix], 20) - lab(ix)
    rb = np.random.default_rng(33); N = len(F)
    v = [f(rb.choice(N, N, replace=True)) for _ in range(R.N_BOOT)]
    out[s] = dict(point=float(f(np.arange(N))), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    print(s, out[s], flush=True)
json.dump(out, open("results/paired_label.json", "w"), indent=1)
