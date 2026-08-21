# 검정 U (사전 명세, rq1-preregistration.md §11.1) — 절대 길이 단위 재계산.
#
# 등록 문구 "gap%를 최적 길이로 되곱한다" 를 문자 그대로 구현한다:
#     d_abs = mean_i( ev_uniform_i * ref_i / 100 ) - mean_i( ev_allocated_i * ref_i / 100 )
# 어느 단계에도 길이 관련 분모가 없다.  등록된 gap 단위 d_split 은 마지막에 세트 평균
# baseline 으로 나누므로, 인스턴스 수준 분모는 없애도 세트 수준 분모가 남는다.
# 두 관례를 모두 계산해 부호·유의성이 일치하는지 함께 보고한다.
#
# CI 는 등록된 방식과 동일: 인스턴스 단위 비모수 부트스트랩, B=400, 백분위수.
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"
X = slice(0, 200); FIT = slice(200, 600); EV = slice(600, 1000)
S_PER, N_PERM, NSET, NSAMP, N_BOOT = 100, 150, 40, 200, 400


def load(p):
    z = np.load(p, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    I = [dict(d) for d in m]
    for d, q in zip(I, m):
        d["c"] = z["c%d" % q["id"]]
    return I


def curve(c, ref, seed):
    K = len(c)
    rng = np.random.default_rng(seed)
    idx = rng.random((N_PERM, K)).argsort(1)
    return 100.0 * (np.minimum.accumulate(np.asarray(c)[idx], 1).mean(0) / ref - 1.0)


def cvx(f):
    n = len(f); xs = np.arange(n, dtype=float); h = []
    for i in range(n):
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (y2 - y1) * (xs[i] - x1) >= (f[i] - y1) * (x2 - x1):
                h.pop()
            else:
                break
        h.append((xs[i], f[i]))
    return np.interp(xs, np.array([p[0] for p in h]), np.array([p[1] for p in h]))


def presid(a, *cov):
    ra = rankdata(a)
    C = np.column_stack([rankdata(c) for c in cov] + [np.ones(len(ra))])
    b, *_ = np.linalg.lstsq(C, ra, rcond=None)
    return ra - C @ b


def alloc(g):
    n = len(g); caps = np.array([len(d["fit"]) for d in g])
    S = min(n * S_PER, int(caps.sum()))
    M = np.concatenate([np.stack([g[i]["mg"], np.full(len(g[i]["mg"]), i)], 1)
                        for i in range(n)])
    take = M[np.argsort(-M[:, 0])[:max(0, S - n)], 1].astype(int)
    return (np.minimum(np.ones(n, int) + np.bincount(take, minlength=n), caps),
            np.minimum(np.full(n, S // n), caps))


def gains(g):
    """(gap 단위 등록 d_split, 순수 절대 차이 d_abs)."""
    ko, ku = alloc(g); n = len(g)
    bu = np.mean([g[i]["ev"][ku[i] - 1] for i in range(n)])
    bo = np.mean([g[i]["ev"][ko[i] - 1] for i in range(n)])
    au = np.mean([g[i]["ev"][ku[i] - 1] * g[i]["ref"] / 100.0 for i in range(n)])
    ao = np.mean([g[i]["ev"][ko[i] - 1] * g[i]["ref"] / 100.0 for i in range(n)])
    return 100 * (bu - bo) / bu, au - ao


def prep(path):
    I = load(path)
    base = [d for d in I if d["sigma"] < 0]
    shift = [d for d in I if d["sigma"] > 0]
    b1 = np.mean([100 * (d["c"][X].mean() / d["ref"] - 1) for d in base])
    for d in base + shift:
        d["r"] = (100 * (d["c"][X].mean() / d["ref"] - 1)) / b1
        d["fit"] = curve(d["c"][FIT], d["ref"], 1)
        d["ev"] = curve(d["c"][EV], d["ref"], 2)
        d["mg"] = -np.diff(cvx(d["fit"]))
    return shift


def sets(shift, seed=5):
    P = sorted(shift, key=lambda d: d["r"]); n = len(P)
    rng = np.random.default_rng(seed); out = []
    for _ in range(NSAMP):
        lo = rng.integers(0, n - NSET)
        w = rng.integers(NSET, n - lo + 1)
        idx = sorted(rng.choice(np.arange(lo, lo + w), NSET, replace=False))
        g = [P[i] for i in idx]
        dg, da = gains(g)
        out.append((np.array(idx),
                    float(np.std(np.log([d["r"] for d in g]))),
                    float(np.median([d["ref"] for d in g])), dg, da))
    return out, n


def run(path, tag):
    S = sets(prep(path))
    rows, n = S
    H = np.array([r[1] for r in rows]); R = np.array([r[2] for r in rows])
    DG = np.array([r[3] for r in rows]); DA = np.array([r[4] for r in rows])
    pg = spearmanr(presid(R, H), presid(DG, H))[0]
    pa = spearmanr(presid(R, H), presid(DA, H))[0]
    rb = np.random.default_rng(9); vg, va = [], []
    for _ in range(N_BOOT):
        keep = set(rb.choice(n, n, replace=True).tolist())
        sub = [r for r in rows if len(set(r[0].tolist()) & keep) >= NSET * 0.6]
        if len(sub) < 30:
            continue
        h = np.array([r[1] for r in sub]); rr = np.array([r[2] for r in sub])
        dg = np.array([r[3] for r in sub]); da = np.array([r[4] for r in sub])
        vg.append(spearmanr(presid(rr, h), presid(dg, h))[0])
        va.append(spearmanr(presid(rr, h), presid(da, h))[0])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    print("%-18s gap %+.3f [%+.3f,%+.3f]   abs %+.3f [%+.3f,%+.3f]   (B_eff %d)"
          % (tag, pg, q(vg)[0], q(vg)[1], pa, q(va)[0], q(va)[1], len(vg)), flush=True)
    return dict(tag=tag, gap=float(pg), ci_gap=list(q(vg)),
                abs_=float(pa), ci_abs=list(q(va)), n_boot_eff=len(vg))


if __name__ == "__main__":
    print("검정 U — tour length | spread,  두 단위 관례")
    out = []
    for arm, pref in [("cluster", "rq1lv"), ("size", "rq1sz")]:
        for s in ["AM", "SYMNCO"]:
            out.append(run(os.path.join(DATA, "%s_%s.npz" % (pref, s)), "%s / %s" % (arm, s)))
    json.dump(out, open("test_u_absolute.json", "w"), indent=1)
    print("saved test_u_absolute.json")
