"""확증 2 분석 — rq1-preregistration.md §9 의 등록 종점만 계산한다.

동결 시각: 수집 완료 전. 실행 후 수정하지 않는다.
입력:  팔 A  rq1lv_{AM,SYMNCO,POMO}.npz   (군집 조임 손잡이, 시드 31415926/2718)
       팔 B  rq1sz_{AM,SYMNCO,POMO}.npz   (크기 손잡이,     시드 27182818/1618)
출력:  콘솔 판정표 + confirm2_results.json

등록된 절차 (§9.4):
  - 배열 3분할  [0:200] 축 / [200:600] 배분 / [600:1000] 평가
  - 세트 200개 × 40 인스턴스, 무작위 창
  - 이질성 = 세트 내 log(severity) 표준편차,  수준 = 세트 내 severity 중앙값
  - 부분상관은 순위 잔차화
  - CI = 인스턴스 단위 비모수 부트스트랩, B=400, 백분위수 방식 (BCa 아님)
"""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata

X = slice(0, 200); FIT = slice(200, 600); EV = slice(600, 1000)
S_PER, N_PERM, NSET, NSAMP, N_BOOT = 100, 150, 40, 200, 400
RHO_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
RHO_NSET, RHO_REP = 30, 25


def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    inst = [dict(d) for d in meta]
    for d, q in zip(inst, meta):
        d["c"] = z[f"c{q['id']}"]
    return inst


def curve(c, ref, seed):
    K = len(c)
    rng = np.random.default_rng(seed)
    idx = rng.random((N_PERM, K)).argsort(1)
    return 100.0 * (np.minimum.accumulate(np.asarray(c)[idx], 1).mean(0) / ref - 1.0)


def cvx(f):
    """하부 볼록 껍질 — 추정 잡음으로 비단조가 된 한계이득을 정칙화."""
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


def prep(path):
    inst = load(path)
    base = [d for d in inst if d["sigma"] < 0]
    shift = [d for d in inst if d["sigma"] > 0]
    b1 = np.mean([100 * (d["c"][X].mean() / d["ref"] - 1) for d in base])
    for d in base + shift:
        d["r"] = (100 * (d["c"][X].mean() / d["ref"] - 1)) / b1
        d["fit"] = curve(d["c"][FIT], d["ref"], 1)
        d["ev"] = curve(d["c"][EV], d["ref"], 2)
        d["mg"] = -np.diff(cvx(d["fit"]))
    return base, shift, float(b1)


def dsplit(g):
    n = len(g)
    caps = np.array([len(d["fit"]) for d in g])
    S = min(n * S_PER, int(caps.sum()))
    M = np.concatenate([np.stack([g[i]["mg"], np.full(len(g[i]["mg"]), i)], 1) for i in range(n)])
    take = M[np.argsort(-M[:, 0])[:max(0, S - n)], 1].astype(int)
    ko = np.minimum(np.ones(n, int) + np.bincount(take, minlength=n), caps)
    ku = np.minimum(np.full(n, S // n), caps)
    b = np.mean([g[i]["ev"][ku[i] - 1] for i in range(n)])
    return 100 * (b - np.mean([g[i]["ev"][ko[i] - 1] for i in range(n)])) / b


def presid(a, *covs):
    """순위 잔차화. 공변량 하나 이상."""
    ra = rankdata(a)
    Cm = np.column_stack([rankdata(c) for c in covs] + [np.ones(len(ra))])
    beta, *_ = np.linalg.lstsq(Cm, ra, rcond=None)
    return ra - Cm @ beta


def build_sets(shift, seed=5):
    P = sorted(shift, key=lambda d: d["r"]); n = len(P)
    rng = np.random.default_rng(seed); out = []
    for _ in range(NSAMP):
        lo = rng.integers(0, n - NSET)
        w = rng.integers(NSET, n - lo + 1)
        idx = sorted(rng.choice(np.arange(lo, lo + w), NSET, replace=False))
        g = [P[i] for i in idx]
        out.append((np.array(idx),
                    float(np.std(np.log([d["r"] for d in g]))),
                    float(np.median([d["r"] for d in g])),
                    dsplit(g)))
    return out, n


def boot_two(sets, n, seed=9):
    """spread|level 과 level|spread 를 같은 재표집 위에서."""
    rb = np.random.default_rng(seed); vs, vl = [], []
    for _ in range(N_BOOT):
        keep = set(rb.choice(n, n, replace=True).tolist())
        sub = [(h, l, d) for ix, h, l, d in sets
               if len(set(ix.tolist()) & keep) >= NSET * 0.6]
        if len(sub) < 30:
            continue
        H = np.array([x[0] for x in sub]); L = np.array([x[1] for x in sub])
        D = np.array([x[2] for x in sub])
        vs.append(spearmanr(presid(H, L), presid(D, L))[0])
        vl.append(spearmanr(presid(L, H), presid(D, H))[0])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(vs), q(vl), len(vs)


def arm(path, label):
    base, shift, b1 = prep(path)
    sets, n = build_sets(shift)
    H = np.array([s[1] for s in sets]); L = np.array([s[2] for s in sets])
    D = np.array([s[3] for s in sets])
    sp = spearmanr(presid(H, L), presid(D, L))[0]
    lv = spearmanr(presid(L, H), presid(D, H))[0]
    ci_s, ci_l, nb = boot_two(sets, n)
    return dict(label=label, n_shift=n, base_gap=b1,
                het_range=[float(H.min()), float(H.max())],
                d_range=[float(D.min()), float(D.max())],
                spread_given_level=float(sp), ci_spread=list(ci_s),
                level_given_spread=float(lv), ci_level=list(ci_l),
                n_boot_eff=nb)


def three_way(path):
    """2차 종점 — rho | (spread, level).  구성 축이 필요하므로 세트를 rho 별로 구성."""
    base, shift, _ = prep(path)
    rows = []
    for rho in RHO_LEVELS:
        for rep in range(RHO_REP):
            rng = np.random.default_rng(3000 + rep)
            nO = int(round(rho * RHO_NSET)); nU = RHO_NSET - nO
            if nU > len(base):
                continue
            ids = (list(rng.choice(len(base), nU, replace=False)),
                   list(rng.choice(len(shift), nO, replace=False)))
            g = [base[i] for i in ids[0]] + [shift[i] for i in ids[1]]
            rows.append((rho,
                         float(np.std(np.log([d["r"] for d in g]))),
                         float(np.median([d["r"] for d in g])),
                         dsplit(g),
                         [("B", i) for i in ids[0]] + [("S", i) for i in ids[1]]))
    P = np.array([r[0] for r in rows]); H = np.array([r[1] for r in rows])
    L = np.array([r[2] for r in rows]); D = np.array([r[3] for r in rows])
    pt = spearmanr(presid(P, H, L), presid(D, H, L))[0]
    npool = len(base) + len(shift)
    rb = np.random.default_rng(4); v = []
    for _ in range(N_BOOT):
        keep = set(rb.choice(npool, npool, replace=True).tolist())
        idx = [i for i, r in enumerate(rows)
               if sum(1 for t, j in r[4] if (j if t == "B" else len(base) + j) in keep)
               >= RHO_NSET * 0.6]
        if len(idx) < 40:
            continue
        v.append(spearmanr(presid(P[idx], H[idx], L[idx]),
                           presid(D[idx], H[idx], L[idx]))[0])
    return dict(point=float(pt),
                ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                n_boot_eff=len(v))


if __name__ == "__main__":
    R = {"armA": {}, "armB": {}, "three_way": {}}
    print("=" * 78)
    print("확증 2 — 등록 종점만 (rq1-preregistration.md §9.4)")
    print("=" * 78)

    for tag, prefix in [("armA", "rq1lv"), ("armB", "rq1sz")]:
        print(f"\n### 팔 {tag[-1]} ({'군집 조임' if tag=='armA' else '크기 변이'}) ###")
        print(f"{'solver':>7} {'het range':>16} {'d_split range':>17} "
              f"{'spread|level':>13} {'CI':>18} {'level|spread':>13} {'CI':>18}")
        for s in ["AM", "SYMNCO", "POMO"]:
            try:
                r = arm(f"{prefix}_{s}.npz", s)
            except FileNotFoundError:
                print(f"{s:>7}   (파일 없음: {prefix}_{s}.npz)")
                continue
            R[tag][s] = r
            print(f"{s:>7} {r['het_range'][0]:7.3f}~{r['het_range'][1]:<7.3f} "
                  f"{r['d_range'][0]:8.2f}~{r['d_range'][1]:<7.2f} "
                  f"{r['spread_given_level']:>+13.3f} "
                  f"[{r['ci_spread'][0]:+.3f},{r['ci_spread'][1]:+.3f}] "
                  f"{r['level_given_spread']:>+13.3f} "
                  f"[{r['ci_level'][0]:+.3f},{r['ci_level'][1]:+.3f}]", flush=True)

    print(f"\n### 2차 종점 — rho | (spread, level), 팔 A ###")
    for s in ["AM", "SYMNCO"]:
        try:
            t = three_way(f"rq1lv_{s}.npz")
        except FileNotFoundError:
            continue
        R["three_way"][s] = t
        inc = t["ci"][0] <= 0 <= t["ci"][1]
        print(f"  {s:>7}: {t['point']:+.3f}  CI [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]  "
              f"-> {'0 포함' if inc else '0 배제'}", flush=True)

    print("\n" + "=" * 78)
    print("판정 (§9.5 표)")
    print("=" * 78)
    A = R["armA"]
    if "AM" in A:
        a1 = A["AM"]["ci_spread"][0] > 0
        a2 = A["AM"]["ci_level"][0] > 0
        print(f"  A1  AM spread|level 하한 {A['AM']['ci_spread'][0]:+.3f} > 0  -> "
              f"{'통과' if a1 else '미통과'}")
        print(f"  A2  AM level|spread 하한 {A['AM']['ci_level'][0]:+.3f} > 0  -> "
              f"{'통과' if a2 else '미통과'}")
        if "SYMNCO" in A:
            print(f"  복제 SymNCO spread {A['SYMNCO']['ci_spread'][0]:+.3f} / "
                  f"level {A['SYMNCO']['ci_level'][0]:+.3f}")
        if "POMO" in A:
            p = A["POMO"]
            ok = (p["ci_spread"][0] <= 0 <= p["ci_spread"][1]) and \
                 (p["ci_level"][0] <= 0 <= p["ci_level"][1])
            print(f"  음성대조 POMO spread [{p['ci_spread'][0]:+.3f},{p['ci_spread'][1]:+.3f}] "
                  f"level [{p['ci_level'][0]:+.3f},{p['ci_level'][1]:+.3f}] -> "
                  f"{'예상대로' if ok else '예상과 다름'}")
        cell = ("둘 다 통과" if (a1 and a2) else "spread만" if a1 else
                "level만" if a2 else "둘 다 실패")
        print(f"\n  => §9.5 판정표 칸: 【{cell}】")
    B = R["armB"]
    if "AM" in B:
        b1 = B["AM"]["ci_spread"][0] > 0
        print(f"  B1  (독립) AM spread|level 하한 {B['AM']['ci_spread'][0]:+.3f} > 0  -> "
              f"{'통과' if b1 else '미통과'}")
    json.dump(R, open("confirm2_results.json", "w"), indent=1)
    print("\nsaved confirm2_results.json")
