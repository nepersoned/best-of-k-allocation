# 확증 실행 분석 — 등록 사양 그대로, 종점 외 검정 없음
#   주 검정   : AM  축A 혼합, d_split CI 하한 >= 2%
#   2차 종점  : 잔차 = d_split(탐욕) - d_split(동결 라벨) , CI 하한 > 0
#   복제      : SYMNCO 축A 혼합 (메커니즘 복제, 비표준 프로토콜)
#   음성 대조 : POMO  축A 혼합 (예상: d_split CI가 0을 포함)
#   라벨 기준선 동결: AM 20:1, SYMNCO 12:1, POMO 1.5:1 — 재적합 없음
#   바닥: 원천별 p95의 median(주) + p90(보수) — max 폐기
import json, warnings, numpy as np
warnings.filterwarnings("ignore")

SEED_ORDERING, SEED_BOOTSTRAP, SEED_SPLIT = 22, 33, 29     # 동결
N_PERM, N_BOOT = 200, 300
FLOOR_REP, FLOOR_PERM = 6, 60
S_PER = 100
LABEL_W = {"AM": 20.0, "SYMNCO": 12.0, "POMO": 1.5}        # 동결
ROLE   = {"AM": "주 검정", "SYMNCO": "복제", "POMO": "음성 대조"}
OUT = []
def P(s=""):
    print(s, flush=True); OUT.append(s)

def curve(c, ref, kmax=None, n_perm=N_PERM, seed=SEED_ORDERING):
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
def prep(cs): return [(-np.diff(cvx(c))) for c in cs]
def greedy(margs, curves, idx, S):
    N = len(idx)
    M = np.concatenate([np.stack([margs[i], np.full(len(margs[i]), j)], 1)
                        for j, i in enumerate(idx)])
    take = M[np.argsort(-M[:,0])[:max(0,S-N)],1].astype(int)
    caps = np.array([len(curves[i]) for i in idx])
    return np.minimum(np.ones(N,int)+np.bincount(take,minlength=N), caps)
def label_alloc(is_clu, caps, S, w):
    """동결 비율 w:1 로 정확히 S 를 쓰는 배분 (잔여 분배 — 라벨 정책에 유리한 쪽)"""
    wt = np.where(is_clu, w, 1.0); wt = wt/wt.sum()
    k = np.maximum(1, np.floor(S*wt).astype(int))
    k = np.minimum(k, caps)
    # 잔여를 가중치 큰 순서로 1씩 분배
    order = np.argsort(-wt)
    while k.sum() < S:
        moved = False
        for i in order:
            if k[i] < caps[i]:
                k[i] += 1; moved = True
                if k.sum() >= S: break
        if not moved: break
    while k.sum() > S:
        for i in order[::-1]:
            if k[i] > 1:
                k[i] -= 1
                if k.sum() <= S: break
    return k
def ev(curves, idx, k): return float(np.mean([curves[i][k[j]-1] for j,i in enumerate(idx)]))

def endpoints(grp, ax, w):
    """반환 dict: d_split_greedy, d_split_label, residual (+부트스트랩 CI), d_in, 바닥"""
    kmax = min(len(d[ax]) for d in grp); S_per = min(S_PER, kmax//4)
    is_clu_all = np.array([d["dist"] == "clustered" for d in grp])
    # 표본분할 (동결 시드)
    rs = np.random.default_rng(SEED_SPLIT); fit, evl = [], []
    for d in grp:
        c = d[ax][:kmax].copy(); rs.shuffle(c); h = len(c)//2
        fit.append(curve(c[:h], d["ref"], h, N_PERM, SEED_ORDERING+1))
        evl.append(curve(c[h:], d["ref"], h, N_PERM, SEED_ORDERING+2))
    mf = prep(fit)
    full = [curve(d[ax][:kmax], d["ref"]) for d in grp]; mfull = prep(full)

    def three(idx):
        N = len(idx); caps = np.array([len(fit[i]) for i in idx])
        S = min(N*S_per, int(caps.sum()))
        ku = np.minimum(np.full(N, S//N), caps)
        kg = greedy(mf, fit, idx, S)
        kl = label_alloc(is_clu_all[idx], caps, S, w)
        b = ev(evl, idx, ku)
        dg = 100*(b - ev(evl, idx, kg))/b
        dl = 100*(b - ev(evl, idx, kl))/b
        return dg, dl, dg-dl
    N = len(grp); ix = np.arange(N)
    dg, dl, res = three(ix)
    rng = np.random.default_rng(SEED_BOOTSTRAP)
    B = np.array([three(rng.integers(0,N,N)) for _ in range(N_BOOT)])
    ci = lambda j: (float(np.percentile(B[:,j],2.5)), float(np.percentile(B[:,j],97.5)))
    # 보조: d_in, 바닥(median/p90)
    caps=np.array([len(c) for c in full]); S=min(N*S_per,int(caps.sum()))
    ku=np.minimum(np.full(N,S//N),caps); ko=greedy(mfull, full, ix, S)
    b=ev(full,ix,ku); d_in=100*(b-ev(full,ix,ko))/b
    rg=np.random.default_rng(SEED_BOOTSTRAP+7); p95=[]
    for d in grp:
        z=d[ax][:kmax]/d["ref"]; ds=[]
        for _ in range(FLOOR_REP):
            cs=[curve(rg.choice(z,kmax,True),1.0,n_perm=FLOOR_PERM) for _ in range(N)]
            m2=prep(cs); c2=np.array([len(c) for c in cs]); S2=min(N*S_per,int(c2.sum()))
            k2=np.minimum(np.full(N,S2//N),c2); g2=greedy(m2,cs,ix,S2)
            bb=ev(cs,ix,k2); ds.append(100*(bb-ev(cs,ix,g2))/bb)
        p95.append(np.percentile(ds,95))
    p95=np.array(p95)
    return dict(dg=dg, dg_ci=ci(0), dl=dl, dl_ci=ci(1), res=res, res_ci=ci(2),
                d_in=d_in, fl_med=float(np.median(p95)), fl_p90=float(np.percentile(p95,90)))

P("="*80); P("확증 실행 — 사전등록 종점만 검정"); P("="*80)
P("시드: instance=20260813(신), sampling=77(신), ordering=22(동결), bootstrap=33(동결)")
P("라벨 기준선 동결: " + str(LABEL_W) + "  (재적합 없음, 정확히 S 소진)")
P("")
RES = {}
for s in ["AM", "SYMNCO", "POMO"]:
    z = np.load(f"confirm_{s}_axisA.npz", allow_pickle=True)
    meta = json.loads(str(z["meta"])); inst=[dict(m) for m in meta]
    for d, m in zip(inst, meta): d["A"] = z[f"c{m['id']}"]
    uni=[d for d in inst if d["dist"]=="uniform"]; clu=[d for d in inst if d["dist"]=="clustered"]
    grp = uni[:25] + clu[:25]
    gu=np.mean([100*(d["A"].min()/d["ref"]-1) for d in uni])
    gc=np.mean([100*(d["A"].min()/d["ref"]-1) for d in clu])
    r = endpoints(grp, "A", LABEL_W[s]); r["ratio"]=gc/max(gu,1e-9); r["gu"]=gu; r["gc"]=gc
    RES[s]=r
    P("-"*80); P(f"{s}  ({ROLE[s]})   축 A · 혼합(uni25+clu25)   N={len(grp)}")
    P(f"  OOD 배율: uniform {gu:.2f}%  clustered {gc:.2f}%  -> {r['ratio']:.1f}x")
    P(f"  [주] d_split(탐욕)  = {r['dg']:7.3f}   CI [{r['dg_ci'][0]:7.3f}, {r['dg_ci'][1]:7.3f}]")
    P(f"       d_split(라벨)  = {r['dl']:7.3f}   CI [{r['dl_ci'][0]:7.3f}, {r['dl_ci'][1]:7.3f}]")
    P(f"  [2차] 잔차          = {r['res']:7.3f}   CI [{r['res_ci'][0]:7.3f}, {r['res_ci'][1]:7.3f}]")
    P(f"  (보조) d_in={r['d_in']:.3f}  바닥 median={r['fl_med']:.3f} p90={r['fl_p90']:.3f}")

P(""); P("="*80); P("사전등록 판정"); P("="*80)
am = RES["AM"]; sy = RES["SYMNCO"]; po = RES["POMO"]
p1 = am["dg_ci"][0] >= 2.0
p2 = am["res_ci"][0] > 0
rep = sy["dg_ci"][0] >= 2.0
neg = po["dg_ci"][0] <= 0 <= po["dg_ci"][1]
P(f"  주 검정  AM d_split CI 하한 {am['dg_ci'][0]:.3f} >= 2%      -> {'통과' if p1 else '미통과'}")
P(f"  2차 종점 AM 잔차 CI 하한 {am['res_ci'][0]:.3f} > 0          -> {'통과' if p2 else '미통과'}")
P(f"  복제     SYMNCO d_split CI 하한 {sy['dg_ci'][0]:.3f} >= 2%  -> {'통과' if rep else '미통과'}")
P(f"  음성대조 POMO d_split CI [{po['dg_ci'][0]:.3f}, {po['dg_ci'][1]:.3f}] 가 0 포함 -> {'예상대로' if neg else '예상과 다름'}")
P("")
P("  => 논문 구조: " + ("감사 + 구성 2절" if p1 else "감사 단독 (②는 탐색 관찰 각주)"))
open("confirm_report.txt","w",encoding="utf-8").write("\n".join(OUT))
print("\nsaved confirm_report.txt", flush=True)
