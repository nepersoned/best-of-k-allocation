# 3솔버 확장 분석 (로컬, GPU 0)
#  1) 솔버별 게이트 재현: 동질 d_in vs 바닥(가짜), 혼합 d_split(진짜)
#  2) 바닥 곡선 (비율 K/(S/N)=10 고정)
#  3) 스케일 붕괴 검정: 바닥 / Φ,  Φ = cv_excess(k)·sqrt(k/K)·sqrt(2 ln N)
import json, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_ORDERING, SEED_BOOTSTRAP = 22, 33
N_PERM, N_BOOT = 200, 300
FLOOR_SRC, FLOOR_REP, FLOOR_PERM = 8, 6, 60
SOLVERS = ["POMO", "AM", "SYMNCO"]
REPORT = []
def P(s=""):
    print(s, flush=True); REPORT.append(s)

# ---------- 공용 ----------
def curve(costs, ref, kmax=None, n_perm=N_PERM, seed=SEED_ORDERING):
    K = len(costs); kmax = kmax or K
    rng = np.random.default_rng(seed)
    idx = rng.random((n_perm, K)).argsort(1)[:, :kmax]
    return 100.0*(np.minimum.accumulate(np.asarray(costs)[idx], 1).mean(0)/ref - 1.0)
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
def alloc(margs, curves, idx, S):
    N = len(idx)
    M = np.concatenate([np.stack([margs[i], np.full(len(margs[i]), j)], 1)
                        for j, i in enumerate(idx)])
    take = M[np.argsort(-M[:,0])[:max(0,S-N)],1].astype(int)
    caps = np.array([len(curves[i]) for i in idx])
    return np.minimum(np.ones(N,int)+np.bincount(take,minlength=N), caps)
def ev(curves, idx, k): return float(np.mean([curves[i][k[j]-1] for j,i in enumerate(idx)]))
def d_in_of(margs, curves, S_per, idx):
    N=len(idx); caps=np.array([len(curves[i]) for i in idx]); S=min(N*S_per,int(caps.sum()))
    ku=np.minimum(np.full(N,S//N),caps); ko=alloc(margs,curves,idx,S)
    b=ev(curves,idx,ku); return 100*(b-ev(curves,idx,ko))/b
def d_split_of(mf, fit, evl, S_per, idx):
    N=len(idx); caps=np.array([len(fit[i]) for i in idx]); S=min(N*S_per,int(caps.sum()))
    ku=np.minimum(np.full(N,S//N),caps); ko=alloc(mf,fit,idx,S)
    b=ev(evl,idx,ku); return 100*(b-ev(evl,idx,ko))/b
def bootci(fn, N, nb=N_BOOT, seed=SEED_BOOTSTRAP):
    rng=np.random.default_rng(seed); v=[fn(rng.integers(0,N,N)) for _ in range(nb)]
    return float(np.percentile(v,2.5)), float(np.percentile(v,97.5))
def floor_p95(grp, ax, kmax, S_per):
    N=len(grp); rng=np.random.default_rng(SEED_BOOTSTRAP+7)
    srcs=[grp[i] for i in rng.choice(len(grp), min(FLOOR_SRC,len(grp)), replace=False)]
    best=-9e9
    for d in srcs:
        z=d[ax][:kmax]/d["ref"]; ds=[]
        for _ in range(FLOOR_REP):
            cs=[curve(rng.choice(z,kmax,True),1.0,n_perm=FLOOR_PERM) for _ in range(N)]
            ds.append(d_in_of(prep(cs),cs,S_per,np.arange(N)))
        best=max(best,float(np.percentile(ds,95)))
    return best

def cv_excess(grp, ax, kmax, k, reps=1500, seed=5):
    """min-of-k 의 표준편차 / (평균 − 기준해)  — 인스턴스 평균.
       분모가 '비용'이 아니라 '기준해 초과분'인 것이 핵심 (gap 수준 차이를 흡수)"""
    rng=np.random.default_rng(seed); out=[]
    for d in grp:
        z=d[ax][:kmax]
        idx=rng.random((reps,len(z))).argsort(1)[:,:k]
        mins=z[idx].min(1)
        exc=mins.mean()-d["ref"]
        if exc>0: out.append(mins.std()/exc)
    return float(np.mean(out))

# ---------- 로드 ----------
DATA={}
for s in SOLVERS:
    for ax in "AB":
        z=np.load(f"expand_{s}_axis{ax}.npz", allow_pickle=True)
        meta=json.loads(str(z["meta"]))
        if s not in DATA: DATA[s]=[dict(m) for m in meta]
        for d,m in zip(DATA[s],meta): d[ax]=z[f"c{m['id']}"]

P("="*78); P("3솔버 확장 — 편향 감사"); P("="*78)

# ---------- 1) 솔버별 '가짜 vs 진짜' 대비 ----------
P(""); P("[1] 같은 보정이 가짜는 죽이고 진짜는 살리는가  (S/N=100)")
P(f"{'solver':>7} {'axis':>5} {'set':>5} {'d_in':>8} {'CI_in':>16} {'d_split':>9} {'CI_split':>16} {'floor':>8}  판정")
TAB={}
for s in SOLVERS:
    inst=DATA[s]
    uni=[d for d in inst if d["dist"]=="uniform"]; clu=[d for d in inst if d["dist"]=="clustered"]
    for ax in "AB":
        for sn, grp in [("동질", uni), ("혼합", uni[:25]+clu[:25])]:
            kmax=min(len(d[ax]) for d in grp); S=min(100, kmax//4)
            cs=[curve(d[ax][:kmax], d["ref"]) for d in grp]; mg=prep(cs)
            rs=np.random.default_rng(29); fit=[]; evl=[]
            for d in grp:
                c=d[ax][:kmax].copy(); rs.shuffle(c); h=len(c)//2
                fit.append(curve(c[:h],d["ref"],h,N_PERM,SEED_ORDERING+1))
                evl.append(curve(c[h:],d["ref"],h,N_PERM,SEED_ORDERING+2))
            mf=prep(fit); N=len(grp); ix=np.arange(N)
            di=d_in_of(mg,cs,S,ix); ds=d_split_of(mf,fit,evl,S,ix)
            ci=bootci(lambda q: d_in_of(mg,cs,S,q),N)
            cs_=bootci(lambda q: d_split_of(mf,fit,evl,S,q),N)
            fl=floor_p95(grp,ax,kmax,S)
            v=("진행" if (cs_[0]>=2.0 and di>fl) else "종료·신호부재" if ci[1]<fl else "유보")
            TAB[(s,ax,sn)]=dict(d_in=di,ci=ci,d_split=ds,ci_s=cs_,floor=fl,v=v)
            P(f"{s:>7} {ax:>5} {sn:>5} {di:>8.3f} [{ci[0]:>6.2f},{ci[1]:>6.2f}] "
              f"{ds:>9.3f} [{cs_[0]:>6.2f},{cs_[1]:>6.2f}] {fl:>8.3f}  {v}")

# ---------- 2)+3) 바닥 곡선과 스케일 붕괴 ----------
P(""); P("[2/3] 바닥 곡선 (비율 K/(S/N)=10 고정) 과 스케일 붕괴")
P(f"{'solver':>7} {'axis':>5} {'K':>6} {'S/N':>5} {'floor':>8} {'cv_exc':>8} {'Phi':>9} {'floor/Phi':>10}")
CURVE={}
for s in SOLVERS:
    inst=DATA[s]; uni=[d for d in inst if d["dist"]=="uniform"]; N=len(uni)
    amp=np.sqrt(2*np.log(N))
    for ax in "AB":
        KM=min(len(d[ax]) for d in uni)
        Ks=[k for k in [200,400,600,800,1000] if k<=KM]
        rows=[]
        for K in Ks:
            k=max(1,K//10)
            fl=floor_p95(uni,ax,K,k)
            cv=cv_excess(uni,ax,K,k)
            phi=cv*np.sqrt(k/K)*amp
            rows.append((K,k,fl,cv,phi,fl/phi if phi>0 else np.nan))
            P(f"{s:>7} {ax:>5} {K:>6} {k:>5} {fl:>8.3f} {cv:>8.4f} {phi:>9.4f} {fl/phi:>10.2f}")
        CURVE[(s,ax)]=rows

# ---------- 그림 ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for r, ax in enumerate("AB"):
    for s in SOLVERS:
        rows=CURVE[(s,ax)]
        K=[x[0] for x in rows]
        axes[r,0].plot(K,[x[2] for x in rows],"o-",label=s)
        axes[r,1].plot(K,[x[5] for x in rows],"o-",label=s)
    axes[r,0].set_title(f"axis {ax} — raw floor"); axes[r,0].set_ylabel("floor p95 (%)")
    axes[r,1].set_title(f"axis {ax} — floor / Phi (scaled)")
    for c in (0,1):
        axes[r,c].set_xscale("log"); axes[r,c].set_xlabel("K"); axes[r,c].grid(alpha=.25)
        axes[r,c].legend(fontsize=8)
plt.tight_layout(); plt.savefig("expand_floor_collapse.png", dpi=140, bbox_inches="tight")

# 붕괴 정도 정량화: 스케일 전/후 솔버 간 변동계수
P(""); P("스케일 전후 솔버 간 변동계수 (작을수록 겹침)")
for ax in "AB":
    Ks=[x[0] for x in CURVE[("POMO",ax)]]
    raw=[]; sc=[]
    for i,K in enumerate(Ks):
        r=[CURVE[(s,ax)][i][2] for s in SOLVERS]; q=[CURVE[(s,ax)][i][5] for s in SOLVERS]
        raw.append(np.std(r)/np.mean(r)); sc.append(np.std(q)/np.mean(q))
    P(f"  축 {ax}: raw CV={np.mean(raw):.3f}  ->  scaled CV={np.mean(sc):.3f}"
      f"   {'붕괴 성립' if np.mean(sc)<np.mean(raw)*0.6 else '붕괴 불충분'}")
open("expand_report.txt","w",encoding="utf-8").write("\n".join(REPORT))
print("\nsaved: expand_report.txt, expand_floor_collapse.png", flush=True)
