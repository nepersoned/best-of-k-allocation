# Figure 2 재계산 — 폐기된 max 대신 median 정의로 바닥 작동 특성
#   (a) K 스윕  (S/N=100 고정, N=50)
#   (b) N 스윕  (K=800, S/N=100 고정)  <- 원천 수와 N 을 분리해 교란 제거
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED_ORDERING, SEED_BOOTSTRAP = 22, 33
FLOOR_REP, FLOOR_PERM, N_SRC = 5, 60, 20
S_PER = 100
def curve(c, ref, kmax=None, n_perm=FLOOR_PERM, seed=SEED_ORDERING):
    K=len(c); kmax=kmax or K
    rng=np.random.default_rng(seed); idx=rng.random((n_perm,K)).argsort(1)[:,:kmax]
    return 100.0*(np.minimum.accumulate(np.asarray(c)[idx],1).mean(0)/ref-1.0)
def cvx(f):
    n=len(f); xs=np.arange(n,dtype=float); h=[]
    for i in range(n):
        while len(h)>=2:
            (x1,y1),(x2,y2)=h[-2],h[-1]
            if (y2-y1)*(xs[i]-x1)>=(f[i]-y1)*(x2-x1): h.pop()
            else: break
        h.append((xs[i],f[i]))
    return np.interp(xs,np.array([p[0] for p in h]),np.array([p[1] for p in h]))
def d_in_null(cs, S_per):
    N=len(cs); mg=[(-np.diff(cvx(c))) for c in cs]
    caps=np.array([len(c) for c in cs]); S=min(N*S_per,int(caps.sum()))
    M=np.concatenate([np.stack([mg[i],np.full(len(mg[i]),i)],1) for i in range(N)])
    take=M[np.argsort(-M[:,0])[:max(0,S-N)],1].astype(int)
    ko=np.minimum(np.ones(N,int)+np.bincount(take,minlength=N),caps)
    ku=np.minimum(np.full(N,S//N),caps)
    b=float(np.mean([cs[i][ku[i]-1] for i in range(N)]))
    return 100*(b-float(np.mean([cs[i][ko[i]-1] for i in range(N)])))/b

def floor_median(pool, ax, kmax, N, S_per, seed=101):
    """참 d=0 귀무: 원천 인스턴스 배열에서 N개 가짜 생성. 원천별 p95의 median."""
    rng=np.random.default_rng(seed)
    srcs=[pool[i] for i in rng.choice(len(pool), min(N_SRC,len(pool)), replace=False)]
    p95=[]
    for d in srcs:
        z=d[ax][:kmax]/d["ref"]; ds=[]
        for _ in range(FLOOR_REP):
            cs=[curve(rng.choice(z,kmax,True),1.0) for _ in range(N)]
            ds.append(d_in_null(cs,S_per))
        p95.append(np.percentile(ds,95))
    p95=np.array(p95)
    return float(np.median(p95)), float(np.percentile(p95,90))

DATA={}
for s in ["POMO","AM","SYMNCO"]:
    z=np.load(f"expand_{s}_axisB.npz",allow_pickle=True); meta=json.loads(str(z["meta"]))
    inst=[dict(m) for m in meta]
    for d,m in zip(inst,meta): d["B"]=z[f"c{m['id']}"]
    DATA[s]=[d for d in inst if d["dist"]=="uniform"]

print("=== (a) K 스윕  S/N=100, N=50  [axis B, 동질] ===", flush=True)
KS=[400,600,800]; A={}
for s in DATA:
    row=[floor_median(DATA[s],"B",K,50,S_PER) for K in KS]
    A[s]=row
    print(f"  {s:>7}: " + "  ".join(f"K={K} med={m:.3f} p90={p:.3f}" for K,(m,p) in zip(KS,row)), flush=True)

print("\n=== (b) N 스윕  K=800, S/N=100  [axis B, 동질] ===", flush=True)
NS=[10,20,30,50]; B={}
for s in DATA:
    row=[floor_median(DATA[s],"B",800,N,S_PER) for N in NS]
    B[s]=row
    print(f"  {s:>7}: " + "  ".join(f"N={N} med={m:.3f} p90={p:.3f}" for N,(m,p) in zip(NS,row)), flush=True)

fig,ax=plt.subplots(1,2,figsize=(11,4))
for s in DATA:
    ax[0].plot(KS,[r[0] for r in A[s]],"o-",label=s)
    ax[1].plot(NS,[r[0] for r in B[s]],"o-",label=s)
ax[0].set_xlabel("K (stored samples per instance)"); ax[0].set_title("(a) floor vs K   (S/N=100, N=50)")
ax[1].set_xlabel("N (instances)"); ax[1].set_title("(b) floor vs N   (K=800, S/N=100)")
for a in ax: a.set_ylabel("noise floor, median of per-source p95 (%)"); a.grid(alpha=.25); a.legend(fontsize=8)
# sqrt(2 log N) 참조선
import math
base=B["POMO"][0][0]/math.sqrt(2*math.log(10))
ax[1].plot(NS,[base*math.sqrt(2*math.log(n)) for n in NS],"k:",lw=1,label="sqrt(2 log N) ref")
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig2_floor_median.png",dpi=140,bbox_inches="tight")
json.dump({"K":KS,"N":NS,"a":{k:[list(x) for x in v] for k,v in A.items()},
           "b":{k:[list(x) for x in v] for k,v in B.items()}},
          open("fig2_floor_median.json","w"), indent=1)
print("\nsaved fig2_floor_median.{png,json}", flush=True)
