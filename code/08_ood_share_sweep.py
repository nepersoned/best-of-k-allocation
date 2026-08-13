# Figure 4 (탐색적/기술적, 검정 없음) — OOD 비중 vs 배분 이득
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SEED_ORDERING, SEED_SPLIT = 22, 29
N_PERM, N_SET, M_PROBE, S_PER, N_ORDER = 200, 12, 20, 100, 20
def curve(c, ref, kmax=None, n_perm=N_PERM, seed=SEED_ORDERING):
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
def d_split(grp, s_per=S_PER):
    rs=np.random.default_rng(SEED_SPLIT); fit=[]; evl=[]
    kmax=min(len(d["A"]) for d in grp)
    for d in grp:
        c=d["A"][:kmax].copy(); rs.shuffle(c); h=len(c)//2
        fit.append(curve(c[:h],d["ref"],h)); evl.append(curve(c[h:],d["ref"],h))
    N=len(grp); mg=[(-np.diff(cvx(c))) for c in fit]
    caps=np.array([len(c) for c in fit]); S=min(N*s_per,int(caps.sum()))
    M=np.concatenate([np.stack([mg[i],np.full(len(mg[i]),i)],1) for i in range(N)])
    take=M[np.argsort(-M[:,0])[:max(0,S-N)],1].astype(int)
    ko=np.minimum(np.ones(N,int)+np.bincount(take,minlength=N),caps)
    ku=np.minimum(np.full(N,S//N),caps)
    b=float(np.mean([evl[i][ku[i]-1] for i in range(N)]))
    return 100*(b-float(np.mean([evl[i][ko[i]-1] for i in range(N)])))/b
def d_charged(grp, m=M_PROBE, s_per=S_PER, seed=909):
    N=len(grp); S=N*s_per; rng=np.random.default_rng(seed); out=[]
    for _ in range(N_ORDER):
        orders=[rng.permutation(d["A"]) for d in grp]
        cv=np.array([o[:m].std()/o[:m].mean() for o in orders]); w=cv/cv.sum()
        k=m+np.floor((S-N*m)*w).astype(int)
        i=0; ow=np.argsort(-w)
        while k.sum()<S: k[ow[i%N]]+=1; i+=1
        k=np.minimum(k,np.array([len(o) for o in orders]))
        ku=np.full(N,S//N)
        g=lambda kk: np.mean([100*(orders[j][:kk[j]].min()/grp[j]["ref"]-1) for j in range(N)])
        gu,ga=g(ku),g(k); out.append(100*(gu-ga)/gu)
    return float(np.mean(out))

z=np.load("confirm_AM_axisA.npz",allow_pickle=True); meta=json.loads(str(z["meta"]))
inst=[dict(m) for m in meta]
for d,m in zip(inst,meta): d["A"]=z[f"c{m['id']}"]
UNI=[d for d in inst if d["dist"]=="uniform"]; CLU=[d for d in inst if d["dist"]=="clustered"]
N=50; shares=[0.0,0.10,0.25,0.50]
print("Figure 4 (exploratory) — AM, axis A, N=50, S/N=100")
print(f"{'OOD share':>10} {'d_split':>9} {'d_charged':>10}")
rows=[]
for sh in shares:
    nc=int(round(sh*N)); nu=N-nc
    ds=[]; dc=[]
    for r in range(N_SET):
        rg=np.random.default_rng(500+r)
        grp=[UNI[i] for i in rg.choice(len(UNI),min(nu,len(UNI)),replace=False)] + \
            ([CLU[i] for i in rg.choice(len(CLU),nc,replace=False)] if nc else [])
        if len(grp)<N: grp=grp+[UNI[i] for i in rg.choice(len(UNI),N-len(grp),replace=True)]
        ds.append(d_split(grp)); dc.append(d_charged(grp))
    rows.append((sh,float(np.mean(ds)),float(np.mean(dc))))
    print(f"{sh*100:>9.0f}% {np.mean(ds):>9.3f} {np.mean(dc):>10.3f}")
fig,ax=plt.subplots(figsize=(6,4))
ax.plot([r[0]*100 for r in rows],[r[1] for r in rows],"o-",label="d_split (signal free)")
ax.plot([r[0]*100 for r in rows],[r[2] for r in rows],"s--",label="d_charged (probe m=20 charged)")
ax.axhline(0,color="k",lw=.8); ax.set_xlabel("out-of-distribution share of workload (%)")
ax.set_ylabel("allocation gain (%)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
ax.set_title("Fig 4 (exploratory, descriptive — no test)")
plt.tight_layout(); plt.savefig("fig4_ood_share.png",dpi=140,bbox_inches="tight")
json.dump([{"share":r[0],"d_split":r[1],"d_charged":r[2]} for r in rows],open("fig4_ood_share.json","w"),indent=1)
print("saved fig4_ood_share.{png,json}")
