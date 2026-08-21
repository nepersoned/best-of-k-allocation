# 가설(사후·탐색): level 항은 '심각도'가 아니라 '인스턴스 규모'를 추적한다.
#   팔 A: 짧은 인스턴스 -> 높은 심각도 (LKH~severity 음)
#   팔 B: 긴 인스턴스   -> 높은 심각도 (LKH~severity 양)
#   두 팔에서 severity 의 부호가 뒤집혔다면, LKH 길이로 바꿔 재면 부호가 같아야 한다.
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata
X=slice(0,200); FIT=slice(200,600); EV=slice(600,1000)
S_PER,N_PERM,NSET,NSAMP=100,150,40,200
def load(p):
    z=np.load(p,allow_pickle=True); m=json.loads(str(z["meta"]))
    I=[dict(d) for d in m]
    for d,q in zip(I,m): d["c"]=z[f"c{q['id']}"]
    return I
def curve(c,ref,seed):
    K=len(c); rng=np.random.default_rng(seed); idx=rng.random((N_PERM,K)).argsort(1)
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
def presid(a,*cov):
    ra=rankdata(a); C=np.column_stack([rankdata(c) for c in cov]+[np.ones(len(ra))])
    b,*_=np.linalg.lstsq(C,ra,rcond=None); return ra-C@b
def dsplit(g):
    n=len(g); caps=np.array([len(d["fit"]) for d in g]); S=min(n*S_PER,int(caps.sum()))
    M=np.concatenate([np.stack([g[i]["mg"],np.full(len(g[i]["mg"]),i)],1) for i in range(n)])
    tk=M[np.argsort(-M[:,0])[:max(0,S-n)],1].astype(int)
    ko=np.minimum(np.ones(n,int)+np.bincount(tk,minlength=n),caps)
    ku=np.minimum(np.full(n,S//n),caps)
    b=np.mean([g[i]["ev"][ku[i]-1] for i in range(n)])
    return 100*(b-np.mean([g[i]["ev"][ko[i]-1] for i in range(n)]))/b
print(f"{'arm':>4} {'solver':>7} {'LKH~sev':>9} | {'sev|het':>9} {'LKH|het':>9} | {'해석'}")
for arm,pref in [("A","rq1lv"),("B","rq1sz")]:
    for s in ["AM","SYMNCO"]:
        I=load(f"{pref}_{s}.npz"); B=[d for d in I if d["sigma"]<0]; S=[d for d in I if d["sigma"]>0]
        b1=np.mean([100*(d["c"][X].mean()/d["ref"]-1) for d in B])
        for d in S:
            d["r"]=(100*(d["c"][X].mean()/d["ref"]-1))/b1
            d["fit"]=curve(d["c"][FIT],d["ref"],1); d["ev"]=curve(d["c"][EV],d["ref"],2)
            d["mg"]=-np.diff(cvx(d["fit"]))
        lkh_sev=spearmanr([d["ref"] for d in S],[d["r"] for d in S])[0]
        P=sorted(S,key=lambda d:d["r"]); n=len(P); rng=np.random.default_rng(5); rows=[]
        for _ in range(NSAMP):
            lo=rng.integers(0,n-NSET); w=rng.integers(NSET,n-lo+1)
            idx=sorted(rng.choice(np.arange(lo,lo+w),NSET,replace=False))
            g=[P[i] for i in idx]
            rows.append((float(np.std(np.log([d["r"] for d in g]))),
                         float(np.median([d["r"] for d in g])),
                         float(np.median([d["ref"] for d in g])), dsplit(g)))
        M=np.array(rows); H,L,R,D=M[:,0],M[:,1],M[:,2],M[:,3]
        sv=spearmanr(presid(L,H),presid(D,H))[0]
        lk=spearmanr(presid(R,H),presid(D,H))[0]
        note = "길수록 이득↓" if lk<0 else "길수록 이득↑"
        print(f"{arm:>4} {s:>7} {lkh_sev:>+9.3f} | {sv:>+9.3f} {lk:>+9.3f} | {note}", flush=True)
