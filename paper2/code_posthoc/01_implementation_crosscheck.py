# 부호 반전 독립 검산: 두 가지 편상관 구현 대조 + 원시 관계 확인
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
def presid(a,*cov):                      # 구현 1: 순위 잔차화 (동결 코드와 동일)
    ra=rankdata(a); C=np.column_stack([rankdata(c) for c in cov]+[np.ones(len(ra))])
    b,*_=np.linalg.lstsq(C,ra,rcond=None); return ra-C@b
def partial_formula(x,y,z):              # 구현 2: 표준 1차 편상관 공식
    rxy=spearmanr(x,y)[0]; rxz=spearmanr(x,z)[0]; ryz=spearmanr(y,z)[0]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))
def dsplit(g):
    n=len(g); caps=np.array([len(d["fit"]) for d in g]); S=min(n*S_PER,int(caps.sum()))
    M=np.concatenate([np.stack([g[i]["mg"],np.full(len(g[i]["mg"]),i)],1) for i in range(n)])
    tk=M[np.argsort(-M[:,0])[:max(0,S-n)],1].astype(int)
    ko=np.minimum(np.ones(n,int)+np.bincount(tk,minlength=n),caps)
    ku=np.minimum(np.full(n,S//n),caps)
    b=np.mean([g[i]["ev"][ku[i]-1] for i in range(n)])
    return 100*(b-np.mean([g[i]["ev"][ko[i]-1] for i in range(n)]))/b
def sets_of(path):
    I=load(path); B=[d for d in I if d["sigma"]<0]; S=[d for d in I if d["sigma"]>0]
    b1=np.mean([100*(d["c"][X].mean()/d["ref"]-1) for d in B])
    for d in S:
        d["r"]=(100*(d["c"][X].mean()/d["ref"]-1))/b1
        d["fit"]=curve(d["c"][FIT],d["ref"],1); d["ev"]=curve(d["c"][EV],d["ref"],2)
        d["mg"]=-np.diff(cvx(d["fit"]))
    P=sorted(S,key=lambda d:d["r"]); n=len(P); rng=np.random.default_rng(5); out=[]
    for _ in range(NSAMP):
        lo=rng.integers(0,n-NSET); w=rng.integers(NSET,n-lo+1)
        idx=sorted(rng.choice(np.arange(lo,lo+w),NSET,replace=False))
        g=[P[i] for i in idx]
        out.append((float(np.std(np.log([d["r"] for d in g]))),
                    float(np.median([d["r"] for d in g])), dsplit(g)))
    return np.array(out)
print(f"{'arm':>4} {'solver':>7} {'raw het~d':>10} {'raw lev~d':>10} {'lev|het (잔차화)':>17} {'lev|het (공식)':>15} {'het~lev':>9}")
for arm,pref in [("A","rq1lv"),("B","rq1sz")]:
    for s in ["AM","SYMNCO","POMO"]:
        M=sets_of(f"{pref}_{s}.npz"); H,L,D=M[:,0],M[:,1],M[:,2]
        r1=spearmanr(presid(L,H),presid(D,H))[0]
        r2=partial_formula(L,D,H)
        print(f"{arm:>4} {s:>7} {spearmanr(H,D)[0]:>+10.3f} {spearmanr(L,D)[0]:>+10.3f} "
              f"{r1:>+17.3f} {r2:>+15.3f} {spearmanr(H,L)[0]:>+9.3f}", flush=True)
