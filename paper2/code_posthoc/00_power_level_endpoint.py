# AM level|spread 종점의 복제 확률 추정 (이중 부트스트랩)
#   외부: 인스턴스 풀 재표집 -> 새 표본 흉내
#   내부: 등록된 CI 절차 그대로 -> 그 표본에서 CI 하한 > 0 인가
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata
X=slice(0,200); FIT=slice(200,600); EV=slice(600,1000)
S_PER,N_PERM,NSET,NSAMP = 100,150,40,200
OUTER, INNER = 40, 150
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
def presid(a,b):
    ra,rb=rankdata(a),rankdata(b); return ra-np.polyval(np.polyfit(rb,ra,1),rb)
z=np.load("rq1conf_AM.npz",allow_pickle=True); m=json.loads(str(z["meta"]))
I=[dict(d) for d in m]
for d,q in zip(I,m): d["c"]=z[f"c{q['id']}"]
B=[d for d in I if d["sigma"]<0]; S=[d for d in I if d["sigma"]>0]
b1=np.mean([100*(d["c"][X].mean()/d["ref"]-1) for d in B])
for d in S:
    d["r"]=(100*(d["c"][X].mean()/d["ref"]-1))/b1
    d["fit"]=curve(d["c"][FIT],d["ref"],1); d["ev"]=curve(d["c"][EV],d["ref"],2)
    d["mg"]=-np.diff(cvx(d["fit"]))
def dsplit(g):
    n=len(g); caps=np.array([len(d["fit"]) for d in g]); Sb=min(n*S_PER,int(caps.sum()))
    M=np.concatenate([np.stack([g[i]["mg"],np.full(len(g[i]["mg"]),i)],1) for i in range(n)])
    take=M[np.argsort(-M[:,0])[:max(0,Sb-n)],1].astype(int)
    ko=np.minimum(np.ones(n,int)+np.bincount(take,minlength=n),caps)
    ku=np.minimum(np.full(n,Sb//n),caps)
    b=np.mean([g[i]["ev"][ku[i]-1] for i in range(n)])
    return 100*(b-np.mean([g[i]["ev"][ko[i]-1] for i in range(n)]))/b
def endpoint(pool, seed):
    P=sorted(pool,key=lambda d:d["r"]); n=len(P); rng=np.random.default_rng(seed); sets=[]
    for _ in range(NSAMP):
        lo=rng.integers(0,n-NSET); w=rng.integers(NSET,n-lo+1)
        idx=sorted(rng.choice(np.arange(lo,lo+w),NSET,replace=False))
        g=[P[i] for i in idx]
        sets.append((np.array(idx), float(np.std(np.log([d["r"] for d in g]))),
                     float(np.median([d["r"] for d in g])), dsplit(g)))
    H=np.array([s[1] for s in sets]); LV=np.array([s[2] for s in sets]); D=np.array([s[3] for s in sets])
    rb=np.random.default_rng(seed+1); vs=[];vl=[]
    for _ in range(INNER):
        keep=set(rb.choice(n,n,replace=True).tolist())
        sub=[(h,l,d) for ix,h,l,d in sets if len(set(ix.tolist())&keep)>=NSET*0.6]
        if len(sub)<30: continue
        hh=np.array([x[0] for x in sub]); ll=np.array([x[1] for x in sub]); dd=np.array([x[2] for x in sub])
        vs.append(spearmanr(presid(hh,ll),presid(dd,ll))[0])
        vl.append(spearmanr(presid(ll,hh),presid(dd,hh))[0])
    return (np.percentile(vs,2.5), np.percentile(vl,2.5),
            spearmanr(presid(H,LV),presid(D,LV))[0], spearmanr(presid(LV,H),presid(D,H))[0])
rng=np.random.default_rng(77); okS=0; okL=0; both=0; rows=[]
for t in range(OUTER):
    pool=[S[i] for i in rng.integers(0,len(S),len(S))]
    ls,ll,ps,pl=endpoint(pool, 1000+t)
    okS+= ls>0; okL+= ll>0; both+= (ls>0 and ll>0); rows.append((ps,pl,ls,ll))
    if t<5: print(f"  rep{t+1}: spread {ps:+.3f} (하한 {ls:+.3f})  level {pl:+.3f} (하한 {ll:+.3f})", flush=True)
print(f"\n외부 {OUTER}회 (인스턴스 풀 재표집, 내부 CI {INNER}회)")
print(f"  spread|level 하한>0 : {okS}/{OUTER} = {100*okS/OUTER:.0f}%")
print(f"  level|spread 하한>0 : {okL}/{OUTER} = {100*okL/OUTER:.0f}%   <- 한계선 여부")
print(f"  둘 다 통과          : {both}/{OUTER} = {100*both/OUTER:.0f}%")
pl=np.array([r[1] for r in rows]); print(f"  level 점추정 분포: 중앙 {np.median(pl):+.3f}  범위 [{pl.min():+.3f}, {pl.max():+.3f}]")
