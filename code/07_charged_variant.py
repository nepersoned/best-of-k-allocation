# CHARGED 변형 (탐색적·기술적, 검정 없음 — 등록 종점은 uncharged 그대로)
#   고정 총예산 S = N * 100.  프로브 m=20/인스턴스를 예산에서 차감.
#   프로브 샘플도 해로 카운트한다(버리지 않음)  -> k_i >= m,  Σk_i = S
#   신호: 프로브 비용의 변동계수 (기준해 미사용 — 배포 시 LKH 없음)
#   평가: 같은 무작위 순서의 앞 k_i개 최솟값. 단일 패스, 분할 없음.
import json, warnings, numpy as np
warnings.filterwarnings("ignore")

M_PROBE, S_PER, N_ORDER, N_BOOT = 20, 100, 40, 300
SEED = 909

def run(costs, refs, is_clu, m=M_PROBE, s_per=S_PER, n_order=N_ORDER, seed=SEED):
    N = len(costs); S = N*s_per
    rng = np.random.default_rng(seed)
    d_alloc, d_label = [], []
    per_inst_u, per_inst_a = [], []
    for _ in range(n_order):
        orders = [rng.permutation(c) for c in costs]
        probe_cv = np.array([o[:m].std()/o[:m].mean() for o in orders])
        # --- CV 비례 배분 (프로브 차감, k_i >= m) ---
        rest = S - N*m
        w = probe_cv/probe_cv.sum()
        k = m + np.floor(rest*w).astype(int)
        # 잔여 분배
        i = 0; order_w = np.argsort(-w)
        while k.sum() < S:
            k[order_w[i % N]] += 1; i += 1
        k = np.minimum(k, np.array([len(o) for o in orders]))
        ku = np.full(N, S//N)
        g = lambda kk: np.array([100*(orders[j][:kk[j]].min()/refs[j]-1) for j in range(N)])
        gu, ga = g(ku), g(k)
        per_inst_u.append(gu); per_inst_a.append(ga)
        d_alloc.append(100*(gu.mean()-ga.mean())/gu.mean())
    U = np.mean(per_inst_u, 0); A = np.mean(per_inst_a, 0)
    # 인스턴스 부트스트랩
    rb = np.random.default_rng(SEED+1)
    bs = []
    for _ in range(N_BOOT):
        ix = rb.integers(0, N, N)
        bs.append(100*(U[ix].mean()-A[ix].mean())/U[ix].mean())
    return (float(np.mean(d_alloc)), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5)),
            float(U.mean()), float(A.mean()))

print("CHARGED 변형 — 탐색적/기술적, 검정 없음")
print(f"  총예산 S/N={S_PER}, 프로브 m={M_PROBE} (차감, 해로 카운트), 신호=프로브 CV, 기준해 미사용")
print(f"{'solver':>7} {'d_charged':>10} {'95% CI':>20} {'uniform gap':>12} {'alloc gap':>10}")
OUT={}
for s in ["AM","SYMNCO","POMO"]:
    z=np.load(f"confirm_{s}_axisA.npz",allow_pickle=True); meta=json.loads(str(z["meta"]))
    inst=[dict(m) for m in meta]
    for d,mm in zip(inst,meta): d["A"]=z[f"c{mm['id']}"]
    uni=[d for d in inst if d["dist"]=="uniform"]; clu=[d for d in inst if d["dist"]=="clustered"]
    grp=uni[:25]+clu[:25]
    costs=[d["A"] for d in grp]; refs=[d["ref"] for d in grp]
    is_clu=np.array([d["dist"]=="clustered" for d in grp])
    d,lo,hi,gu,ga = run(costs,refs,is_clu)
    OUT[s]=(d,lo,hi)
    print(f"{s:>7} {d:>10.3f} [{lo:>8.3f},{hi:>8.3f}] {gu:>12.3f} {ga:>10.3f}")

print("\n프로브 크기 민감도 (AM)")
z=np.load("confirm_AM_axisA.npz",allow_pickle=True); meta=json.loads(str(z["meta"]))
inst=[dict(m) for m in meta]
for d,mm in zip(inst,meta): d["A"]=z[f"c{mm['id']}"]
uni=[d for d in inst if d["dist"]=="uniform"]; clu=[d for d in inst if d["dist"]=="clustered"]
grp=uni[:25]+clu[:25]; costs=[d["A"] for d in grp]; refs=[d["ref"] for d in grp]
is_clu=np.array([d["dist"]=="clustered" for d in grp])
for m in [5,10,20,40]:
    d,lo,hi,_,_ = run(costs,refs,is_clu,m=m)
    print(f"   m={m:>3}  d_charged={d:7.3f}  CI [{lo:7.3f},{hi:7.3f}]  (예산의 {100*m/S_PER:.0f}%)")
json.dump(OUT, open("charged_variant.json","w"), indent=1)
print("\nsaved charged_variant.json")
