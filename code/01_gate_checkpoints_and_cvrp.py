# =====================================================================
# 확장 게이트 — 수집 전에 통과/탈락만 가른다 (GPU 5~10분)
#   G1  AM / SymNCO 체크포인트가 현행 rl4co에 실리는가 (키 완전일치 + gap 타당)
#   G2  RouteFinder pomo-cvrp.ckpt 가 실리고 도는가
#   G3  CVRP 기준해(PyVRP)의 시드 간 산포 <= 0.05%   <-- 핵심 통과 조건
#       미달이면 max_runtime 상향 재시도, 그래도 미달이면 CVRP 탈락
# 탈락한 항목만 빼고 나머지로 수집을 진행한다.
# =====================================================================
import re, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
from huggingface_hub import hf_hub_download

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPREAD_LIMIT = 0.05          # % — 재려는 효과(0.5%)보다 한 자릿수 아래
VERDICT = {}
def R(tag, ok, msg):
    VERDICT[tag] = ok
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {msg}", flush=True)

# ---------------------------------------------------------------- G1
print("="*70, flush=True); print("G1  TSP 솔버 확장 — AM / SymNCO", flush=True); print("="*70, flush=True)
from rl4co.envs import TSPEnv
from rl4co.models import POMO, AttentionModel, SymNCO
import elkai

def remap(sd):
    o = {}
    for k, v in sd.items():
        k2 = k[len("policy."):] if k.startswith("policy.") else k
        k2 = re.sub(r"\.module\.0\.(weight|bias)$", r".module.lins.0.\1", k2)
        k2 = re.sub(r"\.module\.2\.(weight|bias)$", r".module.lins.1.\1", k2)
        k2 = k2.replace("decoder.logit_attention.", "decoder.pointer.")
        o[k2] = v
    return o

def tlen(c, t):
    q = c[np.asarray(t)]; return float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum())

rng = np.random.default_rng(0)
COORDS = rng.random((8, 100, 2)).astype(np.float32)
REFS = []
for c in COORDS:
    D = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    t = elkai.DistanceMatrix(np.rint(D*1e6).astype(np.int64).tolist()).solve_tsp()
    if t[-1] == t[0]: t = t[:-1]
    REFS.append(tlen(c, t))
REFS = np.array(REFS)

from tensordict import TensorDict
env100 = TSPEnv(generator_params=dict(num_loc=100))

def try_tsp(tag, repo, fname, cls):
    try:
        sd = remap(torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=False))
        nl = 1 + max(int(m.group(1)) for k in sd
                     if (m := re.search(r"encoder\.net\.layers\.(\d+)\.", k)))
        has_bn = any("running_mean" in k for k in sd)
        norm = "batch" if has_bn else "instance"
        model = cls(env100, policy_kwargs=dict(normalization=norm, num_encoder_layers=nl))
        res = model.policy.load_state_dict(sd, strict=False)
        miss = [k for k in res.missing_keys if "num_batches" not in k]
        unex = [k for k in res.unexpected_keys if "num_batches" not in k]
        if miss or unex:
            R(tag, False, f"키 불일치 layers={nl} norm={norm} missing={miss[:3]} unexpected={unex[:3]}")
            return None
        pol = model.policy.to(DEV).eval()
        td = env100.reset(TensorDict({"locs": torch.as_tensor(COORDS, device=DEV)},
                                     batch_size=[8]).to(DEV))
        with torch.no_grad():
            out = pol(td, env100, phase="test", decode_type="multistart_greedy",
                      num_starts=100, return_actions=False)
        rw = out["reward"]
        cost = (-rw).view(-1, 8).min(0).values.cpu().numpy() if rw.numel() == 800 else (-rw).cpu().numpy()
        gap = 100*(cost/REFS - 1)
        ok = gap.mean() < 10
        R(tag, ok, f"layers={nl} norm={norm} 키완전일치, multistart-greedy gap {gap.mean():.2f}%")
        return gap.mean() if ok else None
    except Exception as e:
        R(tag, False, f"{type(e).__name__}: {e}")
        return None

try_tsp("AM/tsp100",     "ML4TSPBench/AM",     "tsp100_am.pt",     AttentionModel)
try_tsp("SYMNCO/tsp100", "ML4TSPBench/SYMNCO", "tsp100_symnco.pt", SymNCO)

# ---------------------------------------------------------------- G2
print("", flush=True); print("="*70, flush=True); print("G2  CVRP 체크포인트 — RouteFinder pomo-cvrp", flush=True); print("="*70, flush=True)
CVRP_OK = False
try:
    p = hf_hub_download("ai4co/routefinder", "checkpoints/100/pomo/pomo-cvrp.ckpt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("  ckpt top keys:", [k for k in ck.keys()][:8], flush=True)
    hp = ck.get("hyper_parameters", {})
    print("  hparams:", {k: hp[k] for k in list(hp)[:10]}, flush=True)
    from rl4co.envs import CVRPEnv
    cenv = CVRPEnv(generator_params=dict(num_loc=100))
    m = POMO.load_from_checkpoint(p, map_location=DEV, strict=False, env=cenv)
    pol = m.policy.to(DEV).eval()
    td = cenv.reset(batch_size=[4]).to(DEV)
    with torch.no_grad():
        o = pol(td, cenv, phase="test", decode_type="multistart_greedy",
                num_starts=100, return_actions=False)
    c = (-o["reward"]).view(-1, 4).min(0).values.cpu().numpy()
    R("CVRP/ckpt", True, f"로드·실행 성공, best-of-multistart cost {np.round(c,3)}")
    CVRP_OK = True
except Exception as e:
    R("CVRP/ckpt", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- G3
print("", flush=True); print("="*70, flush=True)
print(f"G3  CVRP 기준해 안정성 — 시드 3개 산포 <= {SPREAD_LIMIT}%  (핵심 통과 조건)", flush=True)
print("="*70, flush=True)
try:
    import pyvrp
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    SCALE, NL, CAP = 100_000, 100, 50
    rg = np.random.default_rng(7)
    insts = []
    for _ in range(10):
        coords = rg.random((NL+1, 2))                 # 0 = depot
        dem = rg.integers(1, 10, NL)
        insts.append((coords, dem))

    def solve(coords, dem, seed, secs):
        mdl = Model()
        d = mdl.add_depot(x=int(coords[0,0]*SCALE), y=int(coords[0,1]*SCALE))
        cl = [mdl.add_client(x=int(coords[i+1,0]*SCALE), y=int(coords[i+1,1]*SCALE),
                             delivery=int(dem[i])) for i in range(NL)]
        mdl.add_vehicle_type(NL, capacity=CAP, start_depot=d, end_depot=d)
        locs = [d] + cl
        for a in locs:
            for b in locs:
                if a is not b:
                    dist = int(round(np.hypot(a.x-b.x, a.y-b.y)))
                    mdl.add_edge(a, b, distance=dist)
        r = mdl.solve(stop=MaxRuntime(secs), seed=seed, display=False)
        return r.cost()

    for secs in [5, 15]:
        spreads = []
        t0 = time.time()
        for coords, dem in insts:
            v = np.array([solve(coords, dem, s, secs) for s in (1, 2, 3)], float)
            spreads.append(100*(v.max()-v.min())/v.mean())
        spreads = np.array(spreads)
        print(f"  max_runtime={secs:>2}s  시드간 산포: 평균 {spreads.mean():.4f}%  "
              f"최대 {spreads.max():.4f}%  ({time.time()-t0:.0f}s)", flush=True)
        if spreads.max() <= SPREAD_LIMIT:
            R("CVRP/ref", True, f"max_runtime={secs}s 에서 최대 산포 {spreads.max():.4f}% <= {SPREAD_LIMIT}%")
            break
    else:
        R("CVRP/ref", False, f"산포 {spreads.max():.4f}% > {SPREAD_LIMIT}% — CVRP 탈락 권고")
except Exception as e:
    R("CVRP/ref", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 정리
print("", flush=True); print("="*70, flush=True); print("게이트 결과", flush=True); print("="*70, flush=True)
for k, v in VERDICT.items():
    print(f"  {'PASS' if v else 'FAIL':<5} {k}", flush=True)
solvers = ["POMO"] + [t.split("/")[0] for t in ("AM/tsp100","SYMNCO/tsp100") if VERDICT.get(t)]
cvrp = VERDICT.get("CVRP/ckpt") and VERDICT.get("CVRP/ref")
print(f"\n수집 대상 확정: TSP 솔버 {solvers}"
      f"   CVRP {'포함' if cvrp else '제외'}", flush=True)
print(f"조건 수 = {len(solvers)} (TSP) + {1 if cvrp else 0} (CVRP) "
      f"= {len(solvers) + (1 if cvrp else 0)}", flush=True)
