# =====================================================================
# Best-of-k Allocation — 규모 증대 1회 (v6 §7 등록된 유일한 escalation)
#   N=100 (uniform 50 + clustered 50), K=1000, S/N=100 (§5.2 등록값)
#   산출 1: §7 게이트 판정 (축 A/B × 동질/혼합)
#   산출 2: 잡음 바닥 곡선 — 저장 배열을 앞에서 잘라 K를 바꾼다 (GPU 추가 0)
#           패널 a: S/N=100 고정, K 증가       -> 등록 예산에서 바닥이 내려가는가
#           패널 b: 비율 K/(S/N)=10 고정        -> 바닥이 K에 의존하나 비율에 의존하나
#   산출 3: 라벨 배분 회수율과 그 잔차 (자명성 차감 후 주장 수준)
# =====================================================================
import os, re, json, time, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensordict import TensorDict

# CONFIRM RUN: instance/decoding seeds replaced, all else frozen
SEED_INSTANCE, SEED_SAMPLING, SEED_ORDERING, SEED_BOOTSTRAP = 20260813, 77, 22, 33
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_UNIFORM, N_CLUSTERED, NLOC = 50, 50, 100
K_FULL, S_PER = 1000, 100                      # §5.2 등록 사양
N_PERM, N_BOOT = 200, 300
FLOOR_SRC, FLOOR_REP, FLOOR_PERM = 12, 8, 60   # 귀무 비용 통제 (원천 인스턴스 표본추출)
OUT = "/content/confirm"; os.makedirs(OUT, exist_ok=True)
assert S_PER * 4 <= K_FULL, "예산-깊이 제약 위반"
def P(s=""):
    print(s, flush=True); REPORT.append(s)
REPORT = []

# ---------------- 1. 인스턴스 ----------------
rng_i = np.random.default_rng(SEED_INSTANCE)
def gen_u(n, r): return r.random((n, 2)).astype(np.float32)
def gen_c(n, r, nc=4, sp=0.06):
    ctr = r.random((nc, 2)); a = r.integers(0, nc, n)
    return np.clip(ctr[a] + r.normal(0, sp, (n, 2)), 0, 1).astype(np.float32)
inst = []
for _ in range(N_UNIFORM):   inst.append(dict(id=len(inst), dist="uniform",   coords=gen_u(NLOC, rng_i)))
for _ in range(N_CLUSTERED): inst.append(dict(id=len(inst), dist="clustered", coords=gen_c(NLOC, rng_i)))
print(f"device={DEVICE} instances={len(inst)} n={NLOC} K={K_FULL} S/N={S_PER}", flush=True)

# ---------------- 2. LKH ----------------
import elkai
def tlen(c, t):
    q = c[np.asarray(t)]; return float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum())
t0 = time.time()
for d in inst:
    D = np.linalg.norm(d["coords"][:, None] - d["coords"][None, :], axis=-1)
    t = elkai.DistanceMatrix(np.rint(D * 1e6).astype(np.int64).tolist()).solve_tsp()
    if t[-1] == t[0]: t = t[:-1]
    d["ref"] = tlen(d["coords"], t)
print(f"LKH {time.time()-t0:.0f}s", flush=True)

# ---------------- 3. 솔버별 수집 (인스턴스·기준해 공유) ----------------
SOLVERS = [("POMO",   "ML4TSPBench/POMO",   "tsp100_pomo.pt"),
           ("AM",     "ML4TSPBench/AM",     "tsp100_am.pt"),
           ("SYMNCO", "ML4TSPBench/SYMNCO", "tsp100_symnco.pt")]
# ---------------- 체크포인트 ----------------
from huggingface_hub import hf_hub_download
from rl4co.envs import TSPEnv
from rl4co.models import POMO
def remap(sd):
    o = {}
    for k, v in sd.items():
        if k.startswith("baseline."):   # rollout baseline 사본 (AM) — 정책이 아니다
            continue
        k2 = k[len("policy."):] if k.startswith("policy.") else k
        k2 = re.sub(r"\.module\.0\.(weight|bias)$", r".module.lins.0.\1", k2)
        k2 = re.sub(r"\.module\.2\.(weight|bias)$", r".module.lins.1.\1", k2)
        o[k2.replace("decoder.logit_attention.", "decoder.pointer.")] = v
    return o
env = TSPEnv(generator_params=dict(num_loc=NLOC))
from rl4co.models import AttentionModel, SymNCO
CLS = {"POMO": POMO, "AM": AttentionModel, "SYMNCO": SymNCO}

def load_policy(name, repo, fname):
    sd = remap(torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=False))
    nl = 1 + max(int(m.group(1)) for k in sd if (m := re.search(r"encoder\.net\.layers\.(\d+)\.", k)))
    norm = "batch" if any("running_mean" in k for k in sd) else "instance"
    mdl = CLS[name](env, policy_kwargs=dict(normalization=norm, num_encoder_layers=nl))
    rr = mdl.policy.load_state_dict(sd, strict=False)
    assert not [k for k in rr.missing_keys if "num_batches" not in k] and            not [k for k in rr.unexpected_keys if "num_batches" not in k], f"{name} 키 불일치"
    print(f"  {name}: layers={nl} norm={norm} 키완전일치", flush=True)
    return mdl.policy.to(DEVICE).eval()

# ---------------- 4. 수집 ----------------
def td_of(c, b=1):
    x = torch.as_tensor(c, dtype=torch.float32, device=DEVICE)
    return env.reset(TensorDict({"locs": x.unsqueeze(0).expand(b, *x.shape).clone()},
                                batch_size=[b]).to(DEVICE))
@torch.no_grad()
def samp(c, k, chunk=250):
    o, done = [], 0
    while done < k:
        b = min(chunk, k - done)
        o.append((-policy(td_of(c, b), env, phase="test", decode_type="sampling",
                          num_starts=0, return_actions=False)["reward"]).cpu().numpy().ravel())
        done += b
    return np.concatenate(o)[:k].astype(np.float64)
def dih8(c):
    x, y = c[:, 0], c[:, 1]
    return [np.stack(v, 1).astype(np.float32) for v in
            [(x,y),(1-x,y),(x,1-y),(1-x,1-y),(y,x),(1-y,x),(y,1-x),(1-y,1-x)]]
@torch.no_grad()
def msg(c):
    return (-policy(td_of(c, 1), env, phase="test", decode_type="multistart_greedy",
                    num_starts=len(c), return_actions=False)["reward"]).cpu().numpy().ravel().astype(np.float64)
for SNAME, REPO, FNAME in SOLVERS:
    print(f"=== {SNAME} ===", flush=True)
    policy = load_policy(SNAME, REPO, FNAME)
    torch.manual_seed(SEED_SAMPLING); t0 = time.time()
    for j, d in enumerate(inst):
        d["A"] = samp(d["coords"], K_FULL)
        d["B"] = np.concatenate([msg(v) for v in dih8(d["coords"])])
        if (j+1) % 50 == 0: print(f"  {SNAME} {j+1}/{len(inst)} {time.time()-t0:.0f}s", flush=True)
    print(f"  {SNAME} 수집 {time.time()-t0:.0f}s", flush=True)
    for ax in "AB":
        p = f"{OUT}/{SNAME}_axis{ax}.npz"
        np.savez_compressed(p, **{f"c{d['id']}": d[ax] for d in inst},
                            meta=json.dumps([{k: d[k] for k in ("id","dist","ref")} for d in inst]))
        print(f"    {p} sha={hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}", flush=True)

print("COLLECT_DONE", flush=True)
