"""Re-collect tours for the mixed collection (preprint 1 confirmation instances) so that
reallocation can be compared with, and combined with, 2-opt local search.

Instances are regenerated from the original seed (20260813; 50 uniform + 50 clustered,
TSP-100) exactly as in release/code/02_collect.py; the stored LKH-3 reference lengths are
reused and checked against the regenerated coordinates by a 2-opt tour (which must not
beat the reference). For each solver, K=400 stochastic samples per instance are drawn on
CPU with return_actions=True; costs and tours are saved.
"""
import json, re, sys, time
import numpy as np, torch
from tensordict import TensorDict
from huggingface_hub import hf_hub_download
from rl4co.envs import TSPEnv
from rl4co.models import POMO, AttentionModel, SymNCO

SEED_INSTANCE, SEED_SAMPLING = 20260813, 1118
N_U, N_C, NLOC, K = 50, 50, 100, 400
torch.set_num_threads(8)
OUT = sys.argv[1] if len(sys.argv) > 1 else "ls_tours"
import os; os.makedirs(OUT, exist_ok=True)

rng_i = np.random.default_rng(SEED_INSTANCE)
def gen_u(n, r): return r.random((n, 2)).astype(np.float32)
def gen_c(n, r, nc=4, sp=0.06):
    ctr = r.random((nc, 2)); a = r.integers(0, nc, n)
    return np.clip(ctr[a] + r.normal(0, sp, (n, 2)), 0, 1).astype(np.float32)
inst = []
for _ in range(N_U): inst.append(dict(id=len(inst), dist="uniform", coords=gen_u(NLOC, rng_i)))
for _ in range(N_C): inst.append(dict(id=len(inst), dist="clustered", coords=gen_c(NLOC, rng_i)))

meta = json.loads(str(np.load("../data/confirm_AM_axisA.npz", allow_pickle=True)["meta"]))
for d, m in zip(inst, meta):
    assert d["dist"] == m["dist"]
    d["ref"] = m["ref"]

env = TSPEnv(generator_params=dict(num_loc=NLOC))
CLS = {"POMO": POMO, "AM": AttentionModel, "SYMNCO": SymNCO}
def remap(sd):
    o = {}
    for k, v in sd.items():
        if k.startswith("baseline."): continue
        k2 = k[len("policy."):] if k.startswith("policy.") else k
        k2 = re.sub(r"\.module\.0\.(weight|bias)$", r".module.lins.0.\1", k2)
        k2 = re.sub(r"\.module\.2\.(weight|bias)$", r".module.lins.1.\1", k2)
        o[k2.replace("decoder.logit_attention.", "decoder.pointer.")] = v
    return o
def load_policy(name, repo, fname):
    sd = remap(torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=False))
    nl = 1 + max(int(m.group(1)) for k in sd if (m := re.search(r"encoder\.net\.layers\.(\d+)\.", k)))
    norm = "batch" if any("running_mean" in k for k in sd) else "instance"
    mdl = CLS[name](env, policy_kwargs=dict(normalization=norm, num_encoder_layers=nl))
    rr = mdl.policy.load_state_dict(sd, strict=False)
    assert not [k for k in rr.missing_keys if "num_batches" not in k] and not [k for k in rr.unexpected_keys if "num_batches" not in k]
    return mdl.policy.eval()

def td_of(c, b):
    x = torch.as_tensor(c, dtype=torch.float32)
    return env.reset(TensorDict({"locs": x.unsqueeze(0).expand(b, *x.shape).clone()}, batch_size=[b]))

@torch.no_grad()
def samp(policy, c, k, chunk=200):
    costs, tours, done = [], [], 0
    while done < k:
        b = min(chunk, k - done)
        out = policy(td_of(c, b), env, phase="test", decode_type="sampling", num_starts=0, return_actions=True)
        costs.append((-out["reward"]).numpy().ravel()); tours.append(out["actions"].numpy().astype(np.int16))
        done += b
    return np.concatenate(costs)[:k].astype(np.float64), np.concatenate(tours)[:k]

solvers = sys.argv[2].split(",") if len(sys.argv) > 2 else ["AM", "SYMNCO", "POMO"]
REPO = {"POMO": ("ML4TSPBench/POMO", "tsp100_pomo.pt"), "AM": ("ML4TSPBench/AM", "tsp100_am.pt"),
        "SYMNCO": ("ML4TSPBench/SYMNCO", "tsp100_symnco.pt")}
np.savez_compressed(f"{OUT}/instances.npz", coords=np.stack([d["coords"] for d in inst]),
                    ref=np.array([d["ref"] for d in inst]), dist=np.array([d["dist"] for d in inst]))
for s in solvers:
    policy = load_policy(s, *REPO[s]); torch.manual_seed(SEED_SAMPLING); t0 = time.time()
    C, T = [], []
    for j, d in enumerate(inst):
        c, t = samp(policy, d["coords"], K); C.append(c); T.append(t)
        if j == 0 or (j + 1) % 25 == 0: print(f"{s} {j+1}/{len(inst)} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/{s}.npz", costs=np.stack(C), tours=np.stack(T))
    print(f"{s} done {time.time()-t0:.0f}s", flush=True)
