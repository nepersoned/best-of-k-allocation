"""논문 ② 그림 5장.  저장 배열 위 오프라인 계산, GPU 불필요.

집합 구성·d_split·순위 잔차화는 동결 코드(code/analysis_confirm2.py)와 동일한
절차·동일한 시드를 쓴다.  그림은 등록 종점을 다시 판정하지 않는다 — 이미 계산되어
문서에 기록된 값을 시각화할 뿐이다.

  fig1  spread 축 (팔 A, 3솔버)                     — §4
  fig2  level 항의 부호 반전과 길이 치환 (2x2)      — §5
  fig3  구성 rho 의 비단조성과 매개                 — §6
  fig4  정책 구간 막대                              — §7
  fig5  사전 명세된 사후 검정 2건 (forest)          — §5.4 / 한계 1

실행:  python make_figures.py [data_dir] [out_dir]
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
CACHE = os.path.join(OUT, "_figcache.npz")

X = slice(0, 200); FIT = slice(200, 600); EV = slice(600, 1000)
S_PER, N_PERM, NSET, NSAMP = 100, 150, 40, 200
RHO_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
RHO_NSET, RHO_REP = 30, 25

plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 300, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
                     "legend.frameon": False})
CLR = {"AM": "#1f4e79", "SYMNCO": "#c1440e", "POMO": "#6b6b6b"}
NAME = {"AM": "AM", "SYMNCO": "Sym-NCO", "POMO": "POMO"}


# ---------- 동결 코드와 동일한 계산 경로 ----------
def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    inst = [dict(d) for d in meta]
    for d, q in zip(inst, meta):
        d["c"] = z["c%d" % q["id"]]
    return inst


def curve(c, ref, seed):
    K = len(c)
    rng = np.random.default_rng(seed)
    idx = rng.random((N_PERM, K)).argsort(1)
    return 100.0 * (np.minimum.accumulate(np.asarray(c)[idx], 1).mean(0) / ref - 1.0)


def cvx(f):
    n = len(f); xs = np.arange(n, dtype=float); h = []
    for i in range(n):
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (y2 - y1) * (xs[i] - x1) >= (f[i] - y1) * (x2 - x1):
                h.pop()
            else:
                break
        h.append((xs[i], f[i]))
    return np.interp(xs, np.array([p[0] for p in h]), np.array([p[1] for p in h]))


def presid(a, *covs):
    ra = rankdata(a)
    Cm = np.column_stack([rankdata(c) for c in covs] + [np.ones(len(ra))])
    beta, *_ = np.linalg.lstsq(Cm, ra, rcond=None)
    return ra - Cm @ beta


def _alloc(g):
    n = len(g); caps = np.array([len(d["fit"]) for d in g])
    S = min(n * S_PER, int(caps.sum()))
    M = np.concatenate([np.stack([g[i]["mg"], np.full(len(g[i]["mg"]), i)], 1)
                        for i in range(n)])
    take = M[np.argsort(-M[:, 0])[:max(0, S - n)], 1].astype(int)
    ko = np.minimum(np.ones(n, int) + np.bincount(take, minlength=n), caps)
    ku = np.minimum(np.full(n, S // n), caps)
    return ko, ku


def dsplit(g):
    """등록된 gap 단위 d_split (%)."""
    ko, ku = _alloc(g); n = len(g)
    b = np.mean([g[i]["ev"][ku[i] - 1] for i in range(n)])
    return 100 * (b - np.mean([g[i]["ev"][ko[i] - 1] for i in range(n)])) / b


def dsplit_abs(g):
    """검정 U — 절대 초과(길이 단위).  gap%를 최적 길이로 되곱한다."""
    ko, ku = _alloc(g); n = len(g)
    au = np.mean([g[i]["ev"][ku[i] - 1] / 100.0 * g[i]["ref"] for i in range(n)])
    ao = np.mean([g[i]["ev"][ko[i] - 1] / 100.0 * g[i]["ref"] for i in range(n)])
    return au - ao


def prep(path):
    inst = load(path)
    base = [d for d in inst if d["sigma"] < 0]
    shift = [d for d in inst if d["sigma"] > 0]
    b1 = np.mean([100 * (d["c"][X].mean() / d["ref"] - 1) for d in base])
    for d in base + shift:
        d["r"] = (100 * (d["c"][X].mean() / d["ref"] - 1)) / b1
        d["fit"] = curve(d["c"][FIT], d["ref"], 1)
        d["ev"] = curve(d["c"][EV], d["ref"], 2)
        d["mg"] = -np.diff(cvx(d["fit"]))
    return base, shift


def window_sets(shift, seed=5):
    """동결 코드 build_sets 와 동일한 창 표집 + 길이/절대단위 열 추가."""
    P = sorted(shift, key=lambda d: d["r"]); n = len(P)
    rng = np.random.default_rng(seed); rows = []
    for _ in range(NSAMP):
        lo = rng.integers(0, n - NSET)
        w = rng.integers(NSET, n - lo + 1)
        idx = sorted(rng.choice(np.arange(lo, lo + w), NSET, replace=False))
        g = [P[i] for i in idx]
        rows.append((float(np.std(np.log([d["r"] for d in g]))),
                     float(np.median([d["r"] for d in g])),
                     float(np.median([d["ref"] for d in g])),
                     dsplit(g), dsplit_abs(g)))
    return np.array(rows)          # spread, level, medlen, d_gap, d_abs


def rho_sets(base, shift):
    rows = []
    for rho in RHO_LEVELS:
        for rep in range(RHO_REP):
            rng = np.random.default_rng(3000 + rep)
            nO = int(round(rho * RHO_NSET)); nU = RHO_NSET - nO
            if nU > len(base):
                continue
            g = ([base[i] for i in rng.choice(len(base), nU, replace=False)]
                 + [shift[i] for i in rng.choice(len(shift), nO, replace=False)])
            rows.append((rho,
                         float(np.std(np.log([d["r"] for d in g]))),
                         float(np.median([d["r"] for d in g])),
                         dsplit(g)))
    return np.array(rows)          # rho, spread, level, d_gap


# ---------- 캐시 ----------
def build():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    out = {}
    for arm, pref in [("A", "rq1lv"), ("B", "rq1sz")]:
        for s in ["AM", "SYMNCO", "POMO"]:
            p = os.path.join(DATA, "%s_%s.npz" % (pref, s))
            if not os.path.exists(p):
                print("  missing", p)
                continue
            print("  computing arm %s %s ..." % (arm, s), flush=True)
            base, shift = prep(p)
            out["%s_%s" % (arm, s)] = window_sets(shift)
            if arm == "A" and s in ("AM", "SYMNCO"):
                out["rho_%s" % s] = rho_sets(base, shift)
    np.savez_compressed(CACHE, **out)
    return out


# ---------- 그림 ----------
def fig1(T):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for a, s in zip(ax, ["AM", "SYMNCO", "POMO"]):
        M = T["A_%s" % s]; H, L, D = M[:, 0], M[:, 1], M[:, 3]
        a.scatter(H, D, s=9, c=CLR[s], alpha=.55, lw=0)
        r = spearmanr(presid(H, L), presid(D, L))[0]
        a.set_title("%s\nspread | level   rho = %+.3f" % (NAME[s], r), fontsize=8)
        a.set_xlabel("within-set spread of severity  (SD of log)")
    ax[0].set_ylabel("out-of-sample gain  d_split  (%)")
    fig.suptitle("Figure 1 — Spread predicts the gain; the negative control is flat",
                 fontsize=8.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_spread.png"), bbox_inches="tight")
    plt.close(fig)


def fig2(T):
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0))
    for i, (arm, knob) in enumerate([("A", "cluster knob"), ("B", "size knob")]):
        M = T["%s_AM" % arm]
        H, L, R, D = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
        for j, (v, short) in enumerate([(L, "severity"), (R, "tour length")]):
            a = ax[i, j]
            a.scatter(presid(v, H), presid(D, H), s=9, c=CLR["AM"], alpha=.55, lw=0)
            rr = spearmanr(presid(v, H), presid(D, H))[0]
            a.set_title("%s — %s | spread   rho = %+.3f" % (knob, short, rr), fontsize=8)
            a.set_xlabel("rank residual of set median %s" % short)
            a.set_ylabel("rank residual of d_split")
    fig.suptitle("Figure 2 — The level term reverses on the severity axis "
                 "and agrees on the length axis  (AM)", fontsize=8.5, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_level_reversal.png"), bbox_inches="tight")
    plt.close(fig)


def fig3(T):
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    labels, vals, cols = [], [], []
    for s in ["AM", "SYMNCO"]:
        M = T["rho_%s" % s]
        P, H, L, D = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
        u = np.unique(P)
        ax[0].plot(u, [H[P == x].mean() for x in u], "o-", ms=3, c=CLR[s], label=NAME[s])
        ax[1].plot(u, [D[P == x].mean() for x in u], "o-", ms=3, c=CLR[s], label=NAME[s])
        labels += ["%s\n| spread" % NAME[s], "%s\n| spread, level" % NAME[s]]
        vals += [spearmanr(presid(P, H), presid(D, H))[0],
                 spearmanr(presid(P, H, L), presid(D, H, L))[0]]
        cols += [CLR[s], CLR[s]]
    ax[0].set_xlabel("out-of-distribution share")
    ax[0].set_ylabel("set spread")
    ax[0].set_title("spread is non-monotone in composition", fontsize=8)
    ax[0].legend()
    ax[1].set_xlabel("out-of-distribution share")
    ax[1].set_ylabel("d_split (%)")
    ax[1].set_title("so is the gain", fontsize=8)
    y = np.arange(len(labels))[::-1]
    ax[2].barh(y, vals, color=cols, alpha=.85, height=.55)
    ax[2].set_yticks(y); ax[2].set_yticklabels(labels, fontsize=7)
    ax[2].axvline(0, c="k", lw=.7)
    ax[2].set_xlabel("partial rank correlation of share")
    ax[2].set_title("composition after controls", fontsize=8)
    fig.suptitle("Figure 3 — Composition acts through spread, with a solver-dependent remainder",
                 fontsize=8.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_composition.png"), bbox_inches="tight")
    plt.close(fig)


# §7.2 표의 기록값 (rq3 정책 모델은 이 스크립트가 재학습하지 않는다)
POLICY = {
    "AM":     {"signal-free": (17.56, 13.11, 23.71), "learned (charged)": (10.46, 2.79, 18.10),
               "label 20:1": (8.01, 2.87, 14.91), "probe CV": (3.84, -1.29, 9.15)},
    "SYMNCO": {"signal-free": (14.68, 12.00, 17.42), "learned (charged)": (12.07, 5.73, 19.47),
               "label 20:1": (7.06, 2.91, 12.39), "probe CV": (4.81, -0.64, 11.00)},
    "POMO":   {"signal-free": (0.01, -0.25, 0.38), "learned (charged)": (-0.18, -0.86, 0.53),
               "label 20:1": (-3.38, -5.61, -1.34), "probe CV": (-0.26, -1.49, 0.86)},
}


def fig4():
    keys = ["signal-free", "learned (charged)", "label 20:1", "probe CV"]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for a, s in zip(ax, ["AM", "SYMNCO", "POMO"]):
        y = np.arange(len(keys))[::-1]
        for k, yy in zip(keys, y):
            m, lo, hi = POLICY[s][k]
            hl = (k == "learned (charged)")
            a.plot([lo, hi], [yy, yy], lw=2.6 if hl else 1.4,
                   c=CLR[s] if hl else "#9a9a9a", solid_capstyle="round")
            a.plot(m, yy, "o", ms=4.5 if hl else 3.2, c=CLR[s] if hl else "#5a5a5a")
        a.axvline(0, c="k", lw=.7)
        a.set_yticks(y)
        a.set_yticklabels(keys if s == "AM" else [], fontsize=7)
        a.set_title(NAME[s], fontsize=8)
        a.set_xlabel("gain (%)")
    fig.suptitle("Figure 4 — A policy that charges its own probe, "
                 "against a same-collection baseline", fontsize=8.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_policy.png"), bbox_inches="tight")
    plt.close(fig)


# 사전 명세된 사후 검정 2건 — 등록 문서 §12 의 기록값
TEST_U = [("cluster / AM", -0.814, -0.856, -0.691),
          ("cluster / Sym-NCO", -0.820, -0.857, -0.583),
          ("size / AM", -0.681, -0.777, -0.483),
          ("size / Sym-NCO", -0.580, -0.712, -0.252)]
TEST_R = [("AM  unrestricted", 0.791, None, None),
          ("AM  range-matched", 0.576, 0.237, 0.665),
          ("Sym-NCO  unrestricted", 0.810, None, None),
          ("Sym-NCO  range-matched", 0.493, 0.159, 0.683)]


def fig5():
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6))
    y = np.arange(len(TEST_U))[::-1]
    for (lab, pt, lo, hi), yy in zip(TEST_U, y):
        c = CLR["AM"] if "AM" in lab else CLR["SYMNCO"]
        ax[0].plot([lo, hi], [yy, yy], lw=2.2, c=c, solid_capstyle="round")
        ax[0].plot(pt, yy, "o", ms=4, c=c)
    ax[0].axvline(0, c="k", lw=.7)
    ax[0].set_yticks(y); ax[0].set_yticklabels([t[0] for t in TEST_U], fontsize=7)
    ax[0].set_xlabel("tour length | spread,  pure absolute difference")
    ax[0].set_title("Test U — the scale reading is not a unit artefact", fontsize=8)

    y = np.arange(len(TEST_R))[::-1]
    for (lab, pt, lo, hi), yy in zip(TEST_R, y):
        c = CLR["AM"] if "AM" in lab else CLR["SYMNCO"]
        if lo is not None:
            ax[1].plot([lo, hi], [yy, yy], lw=2.2, c=c, solid_capstyle="round")
            ax[1].plot(pt, yy, "o", ms=4, c=c)
        else:
            ax[1].plot(pt, yy, "s", ms=4, c=c, mfc="none")
    ax[1].axvline(0, c="k", lw=.7)
    ax[1].axvline(-0.030, c="#8a2f0a", lw=.8, ls="--")
    ax[1].text(-0.030, len(TEST_R) - .35, " size knob  -0.030", fontsize=6.5, color="#8a2f0a")
    ax[1].set_yticks(y); ax[1].set_yticklabels([t[0] for t in TEST_R], fontsize=7)
    ax[1].set_xlabel("spread | level")
    ax[1].set_title("Test R — range explains part, not all", fontsize=8)
    fig.suptitle("Figure 5 — Two post-hoc tests whose verdict sentences were fixed "
                 "before they ran", fontsize=8.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_pretests.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("building set tables (cached in _figcache.npz) ...")
    T = build()
    fig1(T); print("  fig1 ok")
    fig2(T); print("  fig2 ok")
    fig3(T); print("  fig3 ok")
    fig4();  print("  fig4 ok")
    fig5();  print("  fig5 ok")
    print("done ->", os.path.abspath(OUT))
