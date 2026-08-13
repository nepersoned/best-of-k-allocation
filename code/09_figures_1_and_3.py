# Fig 1 (audit headline) + Fig 3 (two-sided discrimination)
# 모든 수치는 expand_report / confirm_report / 바닥 스윕 기록에서만.
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOL = ["POMO", "AM", "SYMNCO"]
COL = {"POMO": "#4c72b0", "AM": "#dd8452", "SYMNCO": "#55a868"}

# ---- 동질 workload, Axis B (표준 프로토콜) — expand_report + 바닥 스윕 ----
HOMO = {
 "POMO":   dict(d_in=2.227, ci_in=(1.63, 2.82), d_sp= 0.457, ci_sp=(-0.44, 1.34), fl=(2.083, 4.211)),
 "AM":     dict(d_in=2.567, ci_in=(1.96, 3.10), d_sp= 0.015, ci_sp=(-1.08, 1.06), fl=(3.440, 5.488)),
 "SYMNCO": dict(d_in=2.206, ci_in=(1.61, 2.72), d_sp=-0.512, ci_sp=(-1.79, 0.38), fl=(3.553, 5.793)),
}
# ---- 혼합 workload, Axis A — confirm_report ----
MIX = {
 "AM":     dict(d_in=11.927, d_sp=11.549, ci_sp=( 7.401, 19.734), fl=(1.218, 2.608)),
 "SYMNCO": dict(d_in=12.421, d_sp=12.017, ci_sp=( 5.183, 19.982), fl=(1.066, 2.991)),
 "POMO":   dict(d_in= 1.040, d_sp=-0.290, ci_sp=(-0.720,  0.238), fl=(1.033, 1.447)),
}

# ========================= Figure 1 =========================
fig, ax = plt.subplots(figsize=(8.2, 4.4))
w = 0.34
for i, s in enumerate(SOL):
    h = HOMO[s]
    # 바닥: median–p90 대역 (정의 민감도를 점이 아니라 폭으로)
    ax.add_patch(plt.Rectangle((i - 0.5, h["fl"][0]), 1.0, h["fl"][1] - h["fl"][0],
                               color="0.75", alpha=0.45, zorder=0,
                               label="noise floor, median–p90" if i == 0 else None))
    ax.hlines(h["fl"][0], i - 0.5, i + 0.5, color="0.45", lw=1.0, zorder=1)
    ax.hlines(h["fl"][1], i - 0.5, i + 0.5, color="0.45", lw=1.0, ls=":", zorder=1)
    # d_in
    ax.bar(i - w/2, h["d_in"], w, color=COL[s], alpha=.95, zorder=3,
           label="in-sample  $d_{in}$" if i == 0 else None)
    ax.errorbar(i - w/2, h["d_in"], yerr=[[h["d_in"]-h["ci_in"][0]], [h["ci_in"][1]-h["d_in"]]],
                fmt="none", ecolor="0.15", capsize=3, lw=1.2, zorder=4)
    # d_split
    ax.bar(i + w/2, h["d_sp"], w, color="white", edgecolor=COL[s], hatch="///", lw=1.4, zorder=3,
           label="out-of-sample  $d_{split}$" if i == 0 else None)
    ax.errorbar(i + w/2, h["d_sp"], yerr=[[h["d_sp"]-h["ci_sp"][0]], [h["ci_sp"][1]-h["d_sp"]]],
                fmt="none", ecolor="0.15", capsize=3, lw=1.2, zorder=4)
ax.axhline(0, color="k", lw=1.1, zorder=2)
ax.set_xticks(range(len(SOL))); ax.set_xticklabels(SOL)
ax.set_ylabel("allocation gain  $d$  (%)")
ax.set_xlim(-0.6, len(SOL) - 0.4); ax.set_ylim(-2.6, 8.0)
ax.set_title("Fig 1 — In-distribution workload (uniform TSP-100, Axis B, N=50, S/N=100)")
ax.legend(fontsize=8, loc="upper right", framealpha=.95)
ax.grid(axis="y", alpha=.22)
for i, s in enumerate(SOL):
    ax.annotate("CI crosses 0", (i + w/2, HOMO[s]["ci_sp"][1] + 0.18),
                ha="center", fontsize=7.5, color="0.25")
plt.tight_layout(); plt.savefig("fig1_audit_headline.png", dpi=150, bbox_inches="tight")
print("saved fig1_audit_headline.png")

# ========================= Figure 3 =========================
fig, ax = plt.subplots(figsize=(9.0, 4.6))
MK = {"POMO": "X", "AM": "o", "SYMNCO": "s"}
xs = {}
pos = 0
for wl, D, lab in [("in-distribution", HOMO, "in-distribution\n(uniform, Axis B)"),
                   ("shifted", MIX, "distribution-shifted\n(uni+clustered, Axis A)")]:
    for s in SOL:
        d = D[s]; xs[(wl, s)] = pos
        # 바닥 대역
        ax.add_patch(plt.Rectangle((pos - .38, d["fl"][0]), .76, d["fl"][1]-d["fl"][0],
                                   color="0.78", alpha=.45, zorder=0))
        # d_in -> d_split 화살표 (= 보정의 효과)
        ax.annotate("", xy=(pos, d["d_sp"]), xytext=(pos, d["d_in"]),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.5,
                                    shrinkA=3, shrinkB=3), zorder=2)
        ax.plot(pos, d["d_in"], marker=MK[s], ms=8, mfc="white", mec=COL[s], mew=1.8, zorder=3)
        ax.errorbar(pos, d["d_sp"], yerr=[[d["d_sp"]-d["ci_sp"][0]], [d["ci_sp"][1]-d["d_sp"]]],
                    fmt=MK[s], ms=8, color=COL[s], ecolor="0.2", capsize=3, lw=1.3, zorder=4)
        pos += 1
    pos += 0.8
ax.axhline(0, color="k", lw=1.1, zorder=1)
ax.axvline(2.9, color="0.6", ls="--", lw=1)
ax.set_xticks([xs[(w_, s)] for w_ in ("in-distribution", "shifted") for s in SOL])
ax.set_xticklabels(SOL * 2, fontsize=9)
ax.tick_params(axis="x", pad=2)
ax.text(1.0, -3.6, "in-distribution\n(uniform, Axis B)", ha="center", fontsize=9, color="0.25")
ax.text(4.8, -3.6, "distribution-shifted\n(uni+clustered, Axis A)", ha="center", fontsize=9, color="0.25")
ax.set_ylabel("allocation gain  $d$  (%)"); ax.set_ylim(-5.6, 23)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", mew=1.6, label="$d_{in}$ (in-sample)"),
    Line2D([], [], marker="o", ls="", color="0.3", label="$d_{split}$ (out-of-sample, 95% CI)"),
    Line2D([], [], marker="X", ls="", color="0.3", label="POMO = negative control"),
    Line2D([], [], color="0.78", lw=8, alpha=.6, label="noise floor, median–p90"),
], fontsize=8, loc="upper left", framealpha=.95)
ax.grid(axis="y", alpha=.22)
ax.set_title("Fig 3 — The same correction, applied to both workloads")
plt.tight_layout(); plt.savefig("fig3_two_sided.png", dpi=150, bbox_inches="tight")
print("saved fig3_two_sided.png")
