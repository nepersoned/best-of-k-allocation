"""Figures for the merged paper, from the exact-curve re-analysis outputs."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
A = json.load(open("results/reanalysis_rq1_rq3.json"))
B = json.load(open("results/reanalysis2.json"))
P1 = json.load(open("results/reanalysis_p1.json"))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
SOLV = ["AM", "SYMNCO", "POMO"]
NAME = {"AM": "AM", "SYMNCO": "Sym-NCO", "POMO": "POMO"}
COL = {"AM": "#1f4e79", "SYMNCO": "#c55a11", "POMO": "#7f7f7f"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


# Fig 1: three estimators on the deterministic pool (Gate 1 values; exact = truth)
mc = {"POMO": 2.227, "AM": 2.567, "SYMNCO": 2.206}
sp = {"POMO": 0.493, "AM": 0.168, "SYMNCO": 0.215}
fig, ax = plt.subplots(figsize=(4.6, 2.6))
x = np.arange(3); w = 0.26
ex = [P1["p1"][s]["indist_axisB"]["point"] for s in SOLV]
lo = [P1["p1"][s]["indist_axisB"]["ci"][0] for s in SOLV]
hi = [P1["p1"][s]["indist_axisB"]["ci"][1] for s in SOLV]
ax.bar(x - w, [mc[s] for s in SOLV], w, label="Monte Carlo curves, in sample", color="#bdd7ee")
ax.bar(x, [sp[s] for s in SOLV], w, label="sample split", color="#f4b183")
ax.bar(x + w, ex, w, yerr=[np.array(ex) - lo, np.array(hi) - ex], capsize=2,
       label="exact curves (true value)", color="#1f4e79")
ax.set_xticks(x, [NAME[s] for s in SOLV]); ax.set_ylabel("headroom (%)")
ax.set_title("In-distribution headroom on a fixed pool of 800 (uniform TSP-100)", fontsize=8.5)
ax.legend(frameon=False, fontsize=7)
save(fig, "fig1_estimators")

# Fig 2: headroom vs OOD share (cluster knob, C2A)
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.4))
for s in ["AM", "SYMNCO"]:
    c = A["rq1"][f"C2A_cluster/{s}"]["composition"]["by_rho"]
    r = sorted(c, key=float)
    ax[0].plot([100 * float(k) for k in r], [c[k]["d"] for k in r], "o-", color=COL[s], label=NAME[s])
    ax[1].plot([100 * float(k) for k in r], [c[k]["spread"] for k in r], "o-", color=COL[s], label=NAME[s])
ax[0].set_xlabel("out-of-distribution share (%)"); ax[0].set_ylabel("headroom (%)")
ax[1].set_xlabel("out-of-distribution share (%)"); ax[1].set_ylabel("spread (SD of log severity)")
ax[0].legend(frameon=False)
save(fig, "fig2_composition")

# Fig 3 & 4: forest plots over the nine cells
cells = [("C1_cluster", "cluster, conf. 1"), ("C2A_cluster", "cluster, conf. 2"), ("C2B_size", "size, conf. 2")]
def forest(ax, key, title):
    y = 0; ticks = []
    for tag, lab in cells:
        for s in SOLV:
            t = A["rq1"][f"{tag}/{s}"][key]
            ax.errorbar(t["point"], y, xerr=[[t["point"] - t["ci"][0]], [t["ci"][1] - t["point"]]],
                        fmt="o", color=COL[s], ms=3.5, capsize=2)
            ticks.append(f"{lab} / {NAME[s]}"); y -= 1
        y -= 0.5
    ys = [i for i in range(0, -12, -1)]
    pos = []; y = 0
    for _ in cells:
        for _ in SOLV:
            pos.append(y); y -= 1
        y -= 0.5
    ax.set_yticks(pos, ticks, fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-1, 1); ax.set_title(title, fontsize=8.5)

fig, ax = plt.subplots(figsize=(4.2, 3.4))
forest(ax, "spread|level", "spread | level  (partial Spearman with headroom)")
save(fig, "fig3_spread")

fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
forest(ax[0], "level|spread", "level | spread")
forest(ax[1], "reflen|spread", "median tour length | spread")
save(fig, "fig4_level_vs_length")

# Fig 5: charged probe rules vs oracle and label rule
fig, ax = plt.subplots(figsize=(4.8, 2.6))
names = [("oracle", "oracle (uncharged)", "#d9d9d9"), ("label", "label 20:1", "#a9d18e"),
         ("probe_cv", "probe CV (charged)", "#f4b183"), ("probe_medmin", "probe (median−min)/median (charged)", "#1f4e79")]
x = np.arange(3); w = 0.2
for j, (k, lab, c) in enumerate(names):
    vals, lo, hi = [], [], []
    for s in SOLV:
        if k in ("oracle", "label"):
            r = A["rq3"][s]; p = r[k]; ci = r[k + "_ci"] if k == "label" else r["oracle_ci"]
        else:
            r = B["rq3_rules"][f"{s}/m20"][k]; p = r["point"]; ci = r["ci"]
        vals.append(p); lo.append(p - ci[0]); hi.append(ci[1] - p)
    ax.bar(x + (j - 1.5) * w, vals, w, yerr=[lo, hi], capsize=2, color=c, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x, [NAME[s] for s in SOLV]); ax.set_ylabel("gain over uniform (%)")
ax.set_ylim(-14, 22)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
save(fig, "fig5_policy")
print("figures written to", os.path.abspath(OUT))
