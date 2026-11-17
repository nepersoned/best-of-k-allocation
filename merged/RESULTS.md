# Re-analysis with exact curves (2026-11-17)

Script: `reanalysis.py`. Raw output: `reanalysis_p1.json`, `reanalysis_rq1_rq3.json`.
Estimator: exact U-statistic best-of-k curves, greedy plug-in headroom, S/N = 100, cap 400.
For paper 2, the x-axis statistics come from samples [0:200] and the headroom from the disjoint samples [200:1000].
Sets, seeds, rank-residualized partials and the instance bootstrap (B = 400) are those of `analysis_confirm2.py`.

## Paper 1

| | AM | SymNCO | POMO |
|---|---|---|---|
| In-distribution, Axis B (true headroom) | 1.12 [0.64, 1.61] | 1.19 [0.73, 1.62] | 1.57 [1.06, 2.07] |
| In-distribution, Axis A | 1.99 [0.90, 2.99] | 2.49 [1.74, 3.42] | 0.67 [0.44, 0.90] |
| OOD 50:50, Axis A | 13.35 [9.28, 19.00] | 11.75 [6.64, 17.03] | 0.50 [0.36, 0.66] |

- The Axis B values match Gate 1 and APIN Appendix B exactly.
- Axis A in-distribution values may include about +0.1–0.3 of phantom gain (Gate 1, §2).
- The OOD values differ from Gate 1 (14.07 / 12.65) because the cap is 400 here and was 500 there.

## Paper 2 — partial Spearman correlations of headroom, 95% CI

| Arm / solver | spread \| level | level \| spread | tour length \| spread |
|---|---|---|---|
| C1 cluster / AM | +0.79 [+0.72, +0.83] | +0.50 [+0.17, +0.71] | −0.49 [−0.67, −0.17] |
| C1 cluster / SymNCO | +0.81 [+0.73, +0.86] | +0.76 [+0.48, +0.78] | −0.69 [−0.72, −0.45] |
| C1 cluster / POMO | +0.42 [+0.29, +0.53] | +0.13 [−0.07, +0.29] | −0.13 [−0.28, +0.07] |
| C2A cluster / AM | +0.81 [+0.72, +0.87] | +0.58 [+0.29, +0.71] | −0.56 [−0.69, −0.28] |
| C2A cluster / SymNCO | +0.78 [+0.71, +0.82] | +0.58 [+0.22, +0.63] | −0.56 [−0.61, −0.20] |
| C2A cluster / POMO | +0.61 [+0.48, +0.76] | +0.31 [+0.13, +0.39] | −0.32 [−0.40, −0.15] |
| C2B size / AM | **+0.43 [+0.32, +0.52]** | −0.90 [−0.94, −0.81] | −0.90 [−0.94, −0.80] |
| C2B size / SymNCO | +0.51 [+0.37, +0.59] | −0.88 [−0.90, −0.79] | −0.89 [−0.91, −0.81] |
| C2B size / POMO | +0.35 [+0.19, +0.49] | −0.78 [−0.82, −0.65] | −0.78 [−0.83, −0.64] |

Headroom ranges: cluster AM/SymNCO 0.8–21.8; POMO 0.1–0.75; size knob 0.1–1.75.

## Composition (C2A cluster)

- AM: share \| spread +0.04 [−0.10, +0.13]; spread \| share +0.43 [+0.20, +0.50]; share \| spread, level +0.11 [−0.06, +0.18].
- SymNCO: +0.08 [−0.12, +0.20]; +0.42 [+0.27, +0.51]; +0.16 [+0.02, +0.25].
- Headroom by OOD share (AM): 3.2, 12.3, 16.7, 18.7, 18.3, 17.0 at shares 0, 10, 25, 50, 75 and 100%.

## Policy collection (rq3), exact

- Signal-free oracle: AM 16.39 [13.31, 19.84]; SymNCO 15.69 [13.64, 17.56]; POMO 0.51 [0.39, 0.61].
- Label 20:1: AM 9.02 [6.86, 11.28]; SymNCO 8.15 [6.64, 9.69]; POMO −11.88 [−12.97, −10.25].
- The learned charged policy (10.46, 12.07) was not re-evaluated, because its model is not stored. The implied recovery is about 64% for AM and 77% for SymNCO. These numbers mix two estimators and must be recomputed before they go into the paper.

## What changed relative to the preprints

1. **The size-knob spread endpoint now passes on all three solvers.** For AM it moves from +0.16 [−0.03, +0.30] to +0.43 [+0.32, +0.52]. Paper 2's "did not replicate" was estimator noise.
2. **The spread result under the cluster knob holds**, at the same size or slightly larger.
3. **The level reversal persists and is sharper**: + under the cluster knob, −0.78 to −0.90 under the size knob. Tour length | spread is negative in all 9 cells and excludes zero in 8 of 9\| spread is negative in 8 of 9 cells, so the scale reading ("shorter instances gain more") is stronger.
4. **POMO as a negative control:** its correlations are now positive, but its headroom is tiny (≤ 0.75%). The negative control should be stated in terms of magnitude, not correlation.
5. **Composition adds nothing beyond spread for AM and something for SymNCO** (as in paper 2). The interior maximum is near 50%.
6. **spread of log b**, the theory variable fitted on the 200-sample slice, does not beat severity spread here. Do not use it as the headline variable.

## Batch 2 (`reanalysis2.py`, `reanalysis2.json`)

### Test U: the scale reading in absolute length units (tour length | spread)
- cluster AM −0.90 [−0.91, −0.82]; cluster SymNCO −0.89 [−0.90, −0.74]
- size AM −0.53 [−0.62, −0.30]; **size SymNCO −0.07 [−0.18, +0.09]: contains zero**
- So the scale reading survives absolute units in 3 of 4 cells, not 4 of 4 as the preprint reported. The paper has to say this.

### Simple charged probe rules (rq3)
The probe is m samples from [0:200], charged against the budget and kept. Statistics are computed on raw tour costs, with no reference tour. Scoring uses exact curves on [200:1000].

| Solver | Oracle (uncharged) | Probe CV, m=20 | **Probe (median−min)/median, m=20** | Difference (medmin − CV) | Recovery (medmin) |
|---|---|---|---|---|---|
| AM | 16.39 [13.31, 19.84] | 4.91 [3.04, 7.22] | **10.09 [7.54, 12.89]** | +5.18 [3.97, 6.74] | 62% |
| SymNCO | 15.69 [13.64, 17.56] | 5.13 [3.29, 7.17] | **10.40 [8.61, 12.14]** | +5.28 [3.98, 6.54] | 66% |
| POMO | 0.51 [0.39, 0.61] | −0.20 [−0.30, −0.10] | −0.16 [−0.27, −0.02] | +0.05 [−0.07, 0.17] | — |

- At m = 40, medmin gives 9.21 (AM) and 9.51 (SymNCO); CV gives 4.22 and 4.65.
- The single-feature medmin rule matches the preprint's learned policy (10.46 / 12.07) and beats CV with a clear margin. So the learned policy is not needed.
- On POMO, every charged rule loses slightly. This supports "do not allocate where the workload does not disturb the solver."
