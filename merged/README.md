# Merged paper: exact-curve re-analysis

This folder holds the material for the merged manuscript *When Does Reallocating a Sampling Budget Pay? Instance-wise Allocation for Neural Routing Solvers*. That paper supersedes the two preprints, arXiv:2608.13087 and arXiv:2609.27917.

It re-analyses the stored cost arrays of both preprints and collects no new data. Paper 1's arrays are in `../data/` and paper 2's are in `../paper2/data/`.

## What changed

The preprints estimated best-of-k curves by averaging random orderings of the stored samples, and corrected for selection by splitting the samples in half. Both steps bias the headroom. The re-analysis uses exact curves instead: the U-statistic over all orderings, Eq. (3) of the paper. The sets, seeds, partial correlations and instance bootstrap are unchanged from `../paper2/code/analysis_confirm2.py`. `RESULTS.md` summarises every number and lists what changed relative to the preprints.

## Files

```
reanalysis.py      paper-1 headroom; paper-2 spread/level/composition endpoints; label rule
reanalysis2.py     absolute-unit scale check (test U); charged probe rules
paired_label.py    paired bootstrap: probe rule minus label rule
estimator_sim.py   estimator accuracy and null at the paper settings (Appendix B)
sh_baseline.py     successive-halving baseline vs the probe rule
extras.py          tour-length translation, budget sweep, disjoint-workload test
make_figures.py    the five figures of the merged paper
results/           JSON output of the scripts above
figures/           figures (.pdf, .png)
RESULTS.md         summary of all re-analysis results
estimator_check/   simulation of estimator accuracy with known truth (Gate 1),
                   and the re-check of paper 1 on the deterministic pool
```

## Running

From this folder, run:

```
python reanalysis.py            # writes results/reanalysis_p1_rq1_rq3.json (about 5 min, CPU only)
python reanalysis.py p1         # paper 1 only
python reanalysis2.py
python paired_label.py
python make_figures.py
```

`reanalysis2.py` and `paired_label.py` import `reanalysis.py`.

The scripts in `estimator_check/` are run from inside that folder. `gate1_estimators.py` takes its regime from the environment variables `G_M`, `G_SN` and `G_CAP`.

Requirements are numpy, scipy and matplotlib.
