# Paper 2 — Spread and Scale

Code, cost arrays, pre-registrations and figures for *"Spread and Scale: What
Determines Whether Test-Time Budget Allocation Pays"* (preprint forthcoming).
The paper is distributed as a preprint; this directory holds only the material
needed to recompute it.

The companion audit — *"Sampling Luck Masquerades as Allocation Gain"*,
[arXiv:2608.13087](https://arxiv.org/abs/2608.13087) — is at the root of this
repository and supplies the estimators this work builds on.

## What is here

```
registration/  the two pre-registration documents, with amendment logs
code/          the confirmation-2 analysis, hashed before collection finished
code_posthoc/  analyses that were not registered endpoints, kept separate
data/          15 .npz arrays of per-instance solution costs
figures/       the five figures, and the script that regenerates them
results/       the recorded endpoint values every number traces to
```

## Pre-registrations

`registration/rq1-preregistration.md` is the governing document. It records, in
order: the exploratory results seen before the endpoints were fixed (§1), the
endpoints themselves (§4, §9), an amendment log marking the direction each
change moves the decision criteria (§7), the confirmation results (§8, §10), two
post-hoc tests specified with their verdict sentences fixed before they were run
(§11), and those tests' outcomes (§12).

Two things in it are worth reading even if you skip the rest. §12.2 records a
test whose result fell between two pre-written verdicts, and says so instead of
picking the favourable one. §12.3 records a transcription error we found and
corrected while making the figures.

## Data

`rq1{c,conf,lv,sz}_{AM,POMO,SYMNCO}.npz` and `rq3_{...}_axisA.npz` — for each
run and solver, every sampled solution cost for 280 TSP instances, with the
LKH-3 reference tour length and the shift parameter per instance.

* `rq1c_*` exploratory, `rq1conf_*` confirmation 1.
* `rq1lv_*` confirmation 2 arm A (cluster-tightening knob).
* `rq1sz_*` confirmation 2 arm B (size knob, `n ∈ {100 … 200}`).
* `rq3_*` the policy run.

Each array is split three ways and the split is fixed across every analysis:
`[0:200]` severity axis, `[200:600]` allocation decision, `[600:1000]`
evaluation. The x-axis and the outcome never share samples.

Because every cost is stored, any allocation policy can be re-evaluated offline
without re-running inference.

## Reproducing

Everything here is CPU-only and runs on the stored arrays. `numpy`, `scipy`, and
`matplotlib` for the figures.

```
cd data && python ../code/analysis_confirm2.py     # registered endpoints only
cd code_posthoc && python 03_test_u_absolute.py ../data
cd figures && python make_figures.py ../data .
```

`code/analysis_confirm2.py` was written and hashed **before collection
finished** (`FROZEN.txt`) and run unmodified. It computes the registered
endpoints and nothing else. Scripts in `code_posthoc/` were written after
seeing results; `README.txt` there records which is which.

Figure 4 plots the recorded policy table rather than retraining the policy.

## Registered versus exploratory

The distinction is maintained in the paper, in the pre-registrations, and in the
directory layout. Two registered conditions were not met and are reported as
such: the size-knob arm did not reproduce the spread result on the primary
solver, and the negative-control solver's interval marginally excluded zero.

## License

MIT (see the repository root). The pretrained checkpoints are distributed by
their respective authors under their own terms.
