# Test-time budget allocation for neural combinatorial optimization

Companion repository for two papers. Each is distributed as a preprint; this
repository holds the cost arrays, the analysis pipelines and the recorded
result files.

**Paper 1 — Sampling Luck Masquerades as Allocation Gain**
([arXiv:2608.13087](https://arxiv.org/abs/2608.13087)). An audit of in-sample
selection bias in test-time allocation for neural combinatorial optimization.
Its material is at the root of this repository, in the layout the paper's
appendix describes.

**Paper 2 — Spread and Scale: What Determines Whether Test-Time Budget
Allocation Pays** (preprint forthcoming). A pre-registered study of which
workload property predicts whether allocation pays. Its material, including
both pre-registration documents, is in [`paper2/`](paper2/).

## What is here

```
code/     paper 1 — analysis pipeline, numbered in execution order
data/     paper 1 — raw per-instance solution costs (12 .npz arrays)
results/  paper 1 — the six files every number in the paper is traced to
figures/  paper 1 — the four figures as published
paper2/   paper 2 — code, arrays, pre-registrations, figures, results
```

Paper 1's directories stay at the root so that the paths its appendix cites keep
resolving.

## Data

`{expand,confirm}_{POMO,AM,SYMNCO}_axis{A,B}.npz` — for each run, solver and
allocation axis, the cost of every sampled solution for 100 TSP-100 instances
(50 uniform, 50 clustered), together with the LKH-3 reference tour length per
instance.

* **Axis A** — 1,000 stochastic decoding rollouts per instance.
* **Axis B** — the full `n × 8 = 800` deterministic multi-start × augmentation pool.
* `expand_*` is the exploratory run (instance seed 20260812, decoding seed 11);
  `confirm_*` is the pre-registered confirmatory run (20260813, 77). Everything
  else is identical between them.

Because every solution cost is stored, any allocation policy can be re-evaluated
offline without re-running inference. This is what makes the null distributions
in `04` affordable, and it forces every policy to be compared on identical
samples.

## Reproducing

`01` and `02` need a GPU (Colab T4 is sufficient; collection takes ~4 min per
solver). Everything from `03` onward is CPU-only and runs on the stored arrays.

```
01_gate_checkpoints_and_cvrp.py     checkpoint loading + CVRP reference stability gate
02_collect.py                       collection (set the two seeds at the top)
03_audit.py                         in-distribution audit table, floor curves
04_floor_definition_sweep.py        why the floor is a median, not a maximum
05_floor_operating_characteristics.py   floor vs K and vs N
06_confirmatory_endpoints.py        the four pre-registered endpoints
07_charged_variant.py               budget-charged policy (exploratory)
08_ood_share_sweep.py               gain vs out-of-distribution share (exploratory)
09_figures_1_and_3.py               figures
```

Scripts expect the `.npz` files in the working directory. Requirements:
`rl4co`, `elkai`, `numpy`, `matplotlib`, `torch` (and `pyvrp` for the CVRP gate
in `01`).

## Checkpoints

Three pretrained solvers from `ML4TSPBench` on the Hugging Face Hub:
`tsp100_pomo.pt`, `tsp100_am.pt`, `tsp100_symnco.pt`. They are legacy-format
`rl4co` state dicts and need key remapping to load into current `rl4co` —
`01_gate_checkpoints_and_cvrp.py` and `02_collect.py` both do this and assert
that no key is missing or unexpected, since a partial load would silently mix
randomly initialized layers into the measured curves.

## Exploratory versus pre-registered

`06` computes the pre-registered endpoints. `07` and `08` are exploratory: they
were not registered in advance, no test is reported on them, and the paper labels
them as descriptive. They are kept in separate scripts so the pre-registered path
can be re-run without them.

## License

MIT (see `LICENSE`). The pretrained checkpoints are distributed by their
respective authors under their own terms.
