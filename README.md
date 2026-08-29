# Checkpoint Selection and Decision Thresholds in Deep NIDS

Code, results and figures for a study of two evaluation conventions in
deep-learning-based network intrusion detection: reporting the final training
epoch, and classifying at argmax.

**Base architecture:** Susilo, B., Muis, A., Sari, R.F. (2025). *Intelligent
Intrusion Detection System Against Various Attacks Based on a Hybrid Deep
Learning Algorithm.* Sensors 25(2), 580. https://doi.org/10.3390/s25020580

**Dataset:** CICIoT2023 — Neto et al. (2023), Sensors 23(13), 5941.
https://doi.org/10.3390/s23135941

---

## What is here

```
code/        Kaggle scripts, in run order (step1 → step4)
results/
  json/      raw per-run records from every experiment
  tables/    derived CSV tables
  figures/   figures used in the manuscript
references/  BibTeX bibliography and the screening spreadsheet
docs/        step-by-step run instructions
```

Large per-run probability arrays (`probs_*.npz`, ~400 MB) are **not** in this
repository. They are in the Zenodo archive linked above; `.gitignore` excludes
them here to keep the repository small.

---

## Experiments

| File | What it contains | Runs |
|---|---|---|
| `day3_results_full.json` | 2 protocols x 3 strategies x 5 seeds | 30 |
| `day4_stability_results.json` | checkpoint selection, both protocols | 40 |
| `day5_focal_stability_results.json` | focal loss stability | 20 |
| `day7_rescued_perclass.json` | per-class recall, final vs rescued | 20 |
| `day8_threshold_analysis.json` | PR-AUC, argmax-F1, best-threshold F1 | 20 |
| `pilot_results_diagnostic.json` | pilot run | 30 |

Total: 160 training runs, all on a Kaggle Tesla T4 x2 instance.

**Protocol A** applies SMOTE and feature scaling to the full dataset before the
train/test split, reproducing the ordering described in Section 2.2.1 of the
base paper. **Protocol B** fits the scaler and applies resampling to the
training partition only.

---

## Reproducing

1. Read `docs/HOW_TO_RUN.md`.
2. Run `code/kaggle_step1_prepare_data.py` to build the working set from the
   CICIoT2023 CSVs. Output: a parquet file with 547,944 rows, 44 features,
   8 classes.
3. Run `code/kaggle_step2_run_experiments_v3.py` for the main grid.
4. Run `code/kaggle_step4_threshold_analysis.py` for the threshold study.
5. Run `code/kaggle_step3_make_tables.py` to regenerate the tables.

Environment: see `requirements.txt`. All experiments run on free Kaggle GPU
quota; no local hardware is required.

---

## Figures

| File | Content |
|---|---|
| `fig_class_dist.png` | class distribution of the working set |
| `fig_final_vs_rescued.png` | final-epoch vs validation-selected accuracy, per run |
| `fig_variance.png` | run-to-run variance under each reporting convention |
| `fig_val_curves.png` | validation trajectories, protocol B |
| `fig_perclass.png` | per-class recall across four conditions |
| `fig_threshold_artifact.png` | PR-AUC vs argmax-F1 vs best-threshold F1 |
| `fig1_pr_curves.png` | precision-recall curves, rare classes |
| `fig2_seed_stability.png` | PR-AUC seed sensitivity by class and strategy |

---

## License

MIT (see `LICENSE`). The CICIoT2023 dataset is distributed by the Canadian
Institute for Cybersecurity under its own terms.
