# Kaggle notebooks

Exported from Kaggle with outputs retained, so results can be inspected
without rerunning.

| Notebook | Produces | Runs | Accelerator |
|---|---|---|---|
| `day4_stability.ipynb` | `day4_stability_results.json` | 40 | GPU T4 x2 |
| `day5_focal_stability.ipynb` | `day5_focal_stability_results.json` | 20 | GPU T4 x2 |
| `day6_analysis.ipynb` | summary tables, per-class recall | — | CPU |
| `day7_rescued_perclass.ipynb` | `day7_rescued_perclass.json` | 20 | GPU T4 x2 |
| `day8_threshold_analysis.ipynb` | `day8_threshold_analysis.json` | 20 | GPU T4 x2 |
| `day9_final_analysis.ipynb` | significance tests, threshold figure | — | CPU |

The pilot and the main grid (Days 1-3) are the `.py` scripts one level up:
`kaggle_step1` through `kaggle_step4`.

Each experiment notebook writes its JSON after every run and skips completed
entries on restart, so an interrupted session resumes without losing work.
