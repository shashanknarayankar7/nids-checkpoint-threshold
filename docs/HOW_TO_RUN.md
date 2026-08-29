# How to run this on Kaggle — step by step

## Before you start

Open the target paper and keep it beside you. It is free:
**Susilo, Muis & Sari (2025), Sensors 25(2), 580** — https://www.mdpi.com/1424-8220/25/2/580

Two things in it matter most:
- **Algorithm 1** (the model, in Keras pseudocode) — my `build_model()` is a direct transcription
- **Section 2.2.1** (preprocessing) — this is where SMOTE is applied before the split

---

## Notebook 1 — Prepare the data

1. Kaggle → **Create → Notebook**
2. Right panel → **+ Add Input** → search `CICIoT2023` → add a dataset that
   contains the CSV files (the "10% subset" mirrors are ideal)
3. Right panel → **Accelerator: None**, **Persistence: Files only**
4. Paste `kaggle_step1_prepare_data.py` into a cell. Run it.
5. Read the printed class counts. You should see 8 classes:
   Benign, DDoS, DoS, Mirai, Recon, Spoofing, BruteForce, Web
6. **Save Version → Save & Run All (Commit)**
7. When it finishes: open the notebook's **Output** tab → **New Dataset** →
   name it `ciciot-working-set`

**Expected:** 10–25 minutes, ~800k rows, a parquet file of roughly 150–250 MB.

**If it says "No CSVs found":** you forgot step 2.
**If it says "No label column found":** print `df.columns` and tell me what you see.

---

## Notebook 2 — The experiment

1. New notebook → **+ Add Input** → your `ciciot-working-set` dataset
2. **Accelerator: None** for the pilot
3. Paste `kaggle_step2_run_experiments.py`. Leave the settings as they are:
   `SAMPLE_ROWS = 200_000`, `SEEDS = [0]`, `EPOCHS = 15`
4. Run it.

**This is your pilot. It takes about 30–60 minutes and answers the only
question that matters: is the leakage gap real?**

Look at the last block it prints:

```
THE LEAKAGE GAP  (Protocol A minus Protocol B)
  none     accuracy +0.0021   macro-F1 +0.0035   MCC +0.0028
  smote    accuracy +0.0410   macro-F1 +0.0733   MCC +0.0502
  focal    accuracy +0.0018   macro-F1 +0.0029   MCC +0.0021
```

**How to read it:**
- The `none` and `focal` rows should be near zero. They are your control —
  they show the split itself is fair.
- The `smote` row is the finding. If it is clearly bigger than the other two,
  **the leak is real and you have a paper.**
- If all three rows are near zero, the leak is small. Don't panic —
  skip to "If the leak is small" below.

---

## Notebook 2, full run

Once the pilot confirms the gap, change three settings and re-run:

```python
SAMPLE_ROWS = None          # whole working set
SEEDS       = [0,1,2,3,4]   # 5 seeds for mean ± std
EPOCHS      = 30
```

Switch **Accelerator → GPU T4 x2**. This is 30 runs and will take
roughly 3–6 hours. Well inside the 12-hour session limit, and it should
cost you only a few hours of your weekly GPU quota.

Then paste `kaggle_step3_make_tables.py` into a new cell below and run it.
It writes four CSV tables and two 300-dpi PNG figures straight into
`/kaggle/working/` — those go directly into your manuscript.

---

## What each table becomes in your paper

| File | Where it goes |
|---|---|
| `table1_overall.csv` | Results §4.1 — the main comparison |
| `table2_leakage_gap.csv` | **Your headline table.** Results §4.2 |
| `table3_per_class_recall.csv` | Results §4.3 — shows the damage is concentrated in minority classes |
| `table4_efficiency.csv` | Results §4.4 — the efficiency numbers the paper never reported |
| `fig1_leakage_gap.png` | Figure 1 |
| `fig2_per_class_recall.png` | Figure 2 |

---

## If the leak turns out to be small

You still have two full contributions, and neither depends on the leak:

1. **Cost-sensitive learning vs SMOTE.** Compare the `B/smote` and `B/focal`
   columns in Table 3. The original paper explicitly names cost-sensitive
   learning as future work and admits its spoofing performance is weak.
   If focal loss beats SMOTE on Spoofing / Web / BruteForce recall, that is a
   clean, publishable improvement on their own stated limitation.

2. **The efficiency characterization.** They reported one number
   (150 s/epoch on an RTX 3060) and nothing else. You report parameter count,
   model size, per-sample CPU latency, and epoch time — and you release the
   code. That alone is a solid methods contribution.

In that case retitle the paper toward *"Cost-Sensitive Learning as a
Leakage-Free Alternative to Oversampling in IoT Intrusion Detection"* and
present the leakage result as a smaller secondary finding.

---

## Things reviewers will ask — handle them upfront

**"Your data is capped at 100k per class, the paper used a 10% subset."**
Say so plainly in your Methods, give the exact counts, and state the reason
(SMOTE-balancing an unsampled DDoS class to parity is computationally
infeasible and would itself distort the comparison). Both protocols use the
*identical* working set, so the A-vs-B comparison stays valid — that is the
only thing that has to be controlled.

**"How do we know your reimplementation is faithful?"**
Show your Protocol A + SMOTE numbers next to their published 99.15% / 99.19%.
If you land in the same neighbourhood, your reconstruction is validated.
Put this in a small table and say exactly that.

**"Is this just an attack on one paper?"**
Never frame it that way. Write "SMOTE-before-split is widespread in this
literature; we use one representative recent study to quantify its effect."
Cite two or three other papers with the same ordering to show it is a
field-level pattern, not one team's mistake.

---

## Housekeeping that makes your paper stronger

- Make both notebooks **public** before submission. Put the links in a
  "Code and Data Availability" section. The paper you are correcting has none.
- Pin the exact Kaggle dataset version you used and record it.
- Keep `results_raw.json` — it contains every confusion matrix and per-class
  score, so you can build any extra table a reviewer asks for without re-running.
