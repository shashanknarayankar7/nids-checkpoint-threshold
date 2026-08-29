# =============================================================================
# STEP 3 - PAPER-READY TABLES AND FIGURES
# Run this in the same notebook as Step 2, after it finishes.
# Produces the four tables your paper needs, plus two figures.
# =============================================================================

import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "/kaggle/working"
with open(os.path.join(OUTPUT_DIR, "results_raw.json")) as f:
    R = json.load(f)

df = pd.DataFrame([{k: r[k] for k in
                    ["protocol", "strategy", "seed", "accuracy",
                     "balanced_accuracy", "macro_f1", "weighted_f1", "mcc",
                     "n_params", "model_size_mb", "sec_per_epoch",
                     "batch1_latency_ms", "n_train", "n_test"]} for r in R])

CLASS_NAMES = list(R[0]["per_class"].keys())


def ms(g, col):
    """mean +- std formatted string"""
    m, s = g[col].mean(), g[col].std()
    return f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"


# ---------------------------------------------------------------- TABLE 1
print("=" * 78)
print("TABLE 1. Overall performance by evaluation protocol and imbalance strategy")
print("=" * 78)
rows = []
for (p, s), g in df.groupby(["protocol", "strategy"]):
    rows.append({
        "Protocol": "A (as published)" if p == "A" else "B (leakage-free)",
        "Strategy": s,
        "Accuracy": ms(g, "accuracy"),
        "Balanced Acc.": ms(g, "balanced_accuracy"),
        "Macro-F1": ms(g, "macro_f1"),
        "Weighted-F1": ms(g, "weighted_f1"),
        "MCC": ms(g, "mcc"),
    })
t1 = pd.DataFrame(rows)
print(t1.to_string(index=False))
t1.to_csv(f"{OUTPUT_DIR}/table1_overall.csv", index=False)


# ---------------------------------------------------------------- TABLE 2
print("\n" + "=" * 78)
print("TABLE 2. The leakage gap (Protocol A minus Protocol B)")
print("=" * 78)
rows = []
for s in df.strategy.unique():
    a = df[(df.protocol == "A") & (df.strategy == s)]
    b = df[(df.protocol == "B") & (df.strategy == s)]
    if len(a) and len(b):
        rows.append({
            "Strategy": s,
            "Δ Accuracy": f"{a.accuracy.mean() - b.accuracy.mean():+.4f}",
            "Δ Balanced Acc.": f"{a.balanced_accuracy.mean() - b.balanced_accuracy.mean():+.4f}",
            "Δ Macro-F1": f"{a.macro_f1.mean() - b.macro_f1.mean():+.4f}",
            "Δ MCC": f"{a.mcc.mean() - b.mcc.mean():+.4f}",
        })
t2 = pd.DataFrame(rows)
print(t2.to_string(index=False))
t2.to_csv(f"{OUTPUT_DIR}/table2_leakage_gap.csv", index=False)


# ---------------------------------------------------------------- TABLE 3
print("\n" + "=" * 78)
print("TABLE 3. Per-class recall - where the damage actually lands")
print("=" * 78)
rec = {}
for p in ["A", "B"]:
    for s in df.strategy.unique():
        runs = [r for r in R if r["protocol"] == p and r["strategy"] == s]
        if not runs:
            continue
        rec[f"{p}/{s}"] = {c: np.mean([r["per_class"][c]["recall"] for r in runs])
                           for c in CLASS_NAMES}
t3 = pd.DataFrame(rec).round(4)
t3.index.name = "Class"
print(t3.to_string())
t3.to_csv(f"{OUTPUT_DIR}/table3_per_class_recall.csv")


# ---------------------------------------------------------------- TABLE 4
print("\n" + "=" * 78)
print("TABLE 4. Computational cost (what the original paper did not report)")
print("=" * 78)
t4 = (df.groupby(["protocol", "strategy"])
        [["n_params", "model_size_mb", "sec_per_epoch",
          "batch1_latency_ms", "n_train"]].mean().round(3))
print(t4.to_string())
t4.to_csv(f"{OUTPUT_DIR}/table4_efficiency.csv")

print(f"\nModel parameters: {int(df.n_params.iloc[0]):,}")
print(f"Model size (FP32): {df.model_size_mb.iloc[0]:.2f} MB")
print(f"Single-sample CPU latency: {df.batch1_latency_ms.mean():.2f} ms")
print("Paper reported: 150 s/epoch on an Intel i9 + RTX 3060, and no "
      "parameter count, model size, or inference latency at all.")


# ---------------------------------------------------------------- FIGURE 1
fig, ax = plt.subplots(figsize=(9, 4.5))
strategies = list(df.strategy.unique())
x = np.arange(len(strategies))
w = 0.35
for i, (p, lab, col) in enumerate([("A", "Protocol A (as published)", "#c44"),
                                   ("B", "Protocol B (leakage-free)", "#48c")]):
    vals = [df[(df.protocol == p) & (df.strategy == s)].macro_f1.mean()
            for s in strategies]
    errs = [df[(df.protocol == p) & (df.strategy == s)].macro_f1.std()
            for s in strategies]
    errs = [0 if np.isnan(e) else e for e in errs]
    ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels(strategies)
ax.set_ylabel("Macro-F1"); ax.set_xlabel("Imbalance handling strategy")
ax.set_title("Macro-F1 under leaked vs. leakage-free evaluation")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_leakage_gap.png", dpi=300)
plt.show()


# ---------------------------------------------------------------- FIGURE 2
fig, ax = plt.subplots(figsize=(10, 5))
t3.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("Recall"); ax.set_xlabel("Attack class")
ax.set_title("Per-class recall across protocols and strategies")
ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_per_class_recall.png", dpi=300)
plt.show()

print("\nAll tables (.csv) and figures (.png) saved to /kaggle/working/")
