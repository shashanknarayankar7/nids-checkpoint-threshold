# =============================================================================
# STEP 4 - THRESHOLD / PRECISION-RECALL ANALYSIS
#
# Run AFTER step 2 v3. Uses the saved probability files - no retraining.
# Takes a few seconds.
#
# The question this answers: did SMOTE and focal loss actually teach the
# model anything new about rare classes, or did they just move the decision
# threshold? PR-AUC is threshold-independent, so if cross-entropy's PR-AUC
# matches theirs, the answer is "just moved the threshold".
# =============================================================================

import json, glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

OUTPUT_DIR = "/kaggle/working"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

with open(os.path.join(OUTPUT_DIR, "class_index.json")) as f:
    CLASS_IDX = json.load(f)
print("Classes:", CLASS_IDX)

RARE = ["BruteForce", "Web"]          # the classes where the story lives
STRATS = ["none", "smote", "focal"]
NICE = {"none": "CrossEntropy", "smote": "SMOTE", "focal": "Focal"}


def load(protocol, strategy, seed=0):
    p = os.path.join(OUTPUT_DIR, f"probs_{protocol}_{strategy}_s{seed}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["y_prob"], d["y_true"]


# =============================================================================
# 1. PR-AUC - is the signal there regardless of threshold?
# =============================================================================
print("\n" + "=" * 78)
print("PR-AUC BY CLASS (protocol B) - threshold-independent signal quality")
print("=" * 78)
print("Higher = the model genuinely separates this class better.")
print("If CrossEntropy matches SMOTE/Focal here, rebalancing added no new")
print("information - it only shifted where the decision boundary sits.\n")

rows = []
for cls, ci in CLASS_IDX.items():
    row = {"Class": cls}
    for s in STRATS:
        got = load("B", s)
        if got is None:
            continue
        y_prob, y_true = got
        binary = (y_true == ci).astype(int)
        p, r, _ = precision_recall_curve(binary, y_prob[:, ci])
        row[NICE[s]] = round(auc(r, p), 4)
        row["Support"] = int(binary.sum())
    rows.append(row)

pr = pd.DataFrame(rows).set_index("Class")
cols = [c for c in ["Support", "CrossEntropy", "SMOTE", "Focal"] if c in pr.columns]
pr = pr[cols]
print(pr.to_string())
pr.to_csv(os.path.join(OUTPUT_DIR, "table_pr_auc.csv"))


# =============================================================================
# 2. OPERATING POINTS - what precision can you get at a target recall?
# =============================================================================
print("\n" + "=" * 78)
print("OPERATING POINTS ON THE RARE CLASSES")
print("=" * 78)
print("For each target recall, the best precision achievable by moving the")
print("threshold, and the resulting false-alarm cost.\n")

op_rows = []
for cls in RARE:
    if cls not in CLASS_IDX:
        continue
    ci = CLASS_IDX[cls]
    print("-" * 78)
    print(f"{cls}")
    print("-" * 78)
    for s in STRATS:
        got = load("B", s)
        if got is None:
            continue
        y_prob, y_true = got
        binary = (y_true == ci).astype(int)
        n_pos = int(binary.sum())
        p, r, th = precision_recall_curve(binary, y_prob[:, ci])

        print(f"  {NICE[s]}   (PR-AUC {auc(r, p):.4f}, {n_pos} true instances)")
        for target in [0.3, 0.5, 0.7, 0.9]:
            i = int(np.argmin(np.abs(r - target)))
            prec = p[i]
            thr = th[i] if i < len(th) else 1.0
            tp = target * n_pos
            fp = tp * (1 - prec) / max(prec, 1e-9)
            print(f"      recall {target:.1f} -> precision {prec:.4f}  "
                  f"(threshold {thr:.4f})  ~{int(round(fp)):,} false alarms "
                  f"for ~{int(round(tp))} catches")
            op_rows.append({"class": cls, "strategy": NICE[s],
                            "target_recall": target, "precision": round(prec, 4),
                            "threshold": round(float(thr), 4),
                            "est_false_alarms": int(round(fp))})
        print()

pd.DataFrame(op_rows).to_csv(
    os.path.join(OUTPUT_DIR, "table_operating_points.csv"), index=False)


# =============================================================================
# 3. THE ARGMAX PENALTY - what default decision-making costs you
# =============================================================================
print("=" * 78)
print("ARGMAX vs BEST THRESHOLD - the cost of the default decision rule")
print("=" * 78)
print("'argmax' is what every paper reports. 'best F1' is what the same")
print("trained model can do with a tuned per-class threshold.\n")

cmp_rows = []
for cls in RARE:
    if cls not in CLASS_IDX:
        continue
    ci = CLASS_IDX[cls]
    for s in STRATS:
        got = load("B", s)
        if got is None:
            continue
        y_prob, y_true = got
        binary = (y_true == ci).astype(int)

        # argmax decision
        pred_arg = (y_prob.argmax(axis=1) == ci).astype(int)
        tp = int(((pred_arg == 1) & (binary == 1)).sum())
        fp = int(((pred_arg == 1) & (binary == 0)).sum())
        fn = int(((pred_arg == 0) & (binary == 1)).sum())
        pa = tp / max(tp + fp, 1)
        ra = tp / max(tp + fn, 1)
        fa = 2 * pa * ra / max(pa + ra, 1e-9)

        # best achievable F1 by threshold
        p, r, th = precision_recall_curve(binary, y_prob[:, ci])
        f1s = 2 * p * r / np.maximum(p + r, 1e-9)
        j = int(np.nanargmax(f1s))

        cmp_rows.append({
            "Class": cls, "Strategy": NICE[s],
            "argmax P": round(pa, 4), "argmax R": round(ra, 4),
            "argmax F1": round(fa, 4),
            "best P": round(p[j], 4), "best R": round(r[j], 4),
            "best F1": round(f1s[j], 4),
            "F1 gain": round(f1s[j] - fa, 4),
            "threshold": round(float(th[j]) if j < len(th) else 1.0, 4),
        })

cmp = pd.DataFrame(cmp_rows)
print(cmp.to_string(index=False))
cmp.to_csv(os.path.join(OUTPUT_DIR, "table_argmax_vs_threshold.csv"), index=False)


# =============================================================================
# 4. WHAT TO CONCLUDE
# =============================================================================
print("\n" + "=" * 78)
print("HOW TO READ THIS")
print("=" * 78)
print("""
1. If CrossEntropy PR-AUC >= SMOTE/Focal PR-AUC on the rare classes:
   rebalancing added NO new discriminative signal. It only moved the
   threshold. That is your paper's central claim, and it means practitioners
   can skip oversampling entirely and tune thresholds instead - cheaper,
   faster, no synthetic data, no leakage risk.

2. If 'best F1' >> 'argmax F1' for CrossEntropy: the standard reporting
   practice (argmax) systematically understates what the trained model can
   do on rare classes. Every paper in this literature reports the argmax
   number.

3. If SMOTE/Focal PR-AUC is genuinely higher: rebalancing DID help the
   representation, and your earlier precision-collapse finding stands as a
   caveat rather than a refutation. Report it that way - honestly.

Either outcome is publishable. Outcome 1 is the stronger paper.
""")

print("Saved: table_pr_auc.csv, table_operating_points.csv, "
      "table_argmax_vs_threshold.csv")
