# =============================================================================
# STEP 1 - PREPARE DATA
# Reads the CICIoT2023 CSVs, maps 34 attack labels to 8 classes,
# builds a size-capped working set, saves it as ONE Parquet file.
#
# Run this ONCE in its own notebook. Save the output as a Kaggle Dataset.
# Accelerator: None (CPU is fine). Runtime: ~10-25 min.
# =============================================================================

import os, glob, gc, json
import numpy as np
import pandas as pd

# ----------------------------- SETTINGS --------------------------------------
INPUT_DIR   = "/kaggle/input"
OUTPUT_DIR  = "/kaggle/working"
BIG_FRAC    = 0.25      # keep this fraction of rows from the huge classes
CAP_PER_CLASS = 100_000 # final max rows per class (keeps SMOTE from exploding)
SEED        = 42
# -----------------------------------------------------------------------------

rng = np.random.RandomState(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 7 attack families + benign. Rule-based so it survives small
# spelling differences between Kaggle mirrors of the dataset.
BIG_CLASSES = {"Benign", "DDoS", "DoS", "Mirai"}

def to_family(label: str) -> str:
    """Map one of the 34 raw CICIoT2023 labels to one of 8 classes."""
    s = str(label).strip()
    low = s.lower()
    if low.startswith("benign"):
        return "Benign"
    if low.startswith("ddos"):
        return "DDoS"
    if low.startswith("dos"):
        return "DoS"
    if low.startswith("mirai"):
        return "Mirai"
    if low.startswith("recon") or "vulnerabilityscan" in low:
        return "Recon"
    if "spoofing" in low or low.startswith("mitm"):
        return "Spoofing"
    if "bruteforce" in low or "brute_force" in low or "dictionary" in low:
        return "BruteForce"
    # everything left is a web / application-layer attack:
    # XSS, SqlInjection, CommandInjection, Backdoor_Malware,
    # BrowserHijacking, Uploading_Attack
    return "Web"


def find_label_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.strip().lower() in ("label", "labels", "class", "attack"):
            return c
    raise ValueError(f"No label column found. Columns were: {list(df.columns)}")


# ---- 1. Find the CSV files --------------------------------------------------
csv_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True))
print(f"Found {len(csv_paths)} CSV files under {INPUT_DIR}")
if len(csv_paths) == 0:
    raise SystemExit(
        "No CSVs found. Use '+ Add Input' on the right-hand panel and attach a "
        "CICIoT2023 dataset, then re-run this cell."
    )
for p in csv_paths[:5]:
    print("   ", p)
if len(csv_paths) > 5:
    print(f"    ... and {len(csv_paths) - 5} more")


# ---- 2. Read them, one at a time, downsampling the huge classes as we go ----
chunks = []
label_col = None
total_raw = 0

for i, path in enumerate(csv_paths):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [skip] {os.path.basename(path)}: {e}")
        continue

    if df.empty:
        continue

    if label_col is None:
        label_col = find_label_column(df)
        print(f"\nUsing label column: '{label_col}'")

    if label_col not in df.columns:
        continue

    total_raw += len(df)

    df["family"] = df[label_col].map(to_family)
    df = df.drop(columns=[label_col])

    # keep every row of the rare families, sample the huge ones
    is_big = df["family"].isin(BIG_CLASSES)
    small_part = df[~is_big]
    big_part = df[is_big].sample(frac=BIG_FRAC, random_state=SEED + i)
    df = pd.concat([small_part, big_part], ignore_index=True)

    # shrink memory: every feature column becomes float32
    for c in df.columns:
        if c != "family":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    chunks.append(df)
    del small_part, big_part
    gc.collect()

    if (i + 1) % 20 == 0:
        print(f"  read {i+1}/{len(csv_paths)} files ...")

print(f"\nRaw rows seen: {total_raw:,}")
data = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()
print(f"After per-file downsampling: {len(data):,} rows")


# ---- 3. Clean --------------------------------------------------------------
feature_cols = [c for c in data.columns if c != "family"]
data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

before = len(data)
data = data.dropna(subset=feature_cols)
print(f"Dropped {before - len(data):,} rows with NaN/inf values")

# drop constant columns - they carry no information and can break scaling
nunique = data[feature_cols].nunique()
constant_cols = list(nunique[nunique <= 1].index)
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
    data = data.drop(columns=constant_cols)
    feature_cols = [c for c in feature_cols if c not in constant_cols]


# ---- 4. Cap each class -----------------------------------------------------
print("\nClass counts BEFORE capping:")
print(data["family"].value_counts())

capped = []
for fam, grp in data.groupby("family"):
    if len(grp) > CAP_PER_CLASS:
        grp = grp.sample(n=CAP_PER_CLASS, random_state=SEED)
    capped.append(grp)
data = pd.concat(capped, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
del capped
gc.collect()

print("\nClass counts AFTER capping (this is your working set):")
print(data["family"].value_counts())
print(f"\nFinal shape: {data.shape}  ({len(feature_cols)} features, 8 classes)")


# ---- 5. Save ---------------------------------------------------------------
out_path = os.path.join(OUTPUT_DIR, "ciciot2023_working_set.parquet")
data.to_parquet(out_path, index=False, compression="snappy")
size_mb = os.path.getsize(out_path) / 1e6
print(f"\nSaved -> {out_path}  ({size_mb:.1f} MB)")

meta = {
    "n_rows": int(len(data)),
    "n_features": int(len(feature_cols)),
    "feature_cols": feature_cols,
    "class_counts": {k: int(v) for k, v in data["family"].value_counts().items()},
    "big_frac": BIG_FRAC,
    "cap_per_class": CAP_PER_CLASS,
    "seed": SEED,
    "n_source_csvs": len(csv_paths),
    "raw_rows_seen": int(total_raw),
}
with open(os.path.join(OUTPUT_DIR, "prepare_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nDONE. Now: Save Version -> then add /kaggle/working as a Dataset "
      "so Step 2 can attach it as input.")
