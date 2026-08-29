# =============================================================================
# STEP 2 - THE EXPERIMENT
#
# Rebuilds the AE-LSTM+CNN model from Algorithm 1 of:
#   Susilo, Muis & Sari (2025), Sensors 25(2), 580.
#
# Runs it under two evaluation protocols:
#   A ("as published") : SMOTE -> scale -> split      <- leaks test data
#   B ("leakage-free") : split -> scale on train -> SMOTE on train only
#
# and under three imbalance strategies: none / smote / focal loss.
#
# Accelerator: None (CPU) for the pilot. GPU T4 for the full run.
# =============================================================================

import os, json, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------- SETTINGS --------------------------------------
# >>> PILOT MODE: set to 200_000 for your first 1-hour sanity check.
# >>> FULL RUN:   set to None to use the whole working set.
SAMPLE_ROWS = 200_000

SEEDS       = [0]              # pilot: [0].  Full run: [0, 1, 2, 3, 4]
EPOCHS      = 15               # pilot: 15.   Full run: 30
BATCH_SIZE  = 512
TEST_SIZE   = 0.30             # the paper's 70/30 split
LR          = 0.01             # paper: SGD lr=0.01
MOMENTUM    = 0.9              # paper: momentum=0.9

PROTOCOLS   = ["A", "B"]
STRATEGIES  = ["none", "smote", "focal"]

DATA_PATH   = None   # None = auto-find the parquet from Step 1
OUTPUT_DIR  = "/kaggle/working"
# -----------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- imports that may need installing ---------------------------------------
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    os.system("pip install -q imbalanced-learn")
    from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("GPU available:", bool(tf.config.list_physical_devices("GPU")))


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def find_parquet():
    import glob
    hits = glob.glob("/kaggle/input/**/ciciot2023_working_set.parquet", recursive=True)
    hits += glob.glob("/kaggle/working/ciciot2023_working_set.parquet")
    if not hits:
        raise SystemExit(
            "Could not find ciciot2023_working_set.parquet. Run Step 1 first, "
            "save it as a Dataset, and attach it as input to this notebook."
        )
    return hits[0]

path = DATA_PATH or find_parquet()
print("Loading:", path)
df = pd.read_parquet(path)

if SAMPLE_ROWS is not None and len(df) > SAMPLE_ROWS:
    # stratified subsample so every class survives the pilot
    df = (df.groupby("family", group_keys=False)
            .apply(lambda g: g.sample(
                n=max(50, int(round(SAMPLE_ROWS * len(g) / len(df)))),
                random_state=42))
            .reset_index(drop=True))
    print(f"PILOT MODE: using {len(df):,} rows")

feature_cols = [c for c in df.columns if c != "family"]
X_all = df[feature_cols].to_numpy(dtype=np.float32)
y_raw = df["family"].to_numpy()

le = LabelEncoder()
y_all = le.fit_transform(y_raw)
CLASS_NAMES = list(le.classes_)
N_FEATURES = X_all.shape[1]
N_CLASSES = len(CLASS_NAMES)

print(f"X: {X_all.shape} | classes ({N_CLASSES}): {CLASS_NAMES}")
print(pd.Series(y_raw).value_counts().to_string())


# =============================================================================
# 2. THE MODEL - a direct transcription of the paper's Algorithm 1
# =============================================================================
def build_model(n_inputs, n_output, loss_fn, lr=LR, momentum=MOMENTUM):
    n_bottleneck = int(round(n_inputs / 2.0))

    visible = keras.Input(shape=(n_inputs, 1))

    # --- left branch: autoencoder ---
    e = layers.Dense(n_inputs)(visible)
    e = layers.BatchNormalization()(e)
    e = layers.LeakyReLU()(e)
    bottleneck = layers.Dense(n_bottleneck)(e)
    d = layers.Dense(n_inputs)(bottleneck)
    d = layers.BatchNormalization()(d)
    d = layers.LeakyReLU()(d)

    # --- right branch: LSTM ---
    lstm = layers.LSTM(n_bottleneck, activation="tanh",
                       return_sequences=True)(visible)
    lstm = layers.Dense(n_inputs)(lstm)

    # --- merge + CNN head ---
    concat = layers.Concatenate()([d, lstm])
    conv = layers.Conv1D(filters=n_bottleneck, kernel_size=2,
                         activation="relu")(concat)
    conv = layers.Flatten()(conv)
    output = layers.Dense(n_output, activation="softmax")(conv)

    model = keras.Model(inputs=visible, outputs=output)
    opt = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])
    return model


def categorical_focal_loss(class_weights, gamma=2.0):
    """Cost-sensitive focal loss - the SMOTE-free alternative."""
    w = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        modulator = tf.pow(1.0 - y_pred, gamma)
        weighted = w * modulator * ce
        return tf.reduce_sum(weighted, axis=-1)

    return loss


# =============================================================================
# 3. ONE EXPERIMENT RUN
# =============================================================================
def run_one(protocol, strategy, seed):
    """protocol: 'A' (as published) or 'B' (leakage-free). Returns a dict."""
    t_start = time.time()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X, y = X_all.copy(), y_all.copy()

    # ---------------- PROTOCOL A: the published order ----------------
    # SMOTE on EVERYTHING -> scale on EVERYTHING -> then split.
    # Synthetic points made from training rows end up in the test set.
    if protocol == "A":
        if strategy == "smote":
            X, y = SMOTE(random_state=seed).fit_resample(X, y)
        scaler = StandardScaler().fit(X)          # <-- sees test rows
        X = scaler.transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)

    # ---------------- PROTOCOL B: the correct order ----------------
    # Split FIRST. Scaler fitted on train only. SMOTE on train only.
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(X_tr)       # <-- train only
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)
        if strategy == "smote":
            X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)

    # ---------------- loss / class weights ----------------
    if strategy == "focal":
        counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
        counts[counts == 0] = 1.0
        cw = counts.sum() / (N_CLASSES * counts)      # inverse frequency
        cw = cw / cw.mean()                           # normalise
        loss_fn = categorical_focal_loss(cw.astype(np.float32))
    else:
        loss_fn = "categorical_crossentropy"

    y_tr_oh = keras.utils.to_categorical(y_tr, N_CLASSES)
    y_te_oh = keras.utils.to_categorical(y_te, N_CLASSES)

    X_tr = X_tr.reshape(-1, N_FEATURES, 1).astype(np.float32)
    X_te = X_te.reshape(-1, N_FEATURES, 1).astype(np.float32)

    # ---------------- train ----------------
    model = build_model(N_FEATURES, N_CLASSES, loss_fn)
    t0 = time.time()
    hist = model.fit(X_tr, y_tr_oh, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     verbose=0, validation_split=0.0)
    train_time = time.time() - t0
    sec_per_epoch = train_time / EPOCHS

    # ---------------- evaluate ----------------
    y_prob = model.predict(X_te, batch_size=2048, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    report = classification_report(
        y_te, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0)

    # ---------------- efficiency ----------------
    n_params = int(model.count_params())
    size_mb = n_params * 4 / 1e6
    probe = X_te[:200]
    _ = model.predict(probe[:1], verbose=0)       # warm-up
    t0 = time.time()
    for i in range(len(probe)):
        model.predict(probe[i:i + 1], verbose=0)
    latency_ms = (time.time() - t0) / len(probe) * 1000

    res = {
        "protocol": protocol,
        "strategy": strategy,
        "seed": seed,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_te, y_pred)),
        "per_class": {c: {"precision": report[c]["precision"],
                          "recall": report[c]["recall"],
                          "f1": report[c]["f1-score"],
                          "support": report[c]["support"]}
                      for c in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "n_params": n_params,
        "model_size_mb": round(size_mb, 3),
        "sec_per_epoch": round(sec_per_epoch, 2),
        "total_train_sec": round(train_time, 1),
        "batch1_latency_ms": round(latency_ms, 3),
        "final_train_loss": float(hist.history["loss"][-1]),
        "wall_sec": round(time.time() - t_start, 1),
    }

    keras.backend.clear_session()
    return res


# =============================================================================
# 4. RUN THE GRID
# =============================================================================
all_results = []
grid = [(p, s, sd) for p in PROTOCOLS for s in STRATEGIES for sd in SEEDS]
print(f"\nRunning {len(grid)} experiments\n" + "=" * 70)

for k, (proto, strat, seed) in enumerate(grid, 1):
    print(f"[{k}/{len(grid)}] protocol={proto}  strategy={strat}  seed={seed} ...",
          flush=True)
    try:
        r = run_one(proto, strat, seed)
        all_results.append(r)
        print(f"    acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
              f"balAcc={r['balanced_accuracy']:.4f}  MCC={r['mcc']:.4f}  "
              f"({r['wall_sec']}s)")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")

    with open(os.path.join(OUTPUT_DIR, "results_raw.json"), "w") as f:
        json.dump(all_results, f, indent=2)

print("\nSaved -> /kaggle/working/results_raw.json")


# =============================================================================
# 5. QUICK LOOK: the headline table
# =============================================================================
if all_results:
    rows = [{k: r[k] for k in
             ["protocol", "strategy", "seed", "accuracy", "balanced_accuracy",
              "macro_f1", "weighted_f1", "mcc", "n_params",
              "sec_per_epoch", "batch1_latency_ms"]}
            for r in all_results]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("HEADLINE: mean over seeds, by protocol x strategy")
    print("=" * 70)
    agg = (summary.groupby(["protocol", "strategy"])
                  [["accuracy", "balanced_accuracy", "macro_f1", "mcc"]]
                  .mean().round(4))
    print(agg.to_string())

    print("\n" + "=" * 70)
    print("THE LEAKAGE GAP  (Protocol A minus Protocol B)")
    print("=" * 70)
    for strat in STRATEGIES:
        a = summary[(summary.protocol == "A") & (summary.strategy == strat)]
        b = summary[(summary.protocol == "B") & (summary.strategy == strat)]
        if len(a) and len(b):
            print(f"  {strat:<7}  accuracy {a.accuracy.mean() - b.accuracy.mean():+.4f}   "
                  f"macro-F1 {a.macro_f1.mean() - b.macro_f1.mean():+.4f}   "
                  f"MCC {a.mcc.mean() - b.mcc.mean():+.4f}")
    print("\nIf the 'smote' row shows a clearly larger gap than the 'none' "
          "row, that difference IS your finding.")
