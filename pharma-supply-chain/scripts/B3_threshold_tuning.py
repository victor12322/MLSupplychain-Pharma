"""PART B (extension): decision-threshold tuning for the TEMPORAL Random Forest.

Temporal design (no shuffle, sorted by Scheduled Delivery Date):
  [-------- train pool (earliest 75%) --------][--- test (latest 25%) ---]
  train pool is split again in time:
  [--- fit (earliest 80% of pool) ---][--- validation (latest 20% of pool) ---]

Threshold is chosen on the VALIDATION slice (latest training rows) by maximising
F1 — never on the test set. The RF is then refit on the FULL train pool and the
chosen threshold is applied once to the untouched temporal test set for the final
precision/recall/F1. Same features and class_weight='balanced' as B1/B2."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)

from lib_clean import load_clean

FIG = r"C:\Users\hp\pharma-supply-chain\figures"
OUT = r"C:\Users\hp\pharma-supply-chain\outputs"
SEED = 42

df = load_clean()
df["po_to_sched_days"] = (df["Scheduled Delivery Date"]
                          - df["PO Sent to Vendor Date"]).dt.days
df["po_date_missing"] = df["po_to_sched_days"].isna().astype(int)
df["weight_missing"] = (~df["weight_is_num"]).astype(int)

CAT = ["Shipment Mode", "Country", "Vendor", "Product Group",
       "Sub Classification", "Managed By", "Fulfill Via",
       "Vendor INCO Term", "First Line Designation"]
NUM = ["Line Item Quantity", "Line Item Value", "Unit Price", "Pack Price",
       "weight_kg", "po_to_sched_days", "weight_missing", "po_date_missing"]

X = df[CAT + NUM].copy()
for c in CAT:
    X[c] = X[c].fillna("Missing").astype(str)
y = df["is_late"]

# ---- temporal ordering ----
order = np.argsort(df["Scheduled Delivery Date"].values, kind="mergesort")
n = len(order)
cut_test = int(n * 0.75)                 # train pool / test boundary
pool_idx, test_idx = order[:cut_test], order[cut_test:]
cut_val = int(len(pool_idx) * 0.80)      # fit / validation boundary inside pool
fit_idx, val_idx = pool_idx[:cut_val], pool_idx[cut_val:]

sched = df["Scheduled Delivery Date"]
print("Temporal slices (by Scheduled Delivery Date):")
for label, idx in [("fit ", fit_idx), ("val ", val_idx), ("test", test_idx)]:
    print(f"  {label}: {sched.iloc[idx].min().date()} -> {sched.iloc[idx].max().date()}"
          f"  n={len(idx):>5}  late={y.iloc[idx].mean():.3f}")


def build_rf():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
    ])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=400, min_samples_leaf=5, n_jobs=-1,
        class_weight="balanced_subsample", random_state=SEED))])


# ---- 1) fit on the earliest part of the pool, get validation probabilities ----
rf_val = build_rf().fit(X.iloc[fit_idx], y.iloc[fit_idx])
val_proba = rf_val.predict_proba(X.iloc[val_idx])[:, 1]
y_val = y.iloc[val_idx].values

# ---- 2) sweep threshold on VALIDATION ----
thresholds = np.round(np.arange(0.10, 0.9001, 0.02), 2)
rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)
    rows.append({"threshold": t,
                 "precision": precision_score(y_val, pred, zero_division=0),
                 "recall": recall_score(y_val, pred, zero_division=0),
                 "f1": f1_score(y_val, pred, zero_division=0)})
sweep = pd.DataFrame(rows)
best = sweep.loc[sweep["f1"].idxmax()]
best_t = float(best["threshold"])
print(f"\nF1-optimal threshold on validation: {best_t:.2f}"
      f"  (val P={best['precision']:.3f} R={best['recall']:.3f} F1={best['f1']:.3f})")
sweep.to_csv(f"{OUT}\\B3_threshold_sweep.csv", index=False)

# ---- 3) refit RF on FULL train pool, evaluate ONCE on untouched test ----
rf_final = build_rf().fit(X.iloc[pool_idx], y.iloc[pool_idx])
test_proba = rf_final.predict_proba(X.iloc[test_idx])[:, 1]
y_test = y.iloc[test_idx].values


def report(tag, t):
    pred = (test_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"operating_point": tag, "threshold": t,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


final = pd.DataFrame([report("default 0.50", 0.50),
                      report(f"tuned {best_t:.2f}", best_t)])
final["roc_auc_test"] = round(roc_auc_score(y_test, test_proba), 3)
pd.set_option("display.width", 160)
print("\n=== FINAL — untouched temporal TEST set (RF refit on full pool) ===\n")
print(final.round(3).to_string(index=False))
final.round(4).to_csv(f"{OUT}\\B3_final_operating_point.csv", index=False)

# ---- 4) figure: P/R/F1 vs threshold on validation ----
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(sweep["threshold"], sweep["precision"], label="precision", color="#2c7fb8")
ax.plot(sweep["threshold"], sweep["recall"], label="recall", color="#c0392b")
ax.plot(sweep["threshold"], sweep["f1"], label="F1", color="#16a085", lw=2.5)
ax.axvline(best_t, color="black", ls="--", lw=1.2,
           label=f"F1-optimal = {best_t:.2f}")
ax.axvline(0.50, color="grey", ls=":", lw=1, label="default 0.50")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title("RF threshold sweep on validation slice (latest training rows)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG}\\B3_threshold_tuning.png", dpi=120)
print(f"\nSaved figure -> {FIG}\\B3_threshold_tuning.png")
print(f"Saved CSVs   -> {OUT}\\B3_threshold_sweep.csv , B3_final_operating_point.csv")
