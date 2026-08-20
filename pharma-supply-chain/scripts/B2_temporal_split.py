"""PART B (extension): temporal vs random split.

Same features, same models (LogReg + RandomForest, class_weight='balanced')
as B1, evaluated two ways:
  - RANDOM   : stratified shuffle (look-ahead bias, matches B1)
  - TEMPORAL : sort by Scheduled Delivery Date (100%-populated proxy for
               order time), earlier 75% train / later 25% test, NO shuffle.

Prints a side-by-side metric table for both models and saves it to CSV, so
we can quantify how much the random split inflated the numbers."""
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from lib_clean import load_clean

OUT = r"C:\Users\hp\pharma-supply-chain\outputs"
SEED = 42

df = load_clean()

# ---- identical feature engineering to B1 ----
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
sched = df["Scheduled Delivery Date"]


def make_pipes():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
    ])
    return {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED))]),
        "rf": Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1,
            class_weight="balanced_subsample", random_state=SEED))]),
    }


def evaluate(Xtr, Xte, ytr, yte):
    out = {}
    for name, pipe in make_pipes().items():
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[name] = {
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
        }
    return out


# ---- RANDOM split (matches B1) ----
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y)
random_res = evaluate(Xtr_r, Xte_r, ytr_r, yte_r)

# ---- TEMPORAL split: order by scheduled date, no shuffle ----
order = np.argsort(sched.values, kind="mergesort")  # stable, chronological
cut = int(len(order) * 0.75)
tr_idx, te_idx = order[:cut], order[cut:]
Xtr_t, Xte_t = X.iloc[tr_idx], X.iloc[te_idx]
ytr_t, yte_t = y.iloc[tr_idx], y.iloc[te_idx]
temporal_res = evaluate(Xtr_t, Xte_t, ytr_t, yte_t)

print("TEMPORAL split boundaries:")
print(f"  train dates: {sched.iloc[tr_idx].min().date()} -> {sched.iloc[tr_idx].max().date()}  (n={len(tr_idx)})")
print(f"  test  dates: {sched.iloc[te_idx].min().date()} -> {sched.iloc[te_idx].max().date()}  (n={len(te_idx)})")
print(f"  late rate  train {ytr_t.mean():.3f}  test {yte_t.mean():.3f}")

# ---- side-by-side table ----
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []
for model in ["logreg", "rf"]:
    for split, res in [("random", random_res), ("temporal", temporal_res)]:
        rec = {"model": model, "split": split}
        rec.update({m: round(res[model][m], 3) for m in metrics})
        rows.append(rec)
comp = pd.DataFrame(rows)

# add delta (temporal - random) rows per model to quantify inflation
delta_rows = []
for model in ["logreg", "rf"]:
    d = {"model": model, "split": "delta (temporal-random)"}
    d.update({m: round(temporal_res[model][m] - random_res[model][m], 3)
              for m in metrics})
    delta_rows.append(d)
comp = pd.concat([comp, pd.DataFrame(delta_rows)], ignore_index=True)

pd.set_option("display.width", 160)
print("\n=== RANDOM vs TEMPORAL split — metric comparison ===\n")
print(comp.to_string(index=False))

comp.to_csv(f"{OUT}\\B2_split_comparison.csv", index=False)
print(f"\nSaved -> {OUT}\\B2_split_comparison.csv")
