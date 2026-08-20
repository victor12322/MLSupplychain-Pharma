"""PART B: Predict whether a shipment will be LATE (delay > 0), using only
features known AT ORDER TIME (no leakage).

Pipeline: ColumnTransformer (one-hot categoricals + impute numerics)
-> Logistic Regression (baseline) and Random Forest (stronger).
Both use class_weight='balanced' for the ~1:8 imbalance.
Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
Feature importance: LogReg coefficients + RF impurity importance +
permutation importance (model-agnostic, robust)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay, classification_report)
from sklearn.inspection import permutation_importance

from lib_clean import load_clean

FIG = r"C:\Users\hp\pharma-supply-chain\figures"
OUT = r"C:\Users\hp\pharma-supply-chain\outputs"
SEED = 42

df = load_clean()

# ---- planned lead time: known at order (both dates precede delivery) ----
df["po_to_sched_days"] = (df["Scheduled Delivery Date"]
                          - df["PO Sent to Vendor Date"]).dt.days
df["po_date_missing"] = df["po_to_sched_days"].isna().astype(int)
# weight: NaN where the source was a text placeholder -> keep flag, impute later
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y)
print(f"Train {X_train.shape}  Test {X_test.shape}")
print(f"Late rate  train {y_train.mean():.3f}  test {y_test.mean():.3f}")

# ---- preprocessing ----
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
])

models = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=SEED),
    "rf": RandomForestClassifier(n_estimators=400, max_depth=None,
                                  min_samples_leaf=5, n_jobs=-1,
                                  class_weight="balanced_subsample",
                                  random_state=SEED),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "pipe": pipe, "proba": proba, "pred": pred,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cm": confusion_matrix(y_test, pred),
    }
    r = results[name]
    print(f"\n===== {name.upper()} (threshold 0.50) =====")
    print(f"accuracy {r['accuracy']:.3f} | precision {r['precision']:.3f} | "
          f"recall {r['recall']:.3f} | F1 {r['f1']:.3f} | ROC-AUC {r['roc_auc']:.3f}")
    print("confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(r["cm"])
    print(classification_report(y_test, pred, digits=3,
                                target_names=["on-time", "late"]))

# ---- metric summary table ----
summ = pd.DataFrame({n: {k: results[n][k] for k in
                    ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                    for n in results}).T.round(3)
summ.to_csv(f"{OUT}\\B1_metrics.csv")
print("\nSUMMARY:\n", summ.to_string())

# ---- ROC curve figure ----
fig, ax = plt.subplots(figsize=(6, 6))
for name in results:
    RocCurveDisplay.from_predictions(y_test, results[name]["proba"],
                                     name=f"{name} (AUC={results[name]['roc_auc']:.3f})",
                                     ax=ax)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC — late-shipment prediction")
fig.tight_layout()
fig.savefig(f"{FIG}\\B1_roc.png", dpi=120)

# ---- confusion matrix figure (RF) ----
fig, ax = plt.subplots(figsize=(4.5, 4))
cm = results["rf"]["cm"]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["on-time", "late"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["on-time", "late"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Random Forest confusion matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.tight_layout()
fig.savefig(f"{FIG}\\B1_confusion_rf.png", dpi=120)

# ================= FEATURE IMPORTANCE =================
# encoded feature names
ohe = results["rf"]["pipe"].named_steps["pre"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(CAT)
feat_names = np.concatenate([cat_names, NUM])

# RF impurity importance
rf = results["rf"]["pipe"].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\nTOP 15 RANDOM FOREST FEATURE IMPORTANCES:")
print(imp.head(15).round(4).to_string())
imp.head(40).to_csv(f"{OUT}\\B1_rf_importance.csv")

# LogReg coefficients (log-odds); positive = pushes toward LATE
lr = results["logreg"]["pipe"].named_steps["clf"]
coef = pd.Series(lr.coef_[0], index=feat_names).sort_values()
print("\nLOGREG: 8 strongest 'pushes LATE' (positive coef):")
print(coef.tail(8).round(3).to_string())
print("\nLOGREG: 8 strongest 'pushes ON-TIME' (negative coef):")
print(coef.head(8).round(3).to_string())
coef.to_csv(f"{OUT}\\B1_logreg_coef.csv")

# Permutation importance (model-agnostic) on RF, aggregated to raw columns
perm = permutation_importance(results["rf"]["pipe"], X_test, y_test,
                              scoring="roc_auc", n_repeats=8,
                              random_state=SEED, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\nPERMUTATION IMPORTANCE (drop in ROC-AUC, by RAW column):")
print(perm_s.round(4).to_string())
perm_s.to_csv(f"{OUT}\\B1_permutation_importance.csv")

# ---- feature importance figure (top 15 RF) ----
fig, ax = plt.subplots(figsize=(9, 6))
top = imp.head(15).sort_values()
ax.barh(top.index, top.values, color="#16a085")
ax.set_title("Top 15 features — Random Forest importance")
fig.tight_layout()
fig.savefig(f"{FIG}\\B1_feature_importance.png", dpi=120)
print(f"\nSaved figures & CSVs to {FIG} and {OUT}")
