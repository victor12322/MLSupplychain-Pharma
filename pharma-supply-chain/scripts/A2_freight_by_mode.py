"""PART A - Phase 2: Transport cost & time by shipment mode.
Only the ~60% of rows with a numeric Freight Cost are used for cost stats;
that limitation is stated explicitly. Cost is normalised per kg and per
line-item value to compare modes fairly (Air moves lighter, urgent loads)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib_clean import load_clean

FIG = r"C:\Users\hp\pharma-supply-chain\figures"
OUT = r"C:\Users\hp\pharma-supply-chain\outputs"

df = load_clean()

# transit time proxy: scheduled - PO sent (known before delivery). Only 44.5%
# of PO dates exist, so we report it separately and don't over-claim.
df["po_to_sched_days"] = (df["Scheduled Delivery Date"]
                          - df["PO Sent to Vendor Date"]).dt.days

modes = ["Air", "Truck", "Air Charter", "Ocean"]
d = df[df["Shipment Mode"].isin(modes)].copy()

# ---- coverage statement ----
tot = len(d)
freight_ok = d["freight_is_num"].sum()
weight_ok = d["weight_is_num"].sum()
print(f"Rows in the 4 named modes: {tot}")
print(f"  with numeric freight: {freight_ok} ({freight_ok/tot:.1%})")
print(f"  with numeric weight : {weight_ok} ({weight_ok/tot:.1%})")

# ---- cost sub-frame: need numeric freight AND weight for $/kg ----
cost = d[d["freight_is_num"] & d["weight_is_num"] & (d["weight_kg"] > 0)].copy()
cost["freight_per_kg"] = cost["freight_usd"] / cost["weight_kg"]
# freight as a share of shipped value (freight numeric + positive line value)
cost = cost[cost["Line Item Value"] > 0]
cost["freight_pct_value"] = cost["freight_usd"] / cost["Line Item Value"]

def trimmed(s):
    """median is robust; also report mean after trimming 1st/99th pct."""
    lo, hi = s.quantile([.01, .99])
    return s[(s >= lo) & (s <= hi)].mean()

print("\n--- COST per shipment mode (numeric freight & weight only) ---")
rows = []
for m in modes:
    sub = cost[cost["Shipment Mode"] == m]
    rows.append({
        "mode": m,
        "n_cost": len(sub),
        "median_freight_usd": sub["freight_usd"].median(),
        "median_usd_per_kg": sub["freight_per_kg"].median(),
        "median_freight_pct_value": sub["freight_pct_value"].median(),
        "late_rate": d[d["Shipment Mode"] == m]["is_late"].mean(),
        "median_po_to_sched_days": d[d["Shipment Mode"] == m]["po_to_sched_days"].median(),
    })
cost_tbl = pd.DataFrame(rows).set_index("mode").round(3)
print(cost_tbl.to_string())
cost_tbl.to_csv(f"{OUT}\\A2_cost_by_mode.csv", encoding="utf-8-sig")

# ---- figure: median $/kg by mode ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
order = cost_tbl.sort_values("median_usd_per_kg").index
axes[0].bar(order, cost_tbl.loc[order, "median_usd_per_kg"], color="#2c7fb8")
axes[0].set_ylabel("Median freight USD / kg")
axes[0].set_title("Unit shipping cost by mode")
for i, m in enumerate(order):
    axes[0].text(i, cost_tbl.loc[m, "median_usd_per_kg"],
                 f"${cost_tbl.loc[m,'median_usd_per_kg']:.1f}",
                 ha="center", va="bottom", fontsize=9)

axes[1].bar(order, cost_tbl.loc[order, "late_rate"] * 100, color="#c0392b")
axes[1].set_ylabel("Late rate (%)")
axes[1].set_title("Late rate by mode")
for i, m in enumerate(order):
    axes[1].text(i, cost_tbl.loc[m, "late_rate"] * 100,
                 f"{cost_tbl.loc[m,'late_rate']*100:.1f}%",
                 ha="center", va="bottom", fontsize=9)
fig.suptitle("Cost vs punctuality trade-off by shipment mode", fontsize=13)
fig.tight_layout()
fig.savefig(f"{FIG}\\A2_cost_vs_late_by_mode.png", dpi=120)
print(f"\nSaved figure -> {FIG}\\A2_cost_vs_late_by_mode.png")
print(f"Saved table  -> {OUT}\\A2_cost_by_mode.csv")
