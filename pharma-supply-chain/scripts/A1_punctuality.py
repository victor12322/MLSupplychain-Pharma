"""PART A - Phase 1: Punctuality.
Which shipments arrive late, broken down by route (country), vendor, and
product group. Writes a figure and a findings text block for the README."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib_clean import load_clean

FIG = r"C:\Users\hp\pharma-supply-chain\figures"
OUT = r"C:\Users\hp\pharma-supply-chain\outputs"

df = load_clean()
n = len(df)
overall = df["is_late"].mean()
print(f"Overall late rate: {overall:.1%} ({df['is_late'].sum()} of {n})")


def breakdown(col, min_n=50, top=12):
    """Late rate per category, keeping only groups with >= min_n shipments
    (rare groups give unstable rates)."""
    g = df.groupby(col).agg(shipments=("is_late", "size"),
                            late=("is_late", "sum"),
                            avg_delay=("delay_days", "mean"))
    g["late_rate"] = g["late"] / g["shipments"]
    g = g[g["shipments"] >= min_n].sort_values("late_rate", ascending=False)
    return g.head(top)


by_country = breakdown("Country")
by_vendor = breakdown("Vendor")
by_pgroup = breakdown("Product Group", min_n=100)

print("\n--- Late rate by COUNTRY (>=50 shipments) ---")
print(by_country[["shipments", "late_rate", "avg_delay"]].round(3).to_string())
print("\n--- Late rate by VENDOR (>=50 shipments) ---")
print(by_vendor[["shipments", "late_rate", "avg_delay"]].round(3).to_string())
print("\n--- Late rate by PRODUCT GROUP (>=100 shipments) ---")
print(by_pgroup[["shipments", "late_rate", "avg_delay"]].round(3).to_string())

# ---- figure: top-10 countries by late rate ----
fig, ax = plt.subplots(figsize=(9, 5))
top10 = by_country.head(10).sort_values("late_rate")
ax.barh(top10.index, top10["late_rate"] * 100, color="#c0392b")
ax.axvline(overall * 100, color="black", ls="--", lw=1,
           label=f"overall {overall:.1%}")
ax.set_xlabel("Late rate (%)")
ax.set_title("Top 10 destination countries by late-delivery rate (≥50 shipments)")
for i, (v, s) in enumerate(zip(top10["late_rate"] * 100, top10["shipments"])):
    ax.text(v + 0.3, i, f"{v:.0f}%  (n={s})", va="center", fontsize=8)
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG}\\A1_late_rate_by_country.png", dpi=120)
print(f"\nSaved figure -> {FIG}\\A1_late_rate_by_country.png")

# ---- persist tables (CSV, utf-8) ----
by_country.to_csv(f"{OUT}\\A1_late_by_country.csv", encoding="utf-8-sig")
by_vendor.to_csv(f"{OUT}\\A1_late_by_vendor.csv", encoding="utf-8-sig")
by_pgroup.to_csv(f"{OUT}\\A1_late_by_product_group.csv", encoding="utf-8-sig")
print(f"Saved tables  -> {OUT}\\A1_late_by_*.csv")
