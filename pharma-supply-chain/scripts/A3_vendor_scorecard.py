"""PART A - Phase 3: Vendor reliability scorecard (OTIF-style).
Combines punctuality, cost efficiency and volume into one ranked table.
OTIF here = On-Time (delay <= 0) AND In-Full proxy. The dataset has no
short-ship field, so 'In Full' is proxied by 'shipment delivered at all'
(all rows are delivered); we therefore report On-Time as the core service
metric and label it honestly rather than inventing a fill rate."""
import pandas as pd
import numpy as np
from lib_clean import load_clean

OUT = r"C:\Users\hp\pharma-supply-chain\outputs"

df = load_clean()

MIN_SHIPMENTS = 50  # ignore vendors too small for a stable rate

g = df.groupby("Vendor")
score = pd.DataFrame({
    "shipments": g.size(),
    "total_value_usd": g["Line Item Value"].sum(),
    "on_time_rate": 1 - g["is_late"].mean(),          # core service level
    "avg_delay_days": g["delay_days"].mean(),
    "p90_delay_days": g["delay_days"].quantile(0.90),  # tail risk
})

# cost efficiency: median freight per kg on rows where both are numeric
cost_src = df[df["freight_is_num"] & df["weight_is_num"] & (df["weight_kg"] > 0)].copy()
cost_src["fpk"] = cost_src["freight_usd"] / cost_src["weight_kg"]
score["median_usd_per_kg"] = cost_src.groupby("Vendor")["fpk"].median()

score = score[score["shipments"] >= MIN_SHIPMENTS].copy()

# ---- composite score: rank-normalise on-time (higher better) and
# cost (lower better); weight service 0.7, cost 0.3. Cost missing -> service only.
score["ot_rank"] = score["on_time_rate"].rank(pct=True)
score["cost_rank"] = (1 - score["median_usd_per_kg"].rank(pct=True))  # cheaper = better
score["composite"] = np.where(
    score["cost_rank"].notna(),
    0.7 * score["ot_rank"] + 0.3 * score["cost_rank"],
    score["ot_rank"])
score = score.sort_values("composite", ascending=False)

show = score[["shipments", "total_value_usd", "on_time_rate",
              "avg_delay_days", "p90_delay_days", "median_usd_per_kg",
              "composite"]].round(3)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

print("=== VENDOR SCORECARD (>=50 shipments, ranked by composite) ===\n")
print("TOP 8 (most reliable):")
print(show.head(8).to_string())
print("\nBOTTOM 8 (least reliable):")
print(show.tail(8).to_string())

score.to_csv(f"{OUT}\\A3_vendor_scorecard.csv", encoding="utf-8-sig")
print(f"\nSaved -> {OUT}\\A3_vendor_scorecard.csv  ({len(score)} vendors)")
