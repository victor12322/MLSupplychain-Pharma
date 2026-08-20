"""Compute delivery delay = Delivered to Client Date - Scheduled Delivery Date
and describe its distribution to inform the late/on-time threshold."""
import pandas as pd
import numpy as np

CSV = r"C:\Users\hp\pharma-supply-chain\data\SCMS_Delivery_History_Dataset.csv"
df = pd.read_csv(CSV, dtype=str)

sched = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce")
deliv = pd.to_datetime(df["Delivered to Client Date"], errors="coerce")

delay = (deliv - sched).dt.days
valid = delay.dropna()

print(f"Rows with a computable delay: {valid.shape[0]} / {len(df)}")
print("\nDELAY (days) DESCRIBE:")
print(valid.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95, .99]).round(2))

print("\nSHARE LATE AT DIFFERENT THRESHOLDS (delay > N days):")
for thr in [0, 1, 2, 3, 5, 7, 14, 30]:
    share = (valid > thr).mean()
    print(f"  > {thr:>2} days late : {share:6.1%}   (n={int((valid>thr).sum())})")

print("\nEXACTLY ON TIME (delay == 0):", f"{(valid==0).mean():.1%}")
print("EARLY (delay < 0):          ", f"{(valid<0).mean():.1%}")
print("LATE (delay > 0):           ", f"{(valid>0).mean():.1%}")

print("\nDELAY RANGE:", int(valid.min()), "to", int(valid.max()), "days")
