"""Shared loading & cleaning for the SCMS dataset.

Central place so Part A and Part B use identical definitions:
- date parsing
- delay (days) and the binary target is_late (delay > 0)
- numeric extraction of Weight and Freight Cost, keeping a flag for the
  ~40% of rows where those fields are text placeholders (not real numbers).
"""
import pandas as pd
import numpy as np

CSV = r"C:\Users\hp\pharma-supply-chain\data\SCMS_Delivery_History_Dataset.csv"

DATE_COLS = ["PQ First Sent to Client Date", "PO Sent to Vendor Date",
             "Scheduled Delivery Date", "Delivered to Client Date",
             "Delivery Recorded Date"]


def _to_num(series):
    """Coerce a column to float; text placeholders become NaN. Returns
    (numeric_series, is_numeric_bool_mask)."""
    num = pd.to_numeric(series, errors="coerce")
    return num, num.notna()


def load_clean():
    df = pd.read_csv(CSV, dtype=str)

    # --- dates: format='mixed' silences the per-element inference warning ---
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")

    # --- target: delay in days, then binary late flag (threshold = 0) ---
    df["delay_days"] = (df["Delivered to Client Date"]
                        - df["Scheduled Delivery Date"]).dt.days
    df["is_late"] = (df["delay_days"] > 0).astype(int)

    # --- numeric extraction with placeholder flags ---
    df["weight_kg"], df["weight_is_num"] = _to_num(df["Weight (Kilograms)"])
    df["freight_usd"], df["freight_is_num"] = _to_num(df["Freight Cost (USD)"])

    # --- other numerics that ARE clean numbers in this dataset ---
    for c in ["Line Item Quantity", "Line Item Value", "Pack Price",
              "Unit Price", "Unit of Measure (Per Pack)",
              "Line Item Insurance (USD)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalise Shipment Mode missing values to an explicit label
    df["Shipment Mode"] = df["Shipment Mode"].fillna("Unknown")

    return df


if __name__ == "__main__":
    d = load_clean()
    print("rows:", len(d))
    print("late rate:", f"{d['is_late'].mean():.1%}")
    print("weight numeric:", f"{d['weight_is_num'].mean():.1%}")
    print("freight numeric:", f"{d['freight_is_num'].mean():.1%}")
