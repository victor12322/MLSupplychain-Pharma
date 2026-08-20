"""Raw inspection of the SCMS dataset: shape, columns, date parseability,
and the share of non-numeric text in Weight / Freight Cost."""
import pandas as pd

CSV = r"C:\Users\hp\pharma-supply-chain\data\SCMS_Delivery_History_Dataset.csv"

df = pd.read_csv(CSV, dtype=str)  # read everything as string first, decide conversions explicitly
print("SHAPE:", df.shape)
print("\nCOLUMNS:")
for c in df.columns:
    print(" -", repr(c))

date_cols = ["Scheduled Delivery Date", "Delivered to Client Date",
             "PO Sent to Vendor Date", "Delivery Recorded Date"]

print("\nDATE COLUMN PARSEABILITY:")
for c in date_cols:
    s = df[c]
    parsed = pd.to_datetime(s, errors="coerce")
    n = len(s)
    ok = parsed.notna().sum()
    # show the distinct non-parseable string values (likely placeholders)
    bad_vals = s[parsed.isna() & s.notna()].value_counts().head(5)
    print(f"\n  {c}: {ok}/{n} parse OK ({ok/n:.1%})")
    if len(bad_vals):
        print("    non-parseable examples:")
        for v, cnt in bad_vals.items():
            print(f"      {cnt:>5}  {v!r}")

print("\nWEIGHT / FREIGHT NUMERIC vs TEXT:")
for c in ["Weight (Kilograms)", "Freight Cost (USD)"]:
    s = df[c]
    num = pd.to_numeric(s, errors="coerce")
    n = len(s)
    numeric_ok = num.notna().sum()
    print(f"\n  {c}: {numeric_ok}/{n} numeric ({numeric_ok/n:.1%})")
    text_vals = s[num.isna() & s.notna()].value_counts().head(8)
    print("    top non-numeric values:")
    for v, cnt in text_vals.items():
        print(f"      {cnt:>5}  {v!r}")

print("\nSHIPMENT MODE:")
print(df["Shipment Mode"].value_counts(dropna=False))
