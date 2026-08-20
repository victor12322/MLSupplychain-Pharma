# Pharmaceutical Supply-Chain Analytics & Late-Delivery Prediction

Descriptive analytics and a leakage-safe machine-learning model on the **USAID SCMS
Delivery History** dataset (10,324 pharmaceutical shipments, 33 columns) — identifying
where deliveries fail and predicting which future shipments will arrive **late**.

## Headline findings

- **A clear cost-vs-reliability trade-off across transport modes.** Air freight costs
  **~6× more per kg than Ocean** ($10.01 vs $1.68) yet is the **most punctual** (9.6% late);
  Ocean is the cheapest but the **least reliable** (17.5% late).
- **Reliability problems concentrate where spend is highest.** The three vendors moving the
  most value/volume (SCMS-from-RDC at $1.09B, Aurobindo at $91.4M, Orgenics) are also among
  the **least punctual** (83–87% on-time) — so the highest-leverage place to intervene is
  also the most expensive.
- **A deployment-realistic model that flags 82% of future late shipments.** A temporally
  validated Random Forest (ROC-AUC **0.807**), tuned to an operating threshold of 0.38,
  catches **300 of 365** future late deliveries — an early-warning signal for stockout risk.
- **The model independently rediscovers the descriptive signal.** Its top predictors —
  route, shipment mode, specific vendors, order size — match Part A's worst routes
  (Burundi, Congo-DRC) and modes (Ocean, Truck), confirming where to target intervention.

## The two parts

**(A) Descriptive supply-chain analytics** — punctuality by route/vendor/product, transport
cost vs time by mode, and a vendor reliability scorecard.
**(B) Late-delivery prediction** — a leakage-safe classifier, evaluated with a temporal
holdout (not a random split) and a tuned decision threshold, so the reported numbers reflect
real forward-looking deployment rather than optimistic in-sample estimates.

---

## Data

`data/SCMS_Delivery_History_Dataset.csv` — one row per shipment. Key fields: scheduled
and actual delivery dates, shipment mode, vendor, destination country, product group,
quantities, prices, weight and freight cost.

**Data-quality note:** `Weight (Kilograms)` and `Freight Cost (USD)` are ~40% text
placeholders (`"Weight Captured Separately"`, `"Freight Included in Commodity Cost"`,
`"Invoiced Separately"`, `"See DN-xxx"`). These are separated from true numbers via
`pd.to_numeric(errors="coerce")` and **never blindly dropped or imputed** — the numeric
subset (~60%) is used for cost stats and a missing-flag is kept for the model.
`PO Sent to Vendor Date` is only 44.5% populated (`"N/A - From RDC"`, `"Date Not Captured"`).

## Method

- **Target:** `delay_days = Delivered to Client Date − Scheduled Delivery Date`;
  **late = delay_days > 0**. Both dates parse for 100% of rows. Base rate: **11.5% late**
  (61.3% exactly on-time, 27.3% early) → a ~1:8 class imbalance.
- All cleaning centralised in `scripts/lib_clean.py` so Part A and Part B share identical
  definitions.
- Reproducible: fixed `random_state=42`, pinned `requirements.txt`.

## Part A — Descriptive findings

### 1. Punctuality
- **11.5%** of shipments (1,186 / 10,324) arrive late.
- Worst routes: **Burundi 38.8% late** (n=98), **Congo-DRC 24.9%** (n=333, and deeply late
  — avg +11.2 days), Mozambique 18.4%, Zambia 15.8%, Zimbabwe 14.3%.
- Worst large vendors: **SCMS-from-RDC 17.2% late** (n=5,404), Aurobindo 14.1%, CIPLA 13.1%.
  Best: ABBVIE 1.2%, S. Buys 1.5%, Chembio 0.9%.
- **ARV drugs (12.5% late) are ~2× less punctual than HIV rapid-test kits (6.5%).**

### 2. Transport cost & time by mode *(cost stats on the ~60% with numeric freight+weight)*
| Mode | Median $/kg | Freight % of value | Late rate |
|------|------------|--------------------|-----------|
| Ocean | $1.68 | 3.5% | 17.5% |
| Truck | $2.50 | 5.9% | 16.1% |
| Air Charter | $4.32 | 8.7% | 11.5% |
| Air | $10.01 | 12.6% | 9.6% |

- **Air costs ~6× more per kg than Ocean** but is the **most punctual**; Ocean is cheapest
  but least punctual → a clear **cost-vs-reliability trade-off**.

### 3. Vendor scorecard (OTIF-style, ≥50 shipments)
On-Time is the reported service level (dataset has no fill-rate field, so true "In-Full"
cannot be computed). Composite = 0.7·on-time-rank + 0.3·cost-rank.
- **Most reliable:** Pharmacy Direct (100% on-time), Hetero Labs (99.3% on $42.9M),
  Mylan (99.4% on $72.2M).
- **Least reliable at scale:** SCMS-from-RDC (82.8% on-time, $1.09B, 5,404 shipments),
  Orgenics (87.0% + most expensive freight $17.2/kg), Aurobindo (85.9% on $91.4M).
- **The highest-value/volume vendors are also the least punctual** — reliability problems
  concentrate exactly where spend is highest.

## Part B — Late-delivery prediction

**Leakage control:** features restricted to what is known *at order time*. Excluded:
actual/recorded delivery dates, `delay_days`, the target, and **Freight Cost** (invoiced
post-shipment). Features used — categorical: Shipment Mode, Country, Vendor, Product Group,
Sub Classification, Managed By, Fulfill Via, Vendor INCO Term, First Line Designation;
numeric: Line Item Quantity/Value, Unit Price, Pack Price, Weight (+missing flag), planned
lead time = Scheduled − PO-Sent (+missing flag).

**Pipeline:** `ColumnTransformer` (one-hot categoricals with `min_frequency=20` +
median-impute & scale numerics) → Logistic Regression (baseline) and Random Forest
(stronger). Both use `class_weight='balanced'` (no synthetic data → importances stay
interpretable). 75/25 stratified split.

Evaluated two ways (`B1_model.py` random split, `B2_temporal_split.py` temporal split).
Because this is a forward-looking predictor, the **temporal split is the primary,
deployment-realistic estimate**; the random split is reported only to quantify look-ahead
inflation. Temporal split sorts by `Scheduled Delivery Date` (order-time proxy) and trains on
the earlier 75% (2006→2013) / tests on the later 25% (2013→2015, no shuffle); the late rate
rises 10.6%→14.1% across that boundary (real non-stationarity a random split would hide).

**Primary results — temporal split (deployment-realistic):**
| Model | Accuracy | Precision (late) | Recall (late) | F1 | ROC-AUC |
|-------|----------|------------------|---------------|----|---------|
| Logistic Regression | 0.550 | 0.209 | 0.786 | 0.331 | 0.632 |
| **Random Forest** | 0.782 | 0.328 | 0.515 | **0.400** | **0.807** |

**Random-split results (optimistic — for comparison only):**
| Model | Accuracy | Precision (late) | Recall (late) | F1 | ROC-AUC |
|-------|----------|------------------|---------------|----|---------|
| Logistic Regression | 0.609 | 0.200 | 0.801 | 0.320 | 0.775 |
| Random Forest | 0.771 | 0.287 | 0.672 | 0.402 | 0.826 |

**How much the random split inflated things:** LogReg ROC-AUC collapses **0.775 → 0.632
(−0.144)** — its apparent skill was largely look-ahead artifact; on genuinely future shipments
it is only modestly better than chance. Random Forest is robust on ranking (**0.826 → 0.807,
−0.019**) but its recall drops **0.672 → 0.515 (−0.157)**: as the late rate rises over time, the
fixed 0.50 threshold catches only ~half of future late shipments. **RF generalises across time;
LogReg mostly memorised period-specific patterns.**

**Why not accuracy alone:** predicting "on-time" for everything scores 88.5% accuracy while
catching zero late shipments. Under imbalance, **recall** (share of late shipments caught),
**precision** (share of alarms that are real), **F1** and **ROC-AUC** (threshold-independent
ranking) are what matter.

**Tuned operating point (`B3_threshold_tuning.py`).** The default 0.50 threshold misses ~half
of future late shipments. The threshold is tuned on a **validation slice carved from the latest
training rows** (never the test set): F1-optimal = **0.38**, then applied once to the untouched
temporal test set (RF refit on the full train pool):

| Operating point | Precision | Recall | F1 | Late caught (of 365) |
|-----------------|-----------|--------|----|----------------------|
| Default 0.50 | 0.328 | 0.515 | 0.400 | 188 |
| **Tuned 0.38** | 0.322 | **0.822** | **0.463** | **300** |

Lowering the threshold to 0.38 nearly **doubles recall (0.515 → 0.822)** at essentially no
precision cost (0.328 → 0.322), trading more false alarms (386 → 632). ROC-AUC is unchanged
(0.807) — the threshold moves the operating point, not the ranking. **Recommended deployment
operating point: RF, temporal, threshold 0.38.**

**Feature importance (ML ↔ descriptive):** permutation importance ranks **Country and
Shipment Mode highest**, matching Part A's worst routes/modes; RF impurity importance is led
by **order-size numerics** (pack price, quantity, value, weight); LogReg coefficients flag
the same risky vendors (Orgenics, Bio-Rad) and routes (Burundi, Congo-DRC) found in Part A.

**Business finding:** late risk is driven by **route + shipment mode + specific vendors +
order size** — intervention should target the worst route/vendor combinations, not spread
evenly.

## Limitations
- Cost analysis uses only the ~60% of rows with numeric freight *and* weight; conclusions
  don't extend to placeholder rows.
- "In-Full" is unavailable → OTIF reduced to On-Time.
- Planned lead time is missing for 55% of rows (no PO date) → imputed with a missing flag.
- The model predicts a binary late/on-time flag, not the magnitude of delay.
- **Split design.** `B1_model.py` uses a stratified *random* shuffle, which has look-ahead
  bias for a forward-looking predictor. `B2_temporal_split.py` addresses this with a temporal
  holdout (train on earlier period, test on later), and its numbers are treated as the primary
  estimate above. The temporal split still uses `Scheduled Delivery Date` as an order-time
  proxy because `PO Sent to Vendor Date` (true order time) is 55% missing.
- The primary tables above report the RF at the default 0.50 threshold; `B3_threshold_tuning.py`
  tunes it on a validation slice to 0.38 (recall 0.515 → 0.822), which is the recommended
  deployment operating point.
- **Threshold objective.** The operating threshold is tuned for F1, which weights precision and
  recall equally. In a stockout-sensitive setting where a missed delay costs more than a false
  alarm, a cost-sensitive threshold using an explicit FN:FP cost ratio would likely sit even
  lower than 0.38 — a natural extension not pursued here to avoid an unjustified cost assumption.

## Reproduce
```bash
pip install -r requirements.txt
cd scripts
python 00_inspect.py            # data-quality report
python 01_delay_distribution.py # target/threshold justification
python A1_punctuality.py        # Part A.1
python A2_freight_by_mode.py    # Part A.2
python A3_vendor_scorecard.py   # Part A.3
python B1_model.py              # Part B: models, metrics, feature importance (random split)
python B2_temporal_split.py     # Part B: temporal vs random split comparison
python B3_threshold_tuning.py   # Part B: threshold tuning on temporal RF
```
Figures → `figures/`, tables → `outputs/`.

## Repo layout
```
data/      raw CSV
scripts/   lib_clean.py + numbered analysis/ML scripts
figures/   PNG charts
outputs/   CSV result tables
```
