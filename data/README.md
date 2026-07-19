# Data

**The original HEDNO datathon data is NOT included in this repository.** It was
provided under the datathon's terms and contains real customer/meter information,
so it cannot be redistributed.

## Files the pipeline expects (in the repository root)

| File | Produced by | Consumed by |
|---|---|---|
| `clean_data.pkl` | provided (pre-cleaned train set) | `Main.ipynb`, `HDBSCAN.ipynb` |
| `clean_data_test.pkl` | provided (test set, no labels used) | `Main.ipynb` |
| `clustered_data.pkl` | `HDBSCAN.ipynb` | `Imputation.ipynb` |
| `imputated_clustered_data.pkl` | `Imputation.ipynb` | `Sensor_Predictions.ipynb` |
| `sensors_df.pkl`, `q_values.pkl` | `Sensor_Predictions.ipynb` / `Main.ipynb` | analysis |
| `Models/` (`processor.pkl`, `detectors.pkl`, `one_hot_columns.pkl`, `hdbscan_model.pkl`) | notebooks | reuse/inference |

## Expected schema of `clean_data.pkl` (pandas DataFrame)

| Column | Type | Meaning |
|---|---|---|
| `ACCT_NBR` | str | anonymized account id (dropped before modeling) |
| `SUCCESSOR` | int | contract successor index (id-like, dropped) |
| `MS_METER_NBR` | str | meter id (dropped) |
| `BS_RATE` | str/int | billing rate code |
| `time_series` | list[float] | consumption measurements; **0 encodes a missing reading** |
| `label` | int {0,1} | target: 1 = confirmed power theft (rare class) |
| `XRHSH` | float | usage type code |
| `VOLTAGE` | str | e.g. `LOW` |
| `PARNO` | float | supply code |
| `CONTRACT_CAPACITY` | float | contracted capacity |
| `ACCT_CONTROL` | float | control/inspection code |
| `ACCT_WGS84_X` / `ACCT_WGS84_Y` | str | WGS84 coordinates with **comma decimal separator** (e.g. `"23,72711"`) |
| `SUPPLIER`, `SUPPLIER_TO` | str/int | supplier codes |
| `REQUEST_TYPE`, `COMPL_REQUEST_STATUS` | str/int | request metadata |

`clean_data_test.pkl` has the same structure without a usable `label`.

## Synthetic demo data

To smoke-test the pipeline structure without the real data:

```sh
python data/make_demo_data.py
```

This writes small synthetic `clean_data.pkl` / `clean_data_test.pkl` files into the
repository root (git-ignored). The demo data matches the schema above — including
comma-decimal coordinates, zero-encoded missing readings, and heavy class
imbalance — but is **randomly generated**: it exists only to exercise the code
path, and any metrics obtained on it are meaningless.
