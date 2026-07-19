# HEDNO Datathon — Power Theft Detection

A machine-learning prototype built for the HEDNO (Hellenic Electricity Distribution
Network Operator) Datathon. The task: given each customer account's consumption
time series plus contract and geographic metadata, flag accounts likely to be
involved in **power theft**, so field inspections can be prioritized.

This is a highly **imbalanced binary classification** problem (~1% positive class)
on roughly **1.5 million accounts**.

> **Scope note.** This is a datathon prototype, not a production system. The
> original data is not redistributable and is **not included** — see
> [data/README.md](data/README.md) for the expected schema and a synthetic demo
> generator that lets the pipeline run end-to-end as a smoke test.

## Approach

```
raw accounts (~1.5M rows)
   │
   ├─ 1. Geospatial clustering ────── HDBSCAN (haversine metric) on WGS84 coords
   │                                  → cluster_labels feature
   ├─ 2. Missing-value imputation ─── zero readings replaced by the non-zero mean
   │                                  of the account's peer group
   │                                  (rate × usage × supplier × voltage × geo-cluster)
   ├─ 3. Feature engineering ──────── time-series unpacked to measurement_i columns,
   │                                  IDs dropped, categoricals one-hot encoded
   ├─ 4. Detector ensemble ────────── RandomForest, XGBoost, CatBoost, LightGBM
   │                                  with class weighting + percentile thresholding
   └─ 5. Meta-learner (experimental)─ Double DQN over features + detector probabilities
```

The key domain insight is in step 2: a zero meter reading means a *missing* value,
not zero consumption, so it is imputed from similar customers in the same
geographic cluster — which is why clustering runs before imputation.

## Repository layout

| Path | Contents |
|---|---|
| `Main.ipynb` | End-to-end orchestration of the full pipeline |
| `HDBSCAN.ipynb` | Step 1 — geospatial clustering (writes `clustered_data.pkl`) |
| `Imputation.ipynb` | Step 2 — group-average imputation (writes `imputated_clustered_data.pkl`) |
| `Sensor_Predictions.ipynb` | Steps 3–4 — encoding, train/test split, detector training and evaluation |
| `DQN.ipynb` | Step 5 scratchpad — experimental, not standalone-runnable |
| `Agent/Utils/GeoClustering.py` | `GeoClustering` — chunked HDBSCAN with haversine metric + `approximate_predict` for new points |
| `Agent/Utils/Imputation.py` | `ImbDataProcessor` — group-conditional imputation, reusable on unseen data |
| `Agent/Utils/ModelRunner.py` | Thin train/test driver used by `Main.ipynb` |
| `Agent/Sensors/*.py` | sklearn-style anomaly detectors (RandomForest, XGBoost, CatBoost, LightGBM) plus extra ensembles explored (AdaBoost, Stacking, Majority Vote) |
| `Agent/dqn_binary_classification_memory_optimized.py` | DQN / Double DQN with prioritized replay (experimental) |
| `data/` | Data documentation + synthetic demo data generator |

## Handling class imbalance

- **Metrics:** precision, recall, F1 and **F2** (recall-weighted — missing a theft
  costs more than a false alarm) are reported instead of relying on accuracy,
  which is trivially ~0.98 here.
- **Weighting:** minority/majority class weights (0.7 / 0.3) applied as class or
  sample weights per detector.
- **Thresholding:** the decision threshold is set from the observed outlier
  fraction rather than the default 0.5.
- **Splits:** all train/test splits are stratified on the label.

## Results (original datathon run, 2023)

Held-out 20% test split, ~1.5M-row dataset, sample-weighted metrics
(recorded from the executed `Sensor_Predictions.ipynb`):

| Model | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|
| **RandomForest** | 0.9872 | **0.7465** | **0.5580** | **0.6386** | **0.5876** |
| CatBoost | 0.9857 | 0.7027 | 0.5032 | 0.5864 | 0.5335 |
| XGBoost | 0.9813 | 0.5580 | 0.3511 | 0.4310 | 0.3792 |
| LightGBM | 0.9810 | 0.5491 | 0.3429 | 0.4222 | 0.3707 |

RandomForest was the strongest single detector. The Double DQN meta-learner did
not produce useful results in the time available (its Q-values collapsed to
near-uniform values) and is kept as an experimental exploration only.

These numbers come from a single stratified split on the original (private)
data; they are not reproducible from this repository alone and should be read as
indicative, not benchmarked.

## How to run

```sh
git clone <repository_url> && cd Hedno-Datathon-AI-Solution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# No access to the original data? Generate a synthetic smoke-test set:
python data/make_demo_data.py
```

Then run the notebooks from the repository root, in order:

1. `HDBSCAN.ipynb` — expects `clean_data.pkl`, writes `clustered_data.pkl`
2. `Imputation.ipynb` — writes `imputated_clustered_data.pkl` (create a `Models/` directory first)
3. `Sensor_Predictions.ipynb` — trains and evaluates the four detectors
4. `Main.ipynb` — full pipeline including the experimental DQN stage (optional; requires TensorFlow, slow)

## Known limitations / next steps

Honest list of what a production version would need to address:

- **Per-chunk clustering:** HDBSCAN is refit per 5,000-row chunk for memory
  reasons, so cluster labels are not globally consistent across chunks and
  `approximate_predict` only reflects the last chunk's model. Fix: fit once on a
  sample, or use a scalable global clustering.
- **Batch-dependent threshold:** detectors derive their percentile threshold from
  the batch being scored; a deployable system needs a threshold frozen on
  validation data.
- **Single split:** no cross-validation, PR-AUC, calibration, or confusion-matrix
  analysis.
- **Pickle-based I/O** and no experiment tracking, tests, or serving layer.
- The DQN stage would need reward shaping and evaluation work before it adds value.
