# Lab 4.1 — Detect data drift in production

[Español](README.md) · **English**

**Course:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Topic:** 4.1 Monitor inference and performance

## Goal

Reproduce locally what the SageMaker AI monitoring solutions (MLflow + Evidently AI) or SageMaker Model Monitor do:

1. Compute a baseline from the training data.
2. Measure each batch's drift with **PSI** and **Kolmogorov-Smirnov**, with alert thresholds.
3. Catch a data-quality error with a completeness constraint.
4. See why a model-quality drop is only known **once the labels arrive**.

No AWS account, credentials, or internet access is needed, except to install packages.

## Dataset

`src/generate_data.py` generates, with a fixed seed, synthetic card-payment transactions for a fraud model (7% fraud): 10,000 reference (training) rows and 14 weekly batches of 2,000. It deliberately includes:

- `amount`: gradual drift from week 1.
- `channel`: sudden shift in week 6 (the app goes from 35% to 60%).
- `customer_age` and `account_age_days`: stable, as a control.
- Week 11: 8% missing `customer_age` due to a pipeline bug.
- Week 9: the fraud pattern changes (*concept drift*) without changing the input distributions.

## Run

Requires Python 3.10 or later. All labs and `ml-academy-media` share **a single virtual environment**; each lab only installs its own `requirements.txt`.

```bash
# Single global environment in the root ml-academy/ folder (create it once).
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r ml-academy-labs/labs/aws-ml-engineer-associate/monitoring-inference/requirements.txt
cd ml-academy-labs/labs/aws-ml-engineer-associate/monitoring-inference
jupyter lab notebook.en.ipynb
```

The notebook ships with the outputs and charts from a run, so you can read it directly on GitHub. To reproduce it, use *Run → Run All Cells*.

| File | Contents |
|---|---|
| `notebook.en.ipynb` | Guided lab in English |
| `src/` | Reusable modules: data, drift, model, and charts |

## What to look for

PSI thresholds: < 0.1 stable, 0.1–0.2 warning, ≥ 0.2 alert. They are a **common convention** from credit risk, **not an AWS rule**.

| Step | Result from the included run |
|---|---|
| 1. Sudden PSI | `channel` jumps from 0.001 (week 5) to 0.254 (week 6): immediate alert. |
| 2. Gradual PSI | `amount` reaches warning in week 7 (0.102), alert in week 11 (0.224), and 0.420 in week 14. The stable features stay at or below 0.011. |
| 3. KS | The `amount` p-value is already < 0.05 in week 2, with a PSI of 0.038. On the stable features, 2 of 28 tests give p < 0.05 by chance: significant is not the same as important. |
| 4. Data quality | Week 11: 8.9% missing `customer_age`. The completeness constraint catches it; PSI (0.002) doesn't, because it ignores missing values. |
| 5. Drift without degradation | Weeks 6 to 8 fire PSI alerts, but AUC stays between 0.84 and 0.88. |
| 6. Label delay | In week 9, the true AUC drops to 0.59 and recall from 0.43 to 0.10. Neither PSI nor the model's alert rate sees it. With a 3-week label delay, the drop becomes known in week 12. |

## AWS equivalent

SageMaker Model Monitor no longer accepts new customers; existing customers can keep using it. AWS documents the open-source SageMaker AI monitoring solutions (SageMaker AI MLflow Apps + Evidently AI), with Amazon QuickSight dashboards and Amazon CloudWatch, as the replacement.

| This lab | On AWS | Main differences |
|---|---|---|
| `drift.baseline` | Evidently reference dataset (the training data), or the Model Monitor baseline (`statistics.json`, `constraints.json`) | On AWS, the reference and the captured data live in S3. |
| `drift.drift_report` (PSI, KS) | Evidently `DataDriftPreset`, run by a SageMaker Pipeline or by a Lambda scheduled with EventBridge | Evidently computes PSI and KS for every feature and logs HTML reports and metrics to an MLflow App. The sample batch pipeline alerts by default when more than 30% of the features drift. |
| Completeness constraint | Evidently `DataSummaryPreset` (missing values, outliers, integrity) | More checks than this single rule. |
| `model.weekly_quality` with delay | Evidently `ClassificationPreset` when ground truth arrives, joined to predictions by inference ID | On an endpoint, Data Capture stores inputs and outputs in S3. |
| `status` and thresholds | Amazon SNS alerts; CloudWatch custom metrics and anomaly detection alarms | Here, alerts are just a column in the report. |

## References

- [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- [Model Monitor availability change and alternatives](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-availability-change.html)
- [Open-source SageMaker AI monitoring solutions (aws-samples)](https://github.com/aws-samples/sample-aiops-on-amazon-sagemakerai/tree/main/monitoring)
- [CloudWatch anomaly detection](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Anomaly_Detection.html)
