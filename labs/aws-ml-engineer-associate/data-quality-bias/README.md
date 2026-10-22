# Lab 1.3 — Calidad de datos, sesgo y desbalance

**ES** · [EN](#english)

**Curso:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Tema:** 1.3 Validar la calidad de datos y gestionar el sesgo

## Objetivo

Reproducir localmente lo que hacen AWS Glue Data Quality, DataBrew o Data Wrangler y SageMaker Clarify en este flujo:

1. Validar datos con reglas al estilo DQDL y ver cómo fallan.
2. Limpiar duplicados, valores inválidos, faltantes y outliers **sin data leakage**.
3. Medir el sesgo pre-training con **CI** y **DPL**.
4. Comparar estrategias para el desbalance: `scale_pos_weight` y SMOTE frente a la línea base.

No requiere cuenta de AWS, credenciales ni internet, salvo para instalar paquetes.

## Dataset

`src/generate_data.py` genera 10.150 solicitudes de préstamo sintéticas con una semilla fija, así que el resultado es reproducible. Contienen problemas a propósito:

- 150 duplicados.
- 15 edades inválidas.
- Alrededor de 9 % de `income` faltante, más frecuente entre impagos.
- 40 outliers de captura.
- Un 8 % de impagos.
- Mayores de 60 subrepresentados y con más impagos.

## Ejecutar

Requiere Python 3.10 o superior.

```bash
cd labs/aws-ml-engineer-associate/data-quality-bias
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_lab.py
```

Al ejecutarse, el laboratorio guarda los datos crudos en `data/loans_raw.csv`.

## Qué observar

| Paso | Resultado esperado |
|---|---|
| 2. Calidad | Score de 67 %: `IsUnique` y `ColumnValues "age"` fallan. En Glue ETL, el trabajo se detendría. |
| 3. Limpieza | Mediana y límites IQR calculados solo con el entrenamiento; división estratificada (≈ 8,6 % de impagos en ambos conjuntos). El score sube a 100 %. |
| 4. Sesgo | CI ≈ +0,72 (faceta d subrepresentada) y DPL ≈ −0,04 (la faceta d tiene más impagos). |
| 5. Desbalance | La línea base logra ≈ 91 % de *accuracy* con *recall* ≈ 0,06. `scale_pos_weight` y SMOTE sacrifican *accuracy* para subir el *recall* a ≈ 0,56–0,58. |

Prueba esto:
- Cambia `Completeness "income" > 0.85` por `> 0.95` en `rules.dqdl`.
- Aplica SMOTE **antes** de dividir los datos y observa cómo se inflan las métricas de prueba.

## Equivalencia en AWS

| Este laboratorio | En AWS | Diferencias principales |
|---|---|---|
| `quality.py` + `rules.dqdl` | AWS Glue Data Quality (DQDL), transform *Evaluate Data Quality* | Glue admite muchas más reglas, reglas dinámicas (`avg(last(3))`), detección de anomalías, recomendaciones, métricas en CloudWatch y la opción de detener el trabajo. Este evaluador solo implementa 6 tipos de regla. |
| `clean.py` | AWS Glue DataBrew (`REMOVE_DUPLICATES`, `FILL_WITH_MEDIAN`, `REPLACE_OUTLIERS`) o SageMaker Data Wrangler | Los servicios son visuales y escalan con recetas o flujos reutilizables. Aquí la limpieza es código pandas. |
| `bias.py` | Métricas pre-training de SageMaker Clarify | Clarify calcula 8 métricas y genera informes, pero ya no acepta clientes nuevos. AWS recomienda calcularlas con pandas, como aquí. |
| `imbalance.py` | XGBoost integrado de SageMaker AI (`scale_pos_weight`) y *Balance data* de Data Wrangler (SMOTE) | En AWS el entrenamiento corre en instancias gestionadas, y el ajuste automático puede optimizar `validation:f1` o `validation:auc`. |

## Referencias

- [AWS Glue Data Quality](https://docs.aws.amazon.com/glue/latest/dg/glue-data-quality.html) · [DQDL](https://docs.aws.amazon.com/glue/latest/dg/dqdl.html)
- [DataBrew: outliers](https://docs.aws.amazon.com/databrew/latest/dg/recipe-actions.outliers.html)
- [Métricas de sesgo pre-training](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.html)
- [Hiperparámetros de XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)

---

## English

**Course:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Topic:** 1.3 Validate data quality and manage bias

### Goal

Reproduce locally what AWS Glue Data Quality, DataBrew or Data Wrangler, and SageMaker Clarify do in this workflow:

1. Validate data with DQDL-style rules and watch them fail.
2. Clean duplicates, invalid values, missing values, and outliers **without data leakage**.
3. Measure pre-training bias with **CI** and **DPL**.
4. Compare imbalance strategies: `scale_pos_weight` and SMOTE against a baseline.

No AWS account, credentials, or internet access is needed, except to install packages.

### Dataset

`src/generate_data.py` generates 10,150 synthetic loan applications with a fixed seed, so results are reproducible. They contain deliberate problems:

- 150 duplicates.
- 15 invalid ages.
- About 9% missing `income`, more common among defaults.
- 40 data-entry outliers.
- 8% defaults.
- Applicants over 60 are underrepresented and default more often.

### Run

Requires Python 3.10 or later.

```bash
cd labs/aws-ml-engineer-associate/data-quality-bias
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_lab.py
```

When it runs, the lab saves the raw data to `data/loans_raw.csv`.

### What to look for

| Step | Expected result |
|---|---|
| 2. Quality | 67% score: `IsUnique` and `ColumnValues "age"` fail. In Glue ETL, the job would stop. |
| 3. Cleaning | Median and IQR bounds computed on the training split only; stratified split (≈ 8.6% defaults in both sets). The score rises to 100%. |
| 4. Bias | CI ≈ +0.72 (facet d underrepresented) and DPL ≈ −0.04 (facet d defaults more). |
| 5. Imbalance | The baseline gets ≈ 91% accuracy with recall ≈ 0.06. `scale_pos_weight` and SMOTE trade accuracy for recall of ≈ 0.56–0.58. |

Try this:
- Change `Completeness "income" > 0.85` to `> 0.95` in `rules.dqdl`.
- Apply SMOTE **before** splitting the data and watch the test metrics inflate.

### AWS equivalent

| This lab | On AWS | Main differences |
|---|---|---|
| `quality.py` + `rules.dqdl` | AWS Glue Data Quality (DQDL), *Evaluate Data Quality* transform | Glue supports many more rules, dynamic rules (`avg(last(3))`), anomaly detection, recommendations, CloudWatch metrics, and the option to stop the job. This evaluator implements only 6 rule types. |
| `clean.py` | AWS Glue DataBrew (`REMOVE_DUPLICATES`, `FILL_WITH_MEDIAN`, `REPLACE_OUTLIERS`) or SageMaker Data Wrangler | The services are visual and scale with reusable recipes or flows. Here, cleaning is pandas code. |
| `bias.py` | SageMaker Clarify pre-training metrics | Clarify computes 8 metrics and builds reports, but no longer accepts new customers. AWS recommends computing them with pandas, as done here. |
| `imbalance.py` | SageMaker AI built-in XGBoost (`scale_pos_weight`) and Data Wrangler *Balance data* (SMOTE) | On AWS, training runs on managed instances, and automatic model tuning can optimize `validation:f1` or `validation:auc`. |
