# Lab 1.3 — Calidad de datos, sesgo y desbalance

**Español** · [English](README.en.md)

**Curso:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Tema:** 1.3 Validar la calidad de datos y gestionar el sesgo

## Objetivo

Reproducir localmente lo que hacen AWS Glue Data Quality, DataBrew o Data Wrangler y SageMaker Clarify:

1. Validar datos con reglas al estilo DQDL y ver cómo fallan.
2. Limpiar duplicados, valores inválidos, faltantes y outliers **sin data leakage**.
3. Medir el sesgo pre-training con **CI** y **DPL**.
4. Comparar estrategias para el desbalance: `scale_pos_weight` y SMOTE frente a la línea base.

No requiere cuenta de AWS, credenciales ni internet, salvo para instalar paquetes.

## Dataset

`src/generate_data.py` genera 10.150 solicitudes de préstamo sintéticas con una semilla fija, así que el resultado es reproducible. Incluyen problemas a propósito:

- 150 duplicados y 15 edades inválidas.
- Alrededor de 9 % de `income` faltante, más frecuente entre impagos.
- 40 outliers de captura.
- 8,6 % de impagos.
- Mayores de 60 subrepresentados y con más impagos.

## Ejecutar

Requiere Python 3.10 o superior. Todos los laboratorios y `ml-academy-media` comparten **un solo entorno virtual**; cada laboratorio solo instala su propio `requirements.txt`.

```bash
# Entorno global único, en la carpeta raíz ml-academy/ (se crea una sola vez).
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r ml-academy-labs/labs/aws-ml-engineer-associate/data-quality-bias/requirements.txt
cd ml-academy-labs/labs/aws-ml-engineer-associate/data-quality-bias
jupyter lab notebook.es.ipynb
```

El notebook ya incluye las salidas y los gráficos de una ejecución, así que puedes leerlo directamente en GitHub. Para reproducirlo, usa *Run → Run All Cells*.

| Archivo | Contenido |
|---|---|
| `notebook.es.ipynb` | Laboratorio guiado en español |
| `rules.dqdl` | Reglas de calidad con sintaxis DQDL |
| `src/` | Módulos reutilizables: datos, calidad, limpieza, sesgo, desbalance y gráficos |

## Qué observar

| Paso | Resultado de la ejecución incluida |
|---|---|
| 1. Datos crudos | 10.150 filas, 8,6 % de impagos, 8,9 % de `income` faltante. Predecir siempre «sin impago» ya da 91 % de *accuracy*. |
| 2. Calidad | Score de 67 %: fallan `IsUnique` (150 duplicados) y `ColumnValues "age"` (15 edades inválidas). En Glue ETL, el trabajo se detendría. |
| 3. Limpieza | Quedan 9.985 filas. La mediana y los límites IQR se calculan solo con el entrenamiento, y la división estratificada mantiene 8,64 % y 8,61 % de impagos. El score sube a 100 %. |
| 4. Sesgo | CI = +0,72: la faceta d está subrepresentada. DPL = −0,041 (8,1 % frente a 12,2 %): la faceta d tiene más impagos. |
| 5. Desbalance | La línea base logra 91,5 % de *accuracy* con *recall* de 0,06. `scale_pos_weight=10.6` sube el *recall* a 0,56 y el F1 a 0,27; SMOTE, aplicado solo al entrenamiento, sube el *recall* a 0,58. |

## Equivalencia en AWS

| Este laboratorio | En AWS | Diferencias principales |
|---|---|---|
| `quality.py` + `rules.dqdl` | AWS Glue Data Quality (DQDL), transform *Evaluate Data Quality* | Glue admite muchas más reglas, reglas dinámicas (`avg(last(3))`), detección de anomalías, recomendaciones, métricas en CloudWatch y la opción de detener el trabajo. Este evaluador solo implementa 6 tipos de regla. |
| `clean.py` | AWS Glue DataBrew (`REMOVE_DUPLICATES`, `FILL_WITH_MEDIAN`, `REPLACE_OUTLIERS`) o SageMaker Data Wrangler | Los servicios son visuales y escalan con recetas o flujos reutilizables. Aquí la limpieza es código pandas. |
| `bias.py` | Métricas pre-training de SageMaker Clarify | Clarify calcula 8 métricas y genera informes, pero ya no acepta clientes nuevos. AWS recomienda calcularlas con pandas, como aquí. |
| `imbalance.py` | XGBoost integrado de SageMaker AI (`scale_pos_weight`) y *Balance data* de Data Wrangler (SMOTE) | En AWS el entrenamiento corre en instancias gestionadas, y el ajuste automático puede optimizar `validation:f1` o `validation:auc`. |

## Referencias

- [AWS Glue Data Quality](https://docs.aws.amazon.com/es_es/glue/latest/dg/glue-data-quality.html) · [DQDL](https://docs.aws.amazon.com/es_es/glue/latest/dg/dqdl.html)
- [DataBrew: outliers](https://docs.aws.amazon.com/es_es/databrew/latest/dg/recipe-actions.outliers.html)
- [Métricas de sesgo pre-training](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/clarify-measure-data-bias.html)
- [Hiperparámetros de XGBoost](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/xgboost_hyperparameters.html)
