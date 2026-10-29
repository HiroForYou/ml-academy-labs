# Lab 4.1 — Detectar data drift en producción

**Español** · [English](README.en.md)

**Curso:** AWS Certified Machine Learning Engineer – Associate (MLA-C02) · **Tema:** 4.1 Monitorear la inferencia y el rendimiento

## Objetivo

Reproducir localmente lo que hacen las soluciones de monitoreo de SageMaker AI (MLflow + Evidently AI) o SageMaker Model Monitor:

1. Calcular una línea base con los datos de entrenamiento.
2. Medir el drift de cada lote con **PSI** y **Kolmogorov-Smirnov**, con umbrales de alerta.
3. Detectar un error de calidad de datos con una restricción de completitud.
4. Ver por qué la caída de calidad del modelo solo se conoce **cuando llegan las etiquetas**.

No requiere cuenta de AWS, credenciales ni internet, salvo para instalar paquetes.

## Dataset

`src/generate_data.py` genera, con una semilla fija, transacciones sintéticas de pago con tarjeta para un modelo de fraude (7 % de fraude): 10.000 de referencia (entrenamiento) y 14 lotes semanales de 2.000. Incluye a propósito:

- `amount`: drift gradual desde la semana 1.
- `channel`: cambio brusco en la semana 6 (la app pasa del 35 % al 60 %).
- `customer_age` y `account_age_days`: estables, como control.
- Semana 11: 8 % de `customer_age` faltante por un error del pipeline.
- Semana 9: cambia el patrón de fraude (*concept drift*) sin cambiar las distribuciones de entrada.

## Ejecutar

Requiere Python 3.10 o superior. Todos los laboratorios y `ml-academy-media` comparten **un solo entorno virtual**; cada laboratorio solo instala su propio `requirements.txt`.

```bash
# Entorno global único, en la carpeta raíz ml-academy/ (se crea una sola vez).
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r ml-academy-labs/labs/aws-ml-engineer-associate/monitoring-inference/requirements.txt
cd ml-academy-labs/labs/aws-ml-engineer-associate/monitoring-inference
jupyter lab notebook.es.ipynb
```

El notebook ya incluye las salidas y los gráficos de una ejecución, así que puedes leerlo directamente en GitHub. Para reproducirlo, usa *Run → Run All Cells*.

| Archivo | Contenido |
|---|---|
| `notebook.es.ipynb` | Laboratorio guiado en español |
| `src/` | Módulos reutilizables: datos, drift, modelo y gráficos |

## Qué observar

Umbrales de PSI: < 0,1 estable, 0,1–0,2 aviso, ≥ 0,2 alerta. Son una **convención habitual** del riesgo crediticio, **no una regla de AWS**.

| Paso | Resultado de la ejecución incluida |
|---|---|
| 1. PSI brusco | `channel` pasa de 0,001 (semana 5) a 0,254 (semana 6): alerta inmediata. |
| 2. PSI gradual | `amount` llega a aviso en la semana 7 (0,102), a alerta en la 11 (0,224) y a 0,420 en la 14. Las variables estables no superan 0,011. |
| 3. KS | El p-valor de `amount` ya es < 0,05 en la semana 2, con PSI de 0,038. En las variables estables, 2 de 28 pruebas dan p < 0,05 por azar: significativo no es lo mismo que importante. |
| 4. Calidad de datos | Semana 11: 8,9 % de `customer_age` faltante. La restricción de completitud lo detecta; el PSI (0,002) no, porque ignora los faltantes. |
| 5. Drift sin degradación | Entre las semanas 6 y 8 hay alertas de PSI, pero la AUC sigue entre 0,84 y 0,88. |
| 6. Retraso de etiquetas | En la semana 9 la AUC real cae a 0,59 y el *recall* de 0,43 a 0,10. Ni el PSI ni la tasa de alertas del modelo lo ven. Con 3 semanas de retraso de etiquetas, la caída se conoce en la semana 12. |

## Equivalencia en AWS

SageMaker Model Monitor ya no acepta clientes nuevos; los clientes existentes pueden seguir usándolo. AWS documenta como reemplazo las soluciones de monitoreo open source de SageMaker AI (SageMaker AI MLflow Apps + Evidently AI), con paneles de Amazon QuickSight y Amazon CloudWatch.

| Este laboratorio | En AWS | Diferencias principales |
|---|---|---|
| `drift.baseline` | Dataset de referencia de Evidently (los datos de entrenamiento), o la línea base de Model Monitor (`statistics.json`, `constraints.json`) | En AWS la referencia y los datos capturados viven en S3. |
| `drift.drift_report` (PSI, KS) | Evidently `DataDriftPreset`, ejecutado por una SageMaker Pipeline o por una Lambda programada con EventBridge | Evidently calcula PSI y KS para cada variable y registra informes HTML y métricas en una MLflow App. El pipeline batch de ejemplo alerta por defecto si deriva más del 30 % de las variables. |
| Restricción de completitud | Evidently `DataSummaryPreset` (faltantes, outliers, integridad) | Más comprobaciones que esta única regla. |
| `model.weekly_quality` con retraso | Evidently `ClassificationPreset` cuando llega el ground truth, unido a las predicciones por ID de inferencia | En un endpoint, Data Capture guarda entradas y salidas en S3. |
| `status` y umbrales | Alertas con Amazon SNS; métricas personalizadas y alarmas de detección de anomalías en CloudWatch | Aquí las alertas son solo una columna del informe. |

## Referencias

- [SageMaker Model Monitor](https://docs.aws.amazon.com/es_es/sagemaker/latest/dg/model-monitor.html)
- [Cambio de disponibilidad de Model Monitor y alternativas](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-availability-change.html)
- [Soluciones de monitoreo open source de SageMaker AI (aws-samples)](https://github.com/aws-samples/sample-aiops-on-amazon-sagemakerai/tree/main/monitoring)
- [Detección de anomalías en CloudWatch](https://docs.aws.amazon.com/es_es/AmazonCloudWatch/latest/monitoring/CloudWatch_Anomaly_Detection.html)
