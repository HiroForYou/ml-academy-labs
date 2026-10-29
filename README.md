# ML Academy — labs

**Español** · [English](README.en.md)

Laboratorios locales y gratuitos para los cursos de [ML Academy](https://github.com/HiroForYou/ml-academy-web). Son notebooks de Jupyter que no requieren cuentas de nube, credenciales ni servicios de pago.

## Estructura

```text
labs/<curso>/<tema>/
├── README.md / README.en.md          # objetivo, ejecución y equivalencia en la nube
├── notebook.es.ipynb / notebook.en.ipynb
├── requirements.txt                  # solo las dependencias de ese laboratorio
├── .gitignore
└── src/                              # módulos reutilizables que importa el notebook
```

Todos los laboratorios usan **un solo entorno virtual global** en la carpeta raíz `ml-academy/.venv`; cada uno declara solo sus dependencias en su `requirements.txt`. Los notebooks se publican con sus salidas para poder leerlos en GitHub sin ejecutarlos.

## Laboratorios

| Curso | Tema | Laboratorio | Dependencias |
|---|---|---|---|
| AWS ML Engineer – Associate | 1.3 Calidad de datos y sesgo | [data-quality-bias](labs/aws-ml-engineer-associate/data-quality-bias) | pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, JupyterLab |
| AWS ML Engineer – Associate | 2.3 Analizar y evaluar el rendimiento (RAG) | [model-evaluation](labs/aws-ml-engineer-associate/model-evaluation) | pandas, scikit-learn, matplotlib, JupyterLab |
| AWS ML Engineer – Associate | 4.1 Monitorear la inferencia (data drift) | [monitoring-inference](labs/aws-ml-engineer-associate/monitoring-inference) | pandas, SciPy, scikit-learn, matplotlib, JupyterLab |

## Desarrollo

```bash
pip install -r ml-academy-labs/requirements-dev.txt   # en el entorno global ml-academy/.venv
cd ml-academy-labs
ruff check . && ruff format --check .   # también revisa los notebooks
```
