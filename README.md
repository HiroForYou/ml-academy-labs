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

Cada laboratorio tiene su propio entorno virtual. Los notebooks se publican con sus salidas para poder leerlos en GitHub sin ejecutarlos.

## Laboratorios

| Curso | Tema | Laboratorio | Dependencias |
|---|---|---|---|
| AWS ML Engineer – Associate | 1.3 Calidad de datos y sesgo | [data-quality-bias](labs/aws-ml-engineer-associate/data-quality-bias) | pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, JupyterLab |

## Desarrollo

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .   # también revisa los notebooks
```
