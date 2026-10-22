# ML Academy — labs

**ES** · [EN](#english)

Laboratorios locales y gratuitos para los cursos de [ML Academy](https://github.com/HiroForYou/ml-academy-web). No requieren cuentas de nube, credenciales ni servicios de pago.

## Estructura

```text
labs/<curso>/<tema>/
├── README.md          # objetivo, ejecución y equivalencia en la nube
├── requirements.txt   # solo las dependencias de ese laboratorio
└── src/
```

Cada laboratorio tiene su propio entorno virtual y sus propias dependencias.

## Laboratorios

| Curso | Tema | Laboratorio | Dependencias |
|---|---|---|---|
| AWS ML Engineer – Associate | 1.3 Calidad de datos y sesgo | [data-quality-bias](labs/aws-ml-engineer-associate/data-quality-bias) | pandas, scikit-learn, XGBoost, imbalanced-learn |

## Desarrollo

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
```

---

## English

Free, local labs for the [ML Academy](https://github.com/HiroForYou/ml-academy-web) courses. They don't require cloud accounts, credentials, or paid services.

Each lab lives in `labs/<course>/<topic>/` with its own `README.md`, `requirements.txt` (only what that lab needs), and `src/`. Use a separate virtual environment per lab.

| Course | Topic | Lab | Dependencies |
|---|---|---|---|
| AWS ML Engineer – Associate | 1.3 Data quality and bias | [data-quality-bias](labs/aws-ml-engineer-associate/data-quality-bias) | pandas, scikit-learn, XGBoost, imbalanced-learn |

Development: `pip install -r requirements-dev.txt`, then `ruff check . && ruff format --check .`.
