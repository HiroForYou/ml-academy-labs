# ML Academy — labs

[Español](README.md) · **English**

Free, local labs for the [ML Academy](https://github.com/HiroForYou/ml-academy-web) courses. They are Jupyter notebooks that don't require cloud accounts, credentials, or paid services.

## Structure

```text
labs/<course>/<topic>/
├── README.md / README.en.md          # goal, how to run, cloud equivalent
├── notebook.es.ipynb / notebook.en.ipynb
├── requirements.txt                  # only what that lab needs
├── .gitignore
└── src/                              # reusable modules imported by the notebook
```

All labs use **a single global virtual environment** in the root `ml-academy/.venv`; each one lists only its own dependencies in its `requirements.txt`. Notebooks are committed with their outputs so you can read them on GitHub without running them.

## Labs

| Course | Topic | Lab | Dependencies |
|---|---|---|---|
| AWS ML Engineer – Associate | 1.3 Data quality and bias | [data-quality-bias](labs/aws-ml-engineer-associate/data-quality-bias) | pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib, JupyterLab |

## Development

```bash
pip install -r ml-academy-labs/requirements-dev.txt   # in the global ml-academy/.venv
cd ml-academy-labs
ruff check . && ruff format --check .   # also checks the notebooks
```
