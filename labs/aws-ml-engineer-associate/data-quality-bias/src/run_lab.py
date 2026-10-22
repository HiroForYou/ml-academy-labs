"""Ejecuta el laboratorio completo. Runs the whole lab.

python src/run_lab.py
"""

import sys
from pathlib import Path

import pandas as pd

import bias
import clean
import imbalance
import quality
from generate_data import generate

LAB_DIR = Path(__file__).resolve().parents[1]
RULES = LAB_DIR / "rules.dqdl"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos en la consola de Windows / accents on Windows consoles
    pd.set_option("display.width", 120)

    section("1. Datos crudos / Raw data")
    raw = generate()
    data_dir = LAB_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    raw.to_csv(data_dir / "loans_raw.csv", index=False)
    print(f"rows={len(raw)}  default rate={raw['default'].mean():.1%}  missing income={raw['income'].isna().mean():.1%}")

    section("2. Calidad (equivalente a Glue Data Quality) / Quality (Glue Data Quality equivalent)")
    score, results = quality.evaluate(raw, RULES)
    quality.print_report("Raw data", score, results)
    if score < 1:
        print("\n  -> En Glue ETL, Evaluate Data Quality detendría el trabajo aquí.")
        print("  -> In Glue ETL, Evaluate Data Quality would stop the job here.")

    section("3. Limpieza / Cleaning")
    valid = clean.drop_invalid(raw)
    print(f"removed duplicates and invalid ages: {len(raw)} -> {len(valid)} rows")
    train, test = clean.split(valid)
    stats = clean.fit_cleaning(train)
    print(f"train stats -> income median={stats['median']:.0f}  IQR bounds=[{stats['low']:.0f}, {stats['high']:.0f}]")
    train, test = clean.apply_cleaning(train, stats), clean.apply_cleaning(test, stats)
    print(f"default rate  train={train['default'].mean():.2%}  test={test['default'].mean():.2%}  (stratified)")
    # Tras imputar ya no hay nulos; las reglas vuelven a pasar. After imputing, rules pass again.
    score, results = quality.evaluate(pd.concat([train, test]), RULES)
    quality.print_report("Clean data", score, results)

    section("4. Sesgo pre-training / Pre-training bias  (facet d = age > 60)")
    facet_d = valid["age"] > 60
    report = bias.report(valid, facet_d, "default")
    print(f"n_a={report['n_a']}  n_d={report['n_d']}  ->  CI  = {report['CI']:+.3f}")
    print(f"q_a={report['q_a']:.3f}  q_d={report['q_d']:.3f}  ->  DPL = {report['DPL']:+.3f}")
    if report["CI"] > 0:
        print("CI > 0: la faceta d está subrepresentada / facet d is underrepresented.")
    if report["DPL"] < 0:
        print("DPL < 0: la faceta d tiene más etiquetas positivas (impagos) / facet d has more positive labels (defaults).")
    elif report["DPL"] > 0:
        print("DPL > 0: la faceta a tiene más etiquetas positivas / facet a has more positive labels.")

    section("5. Desbalance de clases / Class imbalance  (test split, not resampled)")
    print(imbalance.compare(train, test).to_string())
    print("\nAccuracy hides the problem: compare recall, F1 and ROC AUC.")


if __name__ == "__main__":
    main()
