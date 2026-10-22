"""Evaluador mínimo de un subconjunto de DQDL, para entender AWS Glue Data Quality localmente.

Minimal evaluator for a subset of DQDL, to understand AWS Glue Data Quality locally.
Supported rules: IsComplete, IsUnique, Completeness, ColumnValues (between / in), RowCount.
Glue Data Quality supports many more rule types; this is for learning, not a replacement.
"""

import operator
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str


OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "=": operator.eq}


def _compare(value: float, op: str, threshold: float) -> bool:
    return OPERATORS[op](value, threshold)


def parse_rules(text: str) -> list[str]:
    body = re.search(r"Rules\s*=\s*\[(.*)\]", text, re.DOTALL)
    if not body:
        raise ValueError("Expected a ruleset like: Rules = [ ... ]")
    # Separa por comas que no estén dentro de corchetes. Split on commas outside brackets.
    return [r.strip() for r in re.split(r",(?![^\[]*\])", body.group(1)) if r.strip()]


def evaluate_rule(df: pd.DataFrame, rule: str) -> RuleResult:
    if m := re.fullmatch(r'IsComplete "(\w+)"', rule):
        nulls = int(df[m[1]].isna().sum())
        return RuleResult(rule, nulls == 0, f"{nulls} nulls")

    if m := re.fullmatch(r'IsUnique "(\w+)"', rule):
        dupes = int(df[m[1]].duplicated().sum())
        return RuleResult(rule, dupes == 0, f"{dupes} duplicated values")

    if m := re.fullmatch(r'Completeness "(\w+)" (>=|>|<=|<|=) ([\d.]+)', rule):
        ratio = float(df[m[1]].notna().mean())
        return RuleResult(rule, _compare(ratio, m[2], float(m[3])), f"completeness = {ratio:.3f}")

    if m := re.fullmatch(r'ColumnValues "(\w+)" between ([\d.]+) and ([\d.]+)', rule):
        col = df[m[1]].dropna()
        bad = int((~col.between(float(m[2]), float(m[3]))).sum())
        return RuleResult(rule, bad == 0, f"{bad} values out of range")

    if m := re.fullmatch(r'ColumnValues "(\w+)" in \[(.*)\]', rule):
        allowed = {float(v) for v in m[2].split(",")}
        bad = int((~df[m[1]].isin(allowed)).sum())
        return RuleResult(rule, bad == 0, f"{bad} values not allowed")

    if m := re.fullmatch(r"RowCount (>=|>|<=|<|=) (\d+)", rule):
        return RuleResult(rule, _compare(len(df), m[1], int(m[2])), f"{len(df)} rows")

    raise ValueError(f"Unsupported rule in this lab: {rule}")


def evaluate(df: pd.DataFrame, ruleset_path: Path) -> tuple[float, list[RuleResult]]:
    results = [evaluate_rule(df, r) for r in parse_rules(ruleset_path.read_text())]
    score = sum(r.passed for r in results) / len(results)
    return score, results


def print_report(title: str, score: float, results: list[RuleResult]) -> None:
    print(f"\n{title}  —  data quality score: {score:.0%}")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.rule:<45} {r.detail}")
