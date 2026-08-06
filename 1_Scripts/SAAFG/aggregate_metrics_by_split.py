#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Aggregate existing SAAFG Red/Blue eval JSON files for a registry-defined split.

This script does not call any model. It filters case_reports by use_case_id from
the given registry split and recomputes micro/macro metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BASE_DIR / "0_Data" / "6_SAAFG" / "9_result_test_split_2_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SAAFG metrics for a registry-defined split.")
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--red-eval-path", type=Path, required=True)
    parser.add_argument("--blue-eval-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left and right else 0.0


def split_case_ids(registry_path: Path, split: str) -> List[str]:
    cases = read_json(registry_path).get("cases") or []
    if split == "all":
        return [str(case["use_case_id"]) for case in cases]
    return [str(case["use_case_id"]) for case in cases if case.get("split") == split]


def filter_reports(path: Path, case_ids: Sequence[str]) -> List[Dict[str, Any]]:
    wanted = set(case_ids)
    payload = read_json(path)
    reports = payload.get("case_reports") or []
    selected = [report for report in reports if str(report.get("use_case_id")) in wanted]
    found = {str(report.get("use_case_id")) for report in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"{path} does not contain {len(missing)} selected case(s): {missing[:10]}")
    return sorted(selected, key=lambda report: str(report.get("use_case_id")))


def summarize_red(case_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    predicted_total = sum(int(case.get("predicted_threat_total", 0)) for case in case_reports)
    silver_total = sum(int(case.get("silver_threat_total", 0)) for case in case_reports)
    anchor_match_total = sum(int(case.get("primary_anchor_match_count", 0)) for case in case_reports)
    threat_match_total = sum(int(case.get("threat_validity_match_count", 0)) for case in case_reports)

    micro_threat_precision = ratio(threat_match_total, predicted_total)
    micro_threat_recall = ratio(threat_match_total, silver_total)

    return {
        "case_count": len(case_reports),
        "predicted_threat_total": predicted_total,
        "silver_threat_total": silver_total,
        "primary_anchor_match_total": anchor_match_total,
        "threat_validity_match_total": threat_match_total,
        "micro_threat_validity_precision": micro_threat_precision,
        "micro_threat_validity_recall": micro_threat_recall,
        "micro_threat_f1": harmonic_mean(micro_threat_precision, micro_threat_recall),
        "macro_threat_validity_recall": mean(
            float(case.get("threat_validity_recall", 0.0)) for case in case_reports
        ),
    }


def summarize_blue(case_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    raw_predicted_total = sum(int(case.get("raw_predicted_threat_total", 0)) for case in case_reports)
    silver_total = sum(int(case.get("silver_threat_total", 0)) for case in case_reports)
    task_a_valid_total = sum(int(case.get("task_a_valid_threat_total", 0)) for case in case_reports)
    generated_defense_total = sum(int(case.get("generated_defense_total", 0)) for case in case_reports)
    defense_valid_total = sum(int(case.get("defense_valid_count", 0)) for case in case_reports)

    micro_defense_precision = ratio(defense_valid_total, generated_defense_total)
    micro_defense_recall = ratio(defense_valid_total, task_a_valid_total)
    micro_e2e_precision = ratio(defense_valid_total, raw_predicted_total)
    micro_e2e_recall = ratio(defense_valid_total, silver_total)

    return {
        "raw_predicted_threat_total": raw_predicted_total,
        "task_a_valid_threat_total": task_a_valid_total,
        "generated_defense_total": generated_defense_total,
        "defense_valid_total": defense_valid_total,
        "micro_defense_validity_precision": micro_defense_precision,
        "micro_defense_validity_recall": micro_defense_recall,
        "micro_end_to_end_defense_recall": micro_e2e_recall,
        "micro_end_to_end_pipeline_precision": micro_e2e_precision,
        "micro_end_to_end_defense_f1": harmonic_mean(micro_e2e_precision, micro_e2e_recall),
        "macro_end_to_end_defense_recall": mean(
            float(case.get("end_to_end_defense_recall", 0.0)) for case in case_reports
        ),
        "macro_end_to_end_pipeline_precision": mean(
            float(case.get("end_to_end_pipeline_precision", 0.0)) for case in case_reports
        ),
    }


def round_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rounded: Dict[str, Any] = {}
    for key, value in row.items():
        rounded[key] = round(float(value), 6) if isinstance(value, float) else value
    return rounded


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(round_row(row))


def write_markdown(path: Path, row: Dict[str, Any]) -> None:
    main_metrics = [
        "micro_threat_validity_recall",
        "micro_end_to_end_defense_recall",
        "micro_end_to_end_pipeline_precision",
        "macro_end_to_end_defense_recall",
    ]
    lines = [
        f"# {row['run_tag']} Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric in main_metrics:
        lines.append(f"| `{metric}` | {float(row[metric]):.6f} |")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Cases: {row['case_count']}",
            f"- Silver threats: {row['silver_threat_total']}",
            f"- Predicted threats: {row['predicted_threat_total']}",
            f"- Valid threats: {row['threat_validity_match_total']}",
            f"- Valid defenses: {row['defense_valid_total']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_ids = split_case_ids(args.registry_path, args.split)
    red_reports = filter_reports(args.red_eval_path, case_ids)
    blue_reports = filter_reports(args.blue_eval_path, case_ids)

    red_ids = {str(report.get("use_case_id")) for report in red_reports}
    blue_ids = {str(report.get("use_case_id")) for report in blue_reports}
    if red_ids != blue_ids:
        raise ValueError("Red/Blue selected case ids do not match.")

    red_summary = summarize_red(red_reports)
    blue_summary = summarize_blue(blue_reports)
    if red_summary["silver_threat_total"] != sum(int(case.get("silver_threat_total", 0)) for case in blue_reports):
        raise ValueError("Red/Blue silver threat totals do not match.")

    row: Dict[str, Any] = {
        "run_tag": args.run_tag,
        "model": args.model,
        "setting": args.setting,
        "split": args.split,
        "registry_path": str(args.registry_path),
        "red_eval_json": str(args.red_eval_path),
        "blue_eval_json": str(args.blue_eval_path),
    }
    row.update(red_summary)
    row.update(blue_summary)

    prefix = args.output_prefix or args.run_tag
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"{prefix}_metrics.csv"
    out_json = args.output_dir / f"{prefix}_metrics.json"
    out_md = args.output_dir / f"{prefix}_metrics.md"
    write_csv(out_csv, [row])
    out_json.write_text(json.dumps(round_row(row), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, row)
    print(json.dumps(round_row(row), ensure_ascii=False, indent=2))
    print(f"[Done] csv={out_csv}")
    print(f"[Done] json={out_json}")
    print(f"[Done] md={out_md}")


if __name__ == "__main__":
    main()
