#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Aggregate SAAFG v0.2 evaluation metrics on the held-out test split.

This script reuses existing Red/Blue evaluation JSON files and recomputes
case-level aggregates after filtering case_reports by split == "test".
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "0_Data" / "6_SAAFG" / "6_Experiment_Result"
OUTPUT_ROOT = ROOT / "0_Data" / "6_SAAFG" / "9_result_test_split"
SPLIT = "test"

MAIN_METRICS = [
    "micro_threat_validity_recall",
    "micro_end_to_end_defense_recall",
    "micro_end_to_end_pipeline_precision",
    "macro_end_to_end_defense_recall",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left and right else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def rel_pct(new: float, old: float) -> str:
    if old == 0:
        return "NA"
    return f"{((new - old) / old) * 100:.2f}%"


def round_float(value: float) -> float:
    return round(float(value), 6)


def load_split_case_reports(path: Path, split: str = SPLIT) -> List[Dict[str, Any]]:
    payload = read_json(path)
    reports = payload.get("case_reports") or []
    if not isinstance(reports, list):
        raise ValueError(f"case_reports is not a list: {path}")
    return [report for report in reports if report.get("split") == split]


def summarize_red(case_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    predicted_total = sum(int(case.get("predicted_threat_total", 0)) for case in case_reports)
    silver_total = sum(int(case.get("silver_threat_total", 0)) for case in case_reports)
    anchor_match_total = sum(int(case.get("primary_anchor_match_count", 0)) for case in case_reports)
    threat_match_total = sum(int(case.get("threat_validity_match_count", 0)) for case in case_reports)

    micro_anchor_precision = ratio(anchor_match_total, predicted_total)
    micro_anchor_recall = ratio(anchor_match_total, silver_total)
    micro_threat_precision = ratio(threat_match_total, predicted_total)
    micro_threat_recall = ratio(threat_match_total, silver_total)

    return {
        "red_case_count": len(case_reports),
        "predicted_threat_total": predicted_total,
        "silver_threat_total_red": silver_total,
        "primary_anchor_match_total": anchor_match_total,
        "threat_validity_match_total": threat_match_total,
        "micro_primary_anchor_precision": micro_anchor_precision,
        "micro_primary_anchor_recall": micro_anchor_recall,
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
    judge_call_total = sum(int(case.get("judge_call_count", 0)) for case in case_reports)
    judge_parse_valid_total = sum(int(case.get("judge_parse_valid_count", 0)) for case in case_reports)
    judge_schema_valid_total = sum(int(case.get("judge_schema_valid_count", 0)) for case in case_reports)

    micro_defense_precision = ratio(defense_valid_total, generated_defense_total)
    micro_defense_recall = ratio(defense_valid_total, task_a_valid_total)
    micro_e2e_precision = ratio(defense_valid_total, raw_predicted_total)
    micro_e2e_recall = ratio(defense_valid_total, silver_total)

    return {
        "blue_case_count": len(case_reports),
        "raw_predicted_threat_total": raw_predicted_total,
        "silver_threat_total_blue": silver_total,
        "task_a_valid_threat_total": task_a_valid_total,
        "generated_defense_total": generated_defense_total,
        "defense_valid_total": defense_valid_total,
        "micro_defense_validity_precision": micro_defense_precision,
        "micro_defense_validity_recall": micro_defense_recall,
        "micro_defense_f1": harmonic_mean(micro_defense_precision, micro_defense_recall),
        "micro_end_to_end_pipeline_precision": micro_e2e_precision,
        "micro_end_to_end_defense_recall": micro_e2e_recall,
        "micro_end_to_end_defense_f1": harmonic_mean(micro_e2e_precision, micro_e2e_recall),
        "macro_end_to_end_pipeline_precision": mean(
            float(case.get("end_to_end_pipeline_precision", 0.0)) for case in case_reports
        ),
        "macro_end_to_end_defense_recall": mean(
            float(case.get("end_to_end_defense_recall", 0.0)) for case in case_reports
        ),
        "macro_end_to_end_defense_f1": mean(
            float(case.get("end_to_end_defense_f1", 0.0)) for case in case_reports
        ),
        "judge_call_total": judge_call_total,
        "judge_parse_valid_total": judge_parse_valid_total,
        "judge_schema_valid_total": judge_schema_valid_total,
    }


def build_row(config: Dict[str, str]) -> Dict[str, Any]:
    red_path = EXPERIMENT_ROOT / config["red_eval"]
    blue_path = EXPERIMENT_ROOT / config["blue_eval"]
    red_reports = load_split_case_reports(red_path)
    blue_reports = load_split_case_reports(blue_path)

    red_case_ids = {case["use_case_id"] for case in red_reports}
    blue_case_ids = {case["use_case_id"] for case in blue_reports}
    if red_case_ids != blue_case_ids:
        missing_red = sorted(blue_case_ids - red_case_ids)[:5]
        missing_blue = sorted(red_case_ids - blue_case_ids)[:5]
        raise ValueError(
            f"Red/Blue split case mismatch for {config['setting']} {config['model']}: "
            f"missing_red={missing_red}, missing_blue={missing_blue}"
        )

    red = summarize_red(red_reports)
    blue = summarize_blue(blue_reports)
    row: Dict[str, Any] = {
        "rq": config["rq"],
        "model": config["model"],
        "setting": config["setting"],
        "split": SPLIT,
        "ablation_setting": config.get("ablation_setting", ""),
        "red_eval_json": str(red_path.relative_to(ROOT)).replace("\\", "/"),
        "blue_eval_json": str(blue_path.relative_to(ROOT)).replace("\\", "/"),
    }
    row.update(red)
    row.update(blue)
    if row["silver_threat_total_red"] != row["silver_threat_total_blue"]:
        raise ValueError(f"Silver threat total mismatch for {config['setting']} {config['model']}")

    row["case_count"] = red["red_case_count"]
    row["silver_threat_total"] = red["silver_threat_total_red"]
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {}
            for field in fieldnames:
                value = row.get(field, "")
                normalized[field] = round_float(value) if isinstance(value, float) else value
            writer.writerow(normalized)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def rq1_configs() -> List[Dict[str, str]]:
    return [
        {
            "rq": "RQ1",
            "model": "qwen3.5-plus",
            "setting": "SA-NoRAG",
            "red_eval": "sa_NoRAG_qwen35plus/saafg_task_a_eval_norag_v0_2_qwen35plus.json",
            "blue_eval": "sa_NoRAG_qwen35plus/saafg_task_b_eval_norag_v0_2_qwen35plus.json",
        },
        {
            "rq": "RQ1",
            "model": "qwen3.5-plus",
            "setting": "MA-NoRAG",
            "red_eval": "ma_NoRAG_qwen35plus/red_team/saafg_redteam_task_a_eval_norag_v0_2_qwen35plus.json",
            "blue_eval": "ma_NoRAG_qwen35plus/blue_team/saafg_blueteam_task_b_eval_norag_v0_2_qwen35plus.json",
        },
        {
            "rq": "RQ1",
            "model": "qwen3.5-plus",
            "setting": "MA-VanillaRAG",
            "red_eval": "ma_VanillaRAG_qwen35plus/red_team/saafg_redteam_task_a_eval_vanillarag_v0_2_qwen35plus.json",
            "blue_eval": "ma_VanillaRAG_qwen35plus/blue_team/saafg_blueteam_task_b_eval_vanillarag_v0_2_qwen35plus.json",
        },
        {
            "rq": "RQ1",
            "model": "qwen3.5-plus",
            "setting": "MA-SAFR",
            "red_eval": "ma_RecEvoGraphRAG_qwen35plus/red_team/saafg_redteam_task_a_eval_recevographrag_v0_2_qwen35plus.json",
            "blue_eval": "ma_RecEvoGraphRAG_qwen35plus/blue_team/saafg_blueteam_task_b_eval_recevographrag_v0_2_qwen35plus.json",
        },
        {
            "rq": "RQ1",
            "model": "deepseek-v3.2",
            "setting": "SA-NoRAG",
            "red_eval": "sa_NoRAG_deepseek-v32/saafg_task_a_eval_norag_v0_2_deepseek-v32.json",
            "blue_eval": "sa_NoRAG_deepseek-v32/saafg_task_b_eval_norag_v0_2_deepseek-v32.json",
        },
        {
            "rq": "RQ1",
            "model": "deepseek-v3.2",
            "setting": "MA-NoRAG",
            "red_eval": "ma_NoRAG_deepseek-v32/red_team/saafg_redteam_task_a_eval_norag_v0_2_deepseek-v32.json",
            "blue_eval": "ma_NoRAG_deepseek-v32/blue_team/saafg_blueteam_task_b_eval_norag_v0_2_deepseek-v32.json",
        },
        {
            "rq": "RQ1",
            "model": "deepseek-v3.2",
            "setting": "MA-VanillaRAG",
            "red_eval": "ma_VanillaRAG_deepseek-v32/red_team/saafg_redteam_task_a_eval_vanillarag_v0_2_deepseek-v32.json",
            "blue_eval": "ma_VanillaRAG_deepseek-v32/blue_team/saafg_blueteam_task_b_eval_vanillarag_v0_2_deepseek-v32.json",
        },
        {
            "rq": "RQ1",
            "model": "deepseek-v3.2",
            "setting": "MA-SAFR",
            "red_eval": "ma_RecEvoGraphRAG_deepseek-v32/red_team/saafg_redteam_task_a_eval_recevographrag_v0_2_deepseek-v32.json",
            "blue_eval": "ma_RecEvoGraphRAG_deepseek-v32/blue_team/saafg_blueteam_task_b_eval_recevographrag_v0_2_deepseek-v32.json",
        },
        {
            "rq": "RQ1",
            "model": "GPT-5.2",
            "setting": "SA-NoRAG",
            "red_eval": "sa_NoRAG_GPT52/saafg_task_a_eval_norag_v0_2_GPT52.json",
            "blue_eval": "sa_NoRAG_GPT52/saafg_task_b_eval_norag_v0_2_GPT52.json",
        },
        {
            "rq": "RQ1",
            "model": "GPT-5.2",
            "setting": "MA-NoRAG",
            "red_eval": "ma_NoRAG_GPT52/red_team/saafg_redteam_task_a_eval_norag_v0_2_GPT52.json",
            "blue_eval": "ma_NoRAG_GPT52/blue_team/saafg_blueteam_task_b_eval_norag_v0_2_GPT52.json",
        },
        {
            "rq": "RQ1",
            "model": "GPT-5.2",
            "setting": "MA-VanillaRAG",
            "red_eval": "ma_VanillaRAG_GPT52/red_team/saafg_redteam_task_a_eval_vanillarag_v0_2_GPT52.json",
            "blue_eval": "ma_VanillaRAG_GPT52/blue_team/saafg_blueteam_task_b_eval_vanillarag_v0_2_GPT52_bluererun1.json",
        },
        {
            "rq": "RQ1",
            "model": "GPT-5.2",
            "setting": "MA-SAFR",
            "red_eval": "ma_RecEvoGraphRAG_GPT52/red_team/saafg_redteam_task_a_eval_recevographrag_v0_2_GPT52.json",
            "blue_eval": "ma_RecEvoGraphRAG_GPT52/blue_team/saafg_blueteam_task_b_eval_recevographrag_v0_2_GPT52_bluererun1.json",
        },
    ]


def rq2_configs() -> List[Dict[str, str]]:
    return [
        {
            "rq": "RQ2",
            "model": "qwen3.5-plus",
            "setting": "MA-SAFR with RSSG",
            "ablation_setting": "with_RSSG",
            "red_eval": "ma_RecEvoGraphRAG_qwen35plus/red_team/saafg_redteam_task_a_eval_recevographrag_v0_2_qwen35plus.json",
            "blue_eval": "ma_RecEvoGraphRAG_qwen35plus/blue_team/saafg_blueteam_task_b_eval_recevographrag_v0_2_qwen35plus.json",
        },
        {
            "rq": "RQ2",
            "model": "qwen3.5-plus",
            "setting": "MA-SAFR without RSSG",
            "ablation_setting": "without_RSSG_generic_scaffold",
            "red_eval": "ma_RecEvoGraphRAG_without_RSSG_generic_scaffold_qwen35plus/red_team/saafg_redteam_task_a_eval_recevographrag_without_rssg_generic_scaffold_v0_2_qwen35plus.json",
            "blue_eval": "ma_RecEvoGraphRAG_without_RSSG_generic_scaffold_qwen35plus/blue_team/saafg_blueteam_task_b_eval_recevographrag_without_rssg_generic_scaffold_v0_2_qwen35plus.json",
        },
        {
            "rq": "RQ2",
            "model": "deepseek-v3.2",
            "setting": "MA-SAFR with RSSG",
            "ablation_setting": "with_RSSG",
            "red_eval": "ma_RecEvoGraphRAG_deepseek-v32/red_team/saafg_redteam_task_a_eval_recevographrag_v0_2_deepseek-v32.json",
            "blue_eval": "ma_RecEvoGraphRAG_deepseek-v32/blue_team/saafg_blueteam_task_b_eval_recevographrag_v0_2_deepseek-v32.json",
        },
        {
            "rq": "RQ2",
            "model": "deepseek-v3.2",
            "setting": "MA-SAFR without RSSG",
            "ablation_setting": "without_RSSG_generic_scaffold",
            "red_eval": "ma_RecEvoGraphRAG_without_RSSG_generic_scaffold_deepseek-v32/red_team/saafg_redteam_task_a_eval_recevographrag_without_rssg_generic_scaffold_v0_2_deepseek-v32.json",
            "blue_eval": "ma_RecEvoGraphRAG_without_RSSG_generic_scaffold_deepseek-v32/blue_team/saafg_blueteam_task_b_eval_recevographrag_without_rssg_generic_scaffold_v0_2_deepseek-v32.json",
        },
        {
            "rq": "RQ2",
            "model": "GPT-5.2",
            "setting": "MA-SAFR with RSSG",
            "ablation_setting": "with_RSSG",
            "red_eval": "ma_RecEvoGraphRAG_GPT52/red_team/saafg_redteam_task_a_eval_recevographrag_v0_2_GPT52.json",
            "blue_eval": "ma_RecEvoGraphRAG_GPT52/blue_team/saafg_blueteam_task_b_eval_recevographrag_v0_2_GPT52_bluererun1.json",
        },
        {
            "rq": "RQ2",
            "model": "GPT-5.2",
            "setting": "MA-SAFR without RSSG",
            "ablation_setting": "without_RSSG_generic_scaffold",
            "red_eval": "ma_RecEvoGraphRAG_without_RSSG_generic_scaffold_GPT52/red_team/saafg_redteam_task_a_eval_recevographrag_without_rssg_generic_scaffold_v0_2_GPT52.json",
            "blue_eval": "ma_RecEvoGraphRAG_without_RSSG_generic_scaffold_GPT52/blue_team/saafg_blueteam_task_b_eval_recevographrag_without_rssg_generic_scaffold_v0_2_GPT52.json",
        },
    ]


def common_fieldnames() -> List[str]:
    return [
        "rq",
        "model",
        "setting",
        "split",
        "ablation_setting",
        "case_count",
        "predicted_threat_total",
        "raw_predicted_threat_total",
        "silver_threat_total",
        "threat_validity_match_total",
        "task_a_valid_threat_total",
        "generated_defense_total",
        "defense_valid_total",
        "micro_threat_validity_precision",
        "micro_threat_validity_recall",
        "micro_threat_f1",
        "micro_defense_validity_precision",
        "micro_defense_validity_recall",
        "micro_end_to_end_defense_recall",
        "micro_end_to_end_pipeline_precision",
        "micro_end_to_end_defense_f1",
        "macro_threat_validity_recall",
        "macro_end_to_end_defense_recall",
        "macro_end_to_end_pipeline_precision",
        "macro_end_to_end_defense_f1",
        "red_eval_json",
        "blue_eval_json",
    ]


def build_rq1_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Setting",
        "Micro-R_threat",
        "Micro-R_e2e",
        "Micro-P_pipe",
        "Macro-R_e2e",
    ]
    md_rows = [
        [
            row["model"],
            row["setting"],
            pct(row["micro_threat_validity_recall"]),
            pct(row["micro_end_to_end_defense_recall"]),
            pct(row["micro_end_to_end_pipeline_precision"]),
            pct(row["macro_end_to_end_defense_recall"]),
        ]
        for row in rows
    ]
    return markdown_table(headers, md_rows)


def build_rq1_relative_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    output: List[Dict[str, Any]] = []
    for model, model_rows in by_model.items():
        proposed = next(row for row in model_rows if row["setting"] == "MA-SAFR")
        baselines = [row for row in model_rows if row["setting"] != "MA-SAFR"]
        rel_row: Dict[str, Any] = {
            "rq": "RQ1",
            "model": model,
            "split": SPLIT,
            "proposed_setting": "MA-SAFR",
        }
        for metric in MAIN_METRICS:
            best = max(baselines, key=lambda row: float(row[metric]))
            best_value = float(best[metric])
            proposed_value = float(proposed[metric])
            rel_row[f"best_baseline_{metric}"] = best_value
            rel_row[f"best_baseline_setting_{metric}"] = best["setting"]
            rel_row[f"proposed_{metric}"] = proposed_value
            rel_row[f"relative_change_{metric}_pct"] = (
                None if best_value == 0 else ((proposed_value - best_value) / best_value) * 100
            )
        output.append(rel_row)
    return output


def build_rq1_main_markdown(rows: Sequence[Dict[str, Any]], relative_rows: Sequence[Dict[str, Any]]) -> str:
    rel_by_model = {row["model"]: row for row in relative_rows}
    headers = [
        "Model",
        "Setting",
        "Micro-R_threat",
        "Micro-R_e2e",
        "Micro-P_pipe",
        "Macro-R_e2e",
    ]
    md_rows: List[List[str]] = []
    for row in rows:
        cells = [
            row["model"],
            row["setting"],
            pct(row["micro_threat_validity_recall"]),
            pct(row["micro_end_to_end_defense_recall"]),
            pct(row["micro_end_to_end_pipeline_precision"]),
            pct(row["macro_end_to_end_defense_recall"]),
        ]
        if row["setting"] == "MA-SAFR":
            rel = rel_by_model[row["model"]]
            cells = [
                row["model"],
                row["setting"],
                f"{pct(row['micro_threat_validity_recall'])} / {rel_pct(row['micro_threat_validity_recall'], rel['best_baseline_micro_threat_validity_recall'])}",
                f"{pct(row['micro_end_to_end_defense_recall'])} / {rel_pct(row['micro_end_to_end_defense_recall'], rel['best_baseline_micro_end_to_end_defense_recall'])}",
                f"{pct(row['micro_end_to_end_pipeline_precision'])} / {rel_pct(row['micro_end_to_end_pipeline_precision'], rel['best_baseline_micro_end_to_end_pipeline_precision'])}",
                f"{pct(row['macro_end_to_end_defense_recall'])} / {rel_pct(row['macro_end_to_end_defense_recall'], rel['best_baseline_macro_end_to_end_defense_recall'])}",
            ]
        md_rows.append(cells)
    return markdown_table(headers, md_rows)


def build_relative_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["ablation_setting"]] = row

    output: List[Dict[str, Any]] = []
    for model, grouped in by_model.items():
        with_row = grouped["with_RSSG"]
        without_row = grouped["without_RSSG_generic_scaffold"]
        rel_row: Dict[str, Any] = {
            "rq": "RQ2",
            "model": model,
            "split": SPLIT,
            "baseline_setting": "MA-SAFR without RSSG",
            "proposed_setting": "MA-SAFR with RSSG",
        }
        for metric in MAIN_METRICS:
            base = float(without_row[metric])
            proposed = float(with_row[metric])
            rel_row[f"without_{metric}"] = base
            rel_row[f"with_{metric}"] = proposed
            rel_row[f"relative_change_{metric}_pct"] = None if base == 0 else ((proposed - base) / base) * 100
        output.append(rel_row)
    return output


def build_rq2_markdown(relative_rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Micro-R_threat",
        "Micro-R_e2e",
        "Micro-P_pipe",
        "Macro-R_e2e",
    ]
    md_rows = []
    for row in relative_rows:
        md_rows.append(
            [
                row["model"],
                rel_pct(
                    row["with_micro_threat_validity_recall"],
                    row["without_micro_threat_validity_recall"],
                ),
                rel_pct(
                    row["with_micro_end_to_end_defense_recall"],
                    row["without_micro_end_to_end_defense_recall"],
                ),
                rel_pct(
                    row["with_micro_end_to_end_pipeline_precision"],
                    row["without_micro_end_to_end_pipeline_precision"],
                ),
                rel_pct(
                    row["with_macro_end_to_end_defense_recall"],
                    row["without_macro_end_to_end_defense_recall"],
                ),
            ]
        )
    return markdown_table(headers, md_rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rq1_rows = [build_row(config) for config in rq1_configs()]
    rq2_rows = [build_row(config) for config in rq2_configs()]
    rq1_relative_rows = build_rq1_relative_rows(rq1_rows)
    rq2_relative_rows = build_relative_rows(rq2_rows)

    write_csv(OUTPUT_ROOT / "RQ1_test_split_metrics.csv", rq1_rows, common_fieldnames())
    write_csv(OUTPUT_ROOT / "RQ2_test_split_metrics.csv", rq2_rows, common_fieldnames())

    relative_fields = [
        "rq",
        "model",
        "split",
        "baseline_setting",
        "proposed_setting",
    ]
    for metric in MAIN_METRICS:
        relative_fields.extend(
            [
                f"without_{metric}",
                f"with_{metric}",
                f"relative_change_{metric}_pct",
            ]
        )
    write_csv(
        OUTPUT_ROOT / "RQ2_test_split_relative_changes.csv",
        rq2_relative_rows,
        relative_fields,
    )

    rq1_relative_fields = [
        "rq",
        "model",
        "split",
        "proposed_setting",
    ]
    for metric in MAIN_METRICS:
        rq1_relative_fields.extend(
            [
                f"best_baseline_setting_{metric}",
                f"best_baseline_{metric}",
                f"proposed_{metric}",
                f"relative_change_{metric}_pct",
            ]
        )
    write_csv(
        OUTPUT_ROOT / "RQ1_test_split_relative_to_best_baseline.csv",
        rq1_relative_rows,
        rq1_relative_fields,
    )

    (OUTPUT_ROOT / "RQ1_test_split_metrics.md").write_text(
        "# RQ1 Test Split Metrics\n\n" + build_rq1_markdown(rq1_rows),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "RQ1_test_split_main_table.md").write_text(
        "# RQ1 Test Split Main Table\n\n" + build_rq1_main_markdown(rq1_rows, rq1_relative_rows),
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "RQ2_test_split_relative_changes.md").write_text(
        "# RQ2 Test Split Relative Changes\n\n" + build_rq2_markdown(rq2_relative_rows),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": now_utc(),
        "split": SPLIT,
        "notes": [
            "Metrics are recomputed from case_reports after filtering split == test.",
            "RQ2 uses without_RSSG_generic_scaffold as the RSSG ablation baseline.",
            "Relative changes are computed as (with_RSSG - without_RSSG) / without_RSSG * 100.",
        ],
        "rq1_configs": rq1_configs(),
        "rq2_configs": rq2_configs(),
        "outputs": [
            "RQ1_test_split_metrics.csv",
            "RQ1_test_split_metrics.md",
            "RQ1_test_split_relative_to_best_baseline.csv",
            "RQ1_test_split_main_table.md",
            "RQ2_test_split_metrics.csv",
            "RQ2_test_split_relative_changes.csv",
            "RQ2_test_split_relative_changes.md",
        ],
    }
    write_json(OUTPUT_ROOT / "aggregation_manifest.json", manifest)

    print(f"[Done] Wrote test split aggregation to {OUTPUT_ROOT}")
    print(f"[Summary] RQ1 rows={len(rq1_rows)} RQ2 rows={len(rq2_rows)}")


if __name__ == "__main__":
    main()
