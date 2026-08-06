#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Summarize RecEvoGraphRAG feedback_weight_alpha sensitivity outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from edge_weight_common import (
    DEFAULT_ALPHAS,
    DEFAULT_ALPHA_KG_ROOT,
    DEFAULT_EXPERIMENT_ROOT,
    alpha_label,
    alpha_tag,
    experiment_dir,
    feedback_alpha_kg_dir,
    validate_alphas,
)


DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "feedback_weight_alpha_summary"
DEFAULT_OUTPUT_PREFIX = "saafg_recevographrag_feedback_alpha_summary_v0_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RecEvoGraphRAG feedback_weight_alpha outputs.")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--run-tags", nargs="+", default=["qwen35plus"])
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--kg-root", type=Path, default=DEFAULT_ALPHA_KG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(payload, dict):
            yield payload


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def flatten_dict(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            flatten_dict(f"{prefix}_{key}" if prefix else str(key), child, out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def trace_stats(path: Path) -> Dict[str, Any]:
    records = list(iter_jsonl(path))
    retrieval_counts: List[float] = []
    top1_scores: List[float] = []
    item_scores: List[float] = []
    breakdown_values: Dict[str, List[float]] = {}

    for record in records:
        items = record.get("retrieved_knowledge") or []
        retrieval_count = safe_float(record.get("retrieval_count"))
        if retrieval_count is None:
            retrieval_count = float(len(items))
        retrieval_counts.append(retrieval_count)

        if items:
            top1 = safe_float(items[0].get("score"))
            if top1 is not None:
                top1_scores.append(top1)
        for item in items:
            score = safe_float(item.get("score"))
            if score is not None:
                item_scores.append(score)
            score_breakdown = item.get("score_breakdown") or {}
            if isinstance(score_breakdown, dict):
                for key, raw_value in score_breakdown.items():
                    number = safe_float(raw_value)
                    if number is not None:
                        breakdown_values.setdefault(str(key), []).append(number)

    stats: Dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
        "trace_record_count": len(records),
        "mean_retrieval_count": mean(retrieval_counts),
        "mean_top1_score": mean(top1_scores),
        "mean_item_score": mean(item_scores),
    }
    for key, values in sorted(breakdown_values.items()):
        stats[f"mean_score_breakdown_{key}"] = mean(values)
    return stats


def alpha_paths(alpha: float, run_tag: str, experiment_root: Path) -> Dict[str, Path]:
    label = alpha_tag(alpha)
    red_dir = experiment_root / f"ma_RecEvoGraphRAG_{label}_{run_tag}" / "red_team"
    blue_dir = experiment_root / f"ma_RecEvoGraphRAG_{label}_{run_tag}" / "blue_team"
    return {
        "red_run": red_dir / f"saafg_redteam_recevographrag_{label}_v0_2_{run_tag}.json",
        "red_artifact": red_dir / f"saafg_threat_records_pred_recevographrag_{label}_v0_2_{run_tag}.json",
        "red_eval": red_dir / f"saafg_redteam_task_a_eval_recevographrag_{label}_v0_2_{run_tag}.json",
        "red_trace": red_dir / f"saafg_redteam_retrieval_trace_recevographrag_{label}_v0_2_{run_tag}.jsonl",
        "blue_run": blue_dir / f"saafg_blueteam_recevographrag_{label}_v0_2_{run_tag}.json",
        "blue_artifact": blue_dir / f"saafg_security_augmented_flows_pred_recevographrag_{label}_v0_2_{run_tag}.json",
        "blue_eval": blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_{label}_v0_2_{run_tag}.json",
        "blue_trace": blue_dir / f"saafg_blueteam_retrieval_trace_recevographrag_{label}_v0_2_{run_tag}.jsonl",
    }


def summarize_one(alpha: float, run_tag: str, args: argparse.Namespace) -> Dict[str, Any]:
    paths = alpha_paths(alpha, run_tag, args.experiment_root)
    kg_dir = args.kg_root / alpha_label(alpha)
    kg_metadata = read_json_if_exists(kg_dir / "graph_metadata.json")
    red_run = read_json_if_exists(paths["red_run"])
    blue_run = read_json_if_exists(paths["blue_run"])
    red_eval = read_json_if_exists(paths["red_eval"])
    blue_eval = read_json_if_exists(paths["blue_eval"])

    row: Dict[str, Any] = {
        "feedback_weight_alpha": alpha,
        "feedback_weight_alpha_label": alpha_tag(alpha),
        "run_tag": run_tag,
        "kg_dir": str(kg_dir),
    }
    for name, path in sorted(paths.items()):
        row[f"{name}_exists"] = path.exists()
        row[f"{name}_path"] = str(path)

    flatten_dict("kg_feedback_stats", kg_metadata.get("feedback_stats") or {}, row)
    flatten_dict("kg_edge_weight_update_stats", kg_metadata.get("edge_weight_update_stats") or {}, row)
    flatten_dict("red_run_summary", red_run.get("summary") or {}, row)
    flatten_dict("blue_run_summary", blue_run.get("summary") or {}, row)
    flatten_dict("red_eval_summary", red_eval.get("summary") or {}, row)
    flatten_dict("blue_eval_summary", blue_eval.get("summary") or {}, row)
    flatten_dict("red_retrieval", trace_stats(paths["red_trace"]), row)
    flatten_dict("blue_retrieval", trace_stats(paths["blue_trace"]), row)

    required = ["red_run", "red_eval", "blue_run", "blue_eval"]
    missing = [name for name in required if not paths[name].exists()]
    row["status"] = "complete" if not missing else "missing:" + ",".join(missing)
    return row


def collect_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    alphas = validate_alphas(args.alphas)
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        for run_tag in args.run_tags:
            rows.append(summarize_one(alpha, run_tag, args))
    return rows


def ordered_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = [
        "feedback_weight_alpha",
        "feedback_weight_alpha_label",
        "run_tag",
        "status",
        "red_eval_summary_macro_threat_f1",
        "red_eval_summary_micro_threat_f1",
        "red_eval_summary_macro_threat_validity_precision",
        "red_eval_summary_macro_threat_validity_recall",
        "blue_eval_summary_macro_defense_f1",
        "blue_eval_summary_micro_defense_f1",
        "blue_eval_summary_micro_end_to_end_defense_f1",
        "blue_eval_summary_micro_end_to_end_defense_recall",
        "kg_edge_weight_update_stats_updated_edge_count",
        "kg_edge_weight_update_stats_mean_abs_alpha_delta",
        "red_retrieval_mean_retrieval_count",
        "red_retrieval_mean_score_breakdown_evo_edge_weight_score",
        "blue_retrieval_mean_retrieval_count",
        "blue_retrieval_mean_score_breakdown_evo_edge_weight_score",
    ]
    keys = sorted({key for row in rows for key in row})
    return [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = ordered_fieldnames(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    columns = [
        "feedback_weight_alpha",
        "run_tag",
        "status",
        "red_eval_summary_micro_threat_f1",
        "blue_eval_summary_micro_defense_f1",
        "blue_eval_summary_micro_end_to_end_defense_f1",
        "kg_edge_weight_update_stats_mean_abs_alpha_delta",
        "red_retrieval_mean_score_breakdown_evo_edge_weight_score",
        "blue_retrieval_mean_score_breakdown_evo_edge_weight_score",
    ]
    available = [column for column in columns if any(column in row for row in rows)]
    lines = [
        "# RecEvoGraphRAG Feedback Weight Alpha Summary",
        "",
        f"- alpha values: {', '.join(format_value(row['feedback_weight_alpha']) for row in rows)}",
        "",
    ]
    if not rows:
        lines.append("No rows found.")
    else:
        lines.append("| " + " | ".join(available) + " |")
        lines.append("| " + " | ".join("---" for _ in available) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_value(row.get(column)) for column in available) + " |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "output_prefix": args.output_prefix,
            "experiment_root": str(args.experiment_root),
            "kg_root": str(args.kg_root),
            "run_tags": args.run_tags,
            "alphas": validate_alphas(args.alphas),
        },
        "rows": rows,
    }
    json_path = args.output_dir / f"{args.output_prefix}.json"
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    md_path = args.output_dir / f"{args.output_prefix}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"[Done] rows={len(rows)}")
    print(f"[Done] json={json_path}")
    print(f"[Done] csv={csv_path}")
    print(f"[Done] md={md_path}")


if __name__ == "__main__":
    main()
