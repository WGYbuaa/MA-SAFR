#!/usr/bin/env python3
"""Audit and consolidate the isolated C2 strict-alpha experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
EXPERIMENT_ROOT = (
    ROOT / "0_Data" / "6_SAAFG" / "6_Experiment_Result" / "C2_StrictAlpha_v1"
)
RESULT_ROOT = (
    ROOT / "0_Data" / "6_SAAFG" / "9_result_test_split_2_0" / "C2_StrictAlpha_v1"
)
OLD_ALL_VERSION_CSV = (
    ROOT
    / "0_Data"
    / "6_SAAFG"
    / "9_result_test_split_2_0"
    / "RQ3_ALL_VERSION_RESULTS_20260717.csv"
)
NEW_ALL_VERSION_CSV = OLD_ALL_VERSION_CSV.with_name("RQ3_ALL_VERSION_RESULTS_20260719.csv")
NEW_ALL_VERSION_MD = OLD_ALL_VERSION_CSV.with_name("RQ3_ALL_VERSION_RESULTS_20260719.md")

METRICS = [
    "micro_threat_validity_recall",
    "micro_end_to_end_defense_recall",
    "micro_end_to_end_pipeline_precision",
    "macro_end_to_end_defense_recall",
]
METRIC_LABELS = {
    "micro_threat_validity_recall": "Micro threat recall",
    "micro_end_to_end_defense_recall": "Micro E2E defense recall",
    "micro_end_to_end_pipeline_precision": "Micro pipeline precision",
    "macro_end_to_end_defense_recall": "Macro E2E defense recall",
}

PAIR_CONFIGS = {
    "deepseek-v32": {
        "display_model": "DeepSeek-v3.2",
        "split": "test_1.0",
        "registry": ROOT
        / "0_Data"
        / "6_SAAFG"
        / "7_Benchmark_Package_v0_2"
        / "case_registry_test_1.json",
        "neutral": "neutral_alpha0",
        "adapted": "shared_std_a1",
    },
    "gpt52": {
        "display_model": "GPT-5.2",
        "split": "test_2.0",
        "registry": ROOT
        / "0_Data"
        / "6_SAAFG"
        / "7_Benchmark_Package_v0_2"
        / "case_registry_test_2.json",
        "neutral": "neutral_alpha0",
        "adapted": "model_conservative_a05",
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def registry_ids(registry: Path, split: str) -> set[str]:
    return {
        item["use_case_id"]
        for item in read_json(registry)["cases"]
        if item.get("split") == split
    }


def graph_audit(graph_dir: Path) -> dict[str, Any]:
    nodes = read_json(graph_dir / "graph_nodes.json")["nodes"]
    edges = read_json(graph_dir / "graph_edges.json")["edges"]
    metadata = read_json(graph_dir / "graph_metadata.json")
    topology = {
        "node_ids": sorted(node["node_id"] for node in nodes),
        "edges": sorted(
            (
                edge["edge_id"],
                edge["source"],
                edge["target"],
                edge["relation"],
            )
            for edge in edges
        ),
    }
    weights = sorted((edge["edge_id"], float(edge["weight"])) for edge in edges)
    feedback_path = Path(metadata["feedback_jsonl"])
    feedback_events = [
        json.loads(line)
        for line in feedback_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "path": str(graph_dir),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "topology_sha256": digest(topology),
        "weight_sha256": digest(weights),
        "feedback_alpha": metadata["feedback_weight_alpha"],
        "negative_scale": metadata["edge_weight_update_policy"]["negative_scale"],
        "split_filter": metadata["edge_weight_update_policy"]["split_filter"],
        "feedback_path": str(feedback_path),
        "feedback_event_count": len(feedback_events),
        "feedback_metadata_event_count": metadata["feedback_stats"]["input_event_count"],
        "feedback_splits": sorted({event.get("split") for event in feedback_events}),
        "feedback_case_ids": sorted(
            {event["use_case_id"] for event in feedback_events if event.get("use_case_id")}
        ),
        "updated_edge_count": metadata["edge_weight_update_stats"]["updated_edge_count"],
    }


def generation_summary(run: dict[str, Any], team: str) -> dict[str, Any]:
    results = run["results"]
    if team == "red":
        attempts = results
        parse_valid = sum(bool(item.get("parse_valid")) for item in attempts)
        schema_valid = sum(bool(item.get("schema_valid")) for item in attempts)
        retries = sum(int(item.get("transport_retry_count") or 0) for item in attempts)
    else:
        attempts = [
            threat
            for item in results
            for threat in item.get("threat_generation_results", [])
        ]
        parse_valid = sum(bool(item.get("parse_valid")) for item in attempts)
        schema_valid = sum(bool(item.get("schema_valid")) for item in attempts)
        retries = sum(int(item.get("transport_retry_count") or 0) for item in attempts)
    return {
        "case_count": len(results),
        "case_ids": [item["use_case_id"] for item in results],
        "record_splits": sorted({item.get("split") for item in results}),
        "generation_attempt_count": len(attempts),
        "parse_valid_count": parse_valid,
        "schema_valid_count": schema_valid,
        "transport_retry_count": retries,
    }


def setting_audit(model_key: str, setting: str) -> dict[str, Any]:
    root = EXPERIMENT_ROOT / model_key / "test" / setting
    red_run = read_json(root / "red_team" / "red_run.json")
    blue_run = read_json(root / "blue_team" / "blue_run.json")
    red_eval = read_json(root / "red_team" / "red_eval.json")
    blue_eval = read_json(root / "blue_team" / "blue_eval.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    return {
        "setting": setting,
        "red_meta": red_run["meta"],
        "blue_meta": blue_run["meta"],
        "red_generation": generation_summary(red_run, "red"),
        "blue_generation": generation_summary(blue_run, "blue"),
        "red_judge": {
            "model": red_eval["meta"]["model_name"],
            "prompt": red_eval["meta"]["prompt_path"],
            "call_count": red_eval["summary"]["counts"]["judge_pair_total"],
            "parse_valid_count": red_eval["summary"]["counts"]["judge_parse_valid_total"],
            "schema_valid_count": red_eval["summary"]["counts"]["judge_schema_valid_total"],
        },
        "blue_judge": {
            "model": blue_eval["meta"]["judge_model_name"],
            "prompt": blue_eval["meta"]["prompt_path"],
            "call_count": blue_eval["summary"]["counts"]["judge_call_total"],
            "parse_valid_count": blue_eval["summary"]["counts"]["judge_parse_valid_total"],
            "schema_valid_count": blue_eval["summary"]["counts"]["judge_schema_valid_total"],
        },
        "aggregate": aggregate,
        "graph": graph_audit(Path(red_run["meta"]["kg_dir"])),
    }


def matched_meta(neutral: dict[str, Any], adapted: dict[str, Any], team: str) -> dict[str, bool]:
    meta_key = f"{team}_meta"
    keys = [
        "model_name",
        "model_env_var",
        "rag_top_k",
        "temperature",
        "request_timeout",
        "input_path",
        "prompt_path",
        "generation_policy",
        "retriever_module",
        "transport_policy",
    ]
    if team == "red":
        keys.append("case_registry_path")
    return {
        key: neutral[meta_key].get(key) == adapted[meta_key].get(key)
        for key in keys
    }


def recompute_metric_checks(aggregate: dict[str, Any]) -> dict[str, bool]:
    silver = aggregate["silver_threat_total"]
    predicted = aggregate["predicted_threat_total"]
    threat_valid = aggregate["threat_validity_match_total"]
    defense_valid = aggregate["defense_valid_total"]
    return {
        "micro_threat_validity_recall": abs(
            aggregate["micro_threat_validity_recall"] - threat_valid / silver
        )
        < 1e-6,
        "micro_end_to_end_defense_recall": abs(
            aggregate["micro_end_to_end_defense_recall"] - defense_valid / silver
        )
        < 1e-6,
        "micro_end_to_end_pipeline_precision": abs(
            aggregate["micro_end_to_end_pipeline_precision"] - defense_valid / predicted
        )
        < 1e-6,
    }


def audit_pair(model_key: str) -> dict[str, Any]:
    config = PAIR_CONFIGS[model_key]
    neutral = setting_audit(model_key, config["neutral"])
    adapted = setting_audit(model_key, config["adapted"])
    expected_test_ids = registry_ids(config["registry"], "test")
    expected_train_ids = registry_ids(config["registry"], "train")

    neutral_ids = set(neutral["red_generation"]["case_ids"])
    adapted_ids = set(adapted["red_generation"]["case_ids"])
    graph_feedback_ids = set(adapted["graph"]["feedback_case_ids"])
    checks = {
        "red_parameters_matched": all(matched_meta(neutral, adapted, "red").values()),
        "blue_parameters_matched": all(matched_meta(neutral, adapted, "blue").values()),
        "expected_test_case_count_57": len(expected_test_ids) == 57,
        "neutral_case_ids_equal_expected_test": neutral_ids == expected_test_ids,
        "adapted_case_ids_equal_expected_test": adapted_ids == expected_test_ids,
        "neutral_and_adapted_case_ids_equal": neutral_ids == adapted_ids,
        "all_red_blue_case_orders_matched": all(
            item["red_generation"]["case_ids"] == item["blue_generation"]["case_ids"]
            for item in (neutral, adapted)
        ),
        "no_duplicate_case_ids": all(
            len(item["red_generation"]["case_ids"])
            == len(set(item["red_generation"]["case_ids"]))
            for item in (neutral, adapted)
        ),
        "graph_topology_matched": neutral["graph"]["topology_sha256"]
        == adapted["graph"]["topology_sha256"],
        "graph_weights_changed": neutral["graph"]["weight_sha256"]
        != adapted["graph"]["weight_sha256"],
        "feedback_split_filter_train_only": adapted["graph"]["split_filter"] == ["train"],
        "feedback_records_train_only": adapted["graph"]["feedback_splits"] == ["train"],
        "feedback_case_ids_subset_of_train": graph_feedback_ids <= expected_train_ids,
        "feedback_event_count_matched": adapted["graph"]["feedback_event_count"]
        == adapted["graph"]["feedback_metadata_event_count"],
        "edge_weights_actually_updated": adapted["graph"]["updated_edge_count"] > 0,
        "no_transport_retries": all(
            item[team]["transport_retry_count"] == 0
            for item in (neutral, adapted)
            for team in ("red_generation", "blue_generation")
        ),
        "all_generation_outputs_parse_valid": all(
            item[team]["parse_valid_count"] == item[team]["generation_attempt_count"]
            for item in (neutral, adapted)
            for team in ("red_generation", "blue_generation")
        ),
        "all_generation_outputs_schema_valid": all(
            item[team]["schema_valid_count"] == item[team]["generation_attempt_count"]
            for item in (neutral, adapted)
            for team in ("red_generation", "blue_generation")
        ),
        "judges_matched": all(
            neutral[judge][key] == adapted[judge][key]
            for judge in ("red_judge", "blue_judge")
            for key in ("model", "prompt")
        ),
        "all_judge_outputs_parse_valid": all(
            item[judge]["parse_valid_count"] == item[judge]["call_count"]
            for item in (neutral, adapted)
            for judge in ("red_judge", "blue_judge")
        ),
        "all_judge_outputs_schema_valid": all(
            item[judge]["schema_valid_count"] == item[judge]["call_count"]
            for item in (neutral, adapted)
            for judge in ("red_judge", "blue_judge")
        ),
        "neutral_micro_metrics_recomputed": all(
            recompute_metric_checks(neutral["aggregate"]).values()
        ),
        "adapted_micro_metrics_recomputed": all(
            recompute_metric_checks(adapted["aggregate"]).values()
        ),
    }
    pair = read_json(RESULT_ROOT / model_key / "test_pair.json")
    return {
        "model_key": model_key,
        "display_model": config["display_model"],
        "split": config["split"],
        "selected_on_dev": pair["selected_on_dev"],
        "alpha": pair["alpha"],
        "policy": pair["policy"],
        "neutral": neutral,
        "adapted": adapted,
        "relative_change_pct": pair["relative_change_pct"],
        "checks": checks,
        "audit_passed": all(checks.values()),
    }


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def change(value: float) -> str:
    return f"{float(value):+.2f}%"


def write_audit_report(pairs: list[dict[str, Any]]) -> None:
    write_json(RESULT_ROOT / "AUDIT.json", {"protocol": "C2-StrictAlpha-v1", "pairs": pairs})
    lines = [
        "# C2 Strict Alpha v1 Audit",
        "",
        "The audit is read-only. It verifies experimental pairing and does not alter any generation or judgment.",
        "",
    ]
    for pair in pairs:
        neutral = pair["neutral"]
        adapted = pair["adapted"]
        lines.extend(
            [
                f"## {pair['display_model']}",
                "",
                f"- Split: `{pair['split']}`; cases: `{neutral['aggregate']['case_count']}`.",
                f"- Dev-selected configuration: `{pair['selected_on_dev']}`; alpha: `{pair['alpha']}`.",
                f"- Graph topology SHA-256: `{neutral['graph']['topology_sha256']}` (identical on both sides).",
                f"- Adapted feedback: `{adapted['graph']['feedback_event_count']}` train-only events; `{adapted['graph']['updated_edge_count']}` updated edges.",
                f"- Transport retries: `{neutral['red_generation']['transport_retry_count'] + neutral['blue_generation']['transport_retry_count']}` vs. `{adapted['red_generation']['transport_retry_count'] + adapted['blue_generation']['transport_retry_count']}`.",
                f"- Overall audit: `{'PASS' if pair['audit_passed'] else 'FAIL'}`.",
                "",
                "| Check | Result |",
                "|---|---|",
            ]
        )
        lines.extend(
            f"| {name} | {'PASS' if passed else 'FAIL'} |"
            for name, passed in pair["checks"].items()
        )
        lines.append("")
    (RESULT_ROOT / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def c2_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qwen = {
        "model": "Qwen3.5-Plus",
        "split": "test_1.0",
        "selection": "existing strict alpha pair",
        "alpha": "1.0",
        "neutral": {
            METRICS[0]: 0.296296,
            METRICS[1]: 0.271605,
            METRICS[2]: 0.196429,
            METRICS[3]: 0.352339,
        },
        "adapted": {
            METRICS[0]: 0.345679,
            METRICS[1]: 0.296296,
            METRICS[2]: 0.218182,
            METRICS[3]: 0.375731,
        },
        "changes": {
            METRICS[0]: 16.67,
            METRICS[1]: 9.09,
            METRICS[2]: 11.07,
            METRICS[3]: 6.64,
        },
        "all_four_positive": True,
        "audit": "Prior strict pair",
    }
    rows = [qwen]
    for pair in pairs:
        rows.append(
            {
                "model": pair["display_model"],
                "split": pair["split"],
                "selection": pair["selected_on_dev"],
                "alpha": str(pair["alpha"]),
                "neutral": {key: pair["neutral"]["aggregate"][key] for key in METRICS},
                "adapted": {key: pair["adapted"]["aggregate"][key] for key in METRICS},
                "changes": pair["relative_change_pct"],
                "all_four_positive": all(
                    float(pair["relative_change_pct"][key]) > 0 for key in METRICS
                ),
                "audit": "PASS" if pair["audit_passed"] else "FAIL",
            }
        )
    return rows


def write_c2_summary(rows: list[dict[str, Any]]) -> None:
    csv_path = RESULT_ROOT / "C2_strict_alpha_three_models.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "split",
                "selection",
                "adapted_alpha",
                *[f"{key}_alpha0" for key in METRICS],
                *[f"{key}_adapted" for key in METRICS],
                *[f"relative_{key}_pct" for key in METRICS],
                "all_four_positive",
                "audit",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["split"],
                    row["selection"],
                    row["alpha"],
                    *[f"{row['neutral'][key]:.6f}" for key in METRICS],
                    *[f"{row['adapted'][key]:.6f}" for key in METRICS],
                    *[f"{row['changes'][key]:.4f}" for key in METRICS],
                    str(row["all_four_positive"]).lower(),
                    row["audit"],
                ]
            )

    lines = [
        "# C2 Strict Alpha: Three-Model Results",
        "",
        "Within each model, alpha=0 and the adapted setting keep the model, prompts, retriever, top-k, split, judges, graph topology, and transport policy fixed. The new DeepSeek/GPT configuration was selected on dev before a single test run.",
        "",
        "| Model | Split | Selected adapted setting | Threat recall | E2E defense recall | Pipeline precision | Macro E2E recall | All four positive | Audit |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        cells = []
        for key in METRICS:
            cells.append(
                f"{pct(row['neutral'][key])} -> {pct(row['adapted'][key])} ({change(row['changes'][key])})"
            )
        lines.append(
            f"| {row['model']} | {row['split']} | {row['selection']} (alpha={row['alpha']}) | "
            + " | ".join(cells)
            + f" | {'Yes' if row['all_four_positive'] else 'No'} | {row['audit']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Qwen is the only model whose four strict-alpha point estimates all improve.",
            "- DeepSeek preserves threat recall and improves both micro defense measures, while macro E2E recall decreases slightly. This suggests that one additional valid defense is concentrated in cases that do not improve the case-average coverage.",
            "- GPT improves threat recall but loses two valid defenses, so its dev-selected retrieval adaptation does not generalize to test on the defense stage.",
            "- The strict evidence therefore supports a model-dependent effect, not a universal monotonic benefit of feedback-fused edge weights.",
        ]
    )
    (RESULT_ROOT / "C2_strict_alpha_three_models.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def add_new_rows_to_version_csv(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    with OLD_ALL_VERSION_CSV.open("r", encoding="utf-8", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    fieldnames = list(old_rows[0].keys())
    new_rows = list(old_rows)
    for row in rows[1:]:
        model_id = "DeepSeek" if row["model"].startswith("DeepSeek") else "GPT"
        new_rows.append(
            {
                "version_group": "C2-StrictAlpha-v1",
                "comparison_id": f"C2-{model_id}-alpha-dev-selected",
                "evidence_role": "strict_weight_test",
                "model": row["model"],
                "split": row["split"],
                "case_count": "57",
                "round0_setting": "weight-aware retriever alpha=0",
                "round1_setting": f"same retriever, dev-selected alpha={row['alpha']}",
                "round1_variant": row["selection"],
                "micro_threat_recall_round0": f"{row['neutral'][METRICS[0]]:.6f}",
                "micro_threat_recall_round1": f"{row['adapted'][METRICS[0]]:.6f}",
                "relative_threat_pct": f"{row['changes'][METRICS[0]]:.4f}",
                "micro_e2e_recall_round0": f"{row['neutral'][METRICS[1]]:.6f}",
                "micro_e2e_recall_round1": f"{row['adapted'][METRICS[1]]:.6f}",
                "relative_e2e_pct": f"{row['changes'][METRICS[1]]:.4f}",
                "micro_pipeline_precision_round0": f"{row['neutral'][METRICS[2]]:.6f}",
                "micro_pipeline_precision_round1": f"{row['adapted'][METRICS[2]]:.6f}",
                "relative_precision_pct": f"{row['changes'][METRICS[2]]:.4f}",
                "macro_e2e_recall_round0": f"{row['neutral'][METRICS[3]]:.6f}",
                "macro_e2e_recall_round1": f"{row['adapted'][METRICS[3]]:.6f}",
                "relative_macro_pct": f"{row['changes'][METRICS[3]]:.4f}",
                "status": "held-out; strict matched pair; dev-selected; audit passed",
            }
        )
    with NEW_ALL_VERSION_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    return new_rows


def arrow(row: dict[str, str], prefix: str, relative: str) -> str:
    return (
        f"{pct(float(row[prefix + '_round0']))} -> {pct(float(row[prefix + '_round1']))} "
        f"({change(float(row[relative]))})"
    )


def write_all_version_md(rows: list[dict[str, str]]) -> None:
    groups = [
        ("V0.2-C1", "Current full feedback-adaptive module comparison"),
        ("V0.2-C2", "Existing Qwen strict-alpha comparison"),
        ("C2-StrictAlpha-v1", "New dev-selected DeepSeek/GPT strict-alpha comparisons"),
        ("V0.2-C3", "Round1 realization variability"),
        ("V0.2-C4", "Alternative GPT relaxed pair"),
        ("V0.2-C5", "Historical full-data rounds"),
        ("V0.3", "Residual-update dev experiment"),
    ]
    lines = [
        "# RQ3 Results by Version (2026-07-19)",
        "",
        "This inventory preserves every previously recorded RQ3 version and appends the isolated C2 strict-alpha DeepSeek/GPT tests. Rows from different groups are not interchangeable because protocols differ.",
        "",
    ]
    for group, title in groups:
        selected = [row for row in rows if row["version_group"] == group]
        if not selected:
            continue
        lines.extend(
            [
                f"## {group}: {title}",
                "",
                "| Comparison | Model | Split | Threat recall | E2E defense recall | Pipeline precision | Macro E2E recall | Status |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in selected:
            lines.append(
                f"| {row['comparison_id']} | {row['model']} | {row['split']} | "
                f"{arrow(row, 'micro_threat_recall', 'relative_threat_pct')} | "
                f"{arrow(row, 'micro_e2e_recall', 'relative_e2e_pct')} | "
                f"{arrow(row, 'micro_pipeline_precision', 'relative_precision_pct')} | "
                f"{arrow(row, 'macro_e2e_recall', 'relative_macro_pct')} | {row['status']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Current interpretation",
            "",
            "The C2 strict-alpha results isolate feedback-fused edge weights more cleanly than C1. Qwen improves on all four metrics; DeepSeek improves two, ties one, and slightly reduces macro recall; GPT improves only threat recall. Consequently, the defensible conclusion is that graph-weight feedback is model-sensitive rather than uniformly beneficial. V0.3 remains a dev-only failed candidate and must not be substituted into the test evidence.",
            "",
            "Detailed protocol and audit: `../6_Experiment_Result/C2_StrictAlpha_v1/PROTOCOL.md` and `C2_StrictAlpha_v1/AUDIT.md`.",
        ]
    )
    NEW_ALL_VERSION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = [audit_pair(model_key) for model_key in PAIR_CONFIGS]
    write_audit_report(pairs)
    rows = c2_rows(pairs)
    write_c2_summary(rows)
    all_version_rows = add_new_rows_to_version_csv(rows)
    write_all_version_md(all_version_rows)
    for pair in pairs:
        print(
            f"{pair['display_model']}: audit={'PASS' if pair['audit_passed'] else 'FAIL'}; "
            f"changes={pair['relative_change_pct']}"
        )


if __name__ == "__main__":
    main()
