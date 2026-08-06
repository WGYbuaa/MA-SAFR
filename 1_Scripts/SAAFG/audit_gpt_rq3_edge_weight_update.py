#!/usr/bin/env python3
"""Audit the isolated GPT-5.2 shared-graph strict-alpha pair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
EXPERIMENT_ROOT = (
    ROOT
    / "0_Data"
    / "6_SAAFG"
    / "6_Experiment_Result"
    / "C2_StrictAlpha_GPT_SharedStandard_v1"
    / "gpt52"
    / "test"
)
RESULT_ROOT = (
    ROOT
    / "0_Data"
    / "6_SAAFG"
    / "9_result_test_split_2_0"
    / "C2_StrictAlpha_GPT_SharedStandard_v1"
)
REGISTRY = (
    ROOT
    / "0_Data"
    / "6_SAAFG"
    / "7_Benchmark_Package_v0_2"
    / "case_registry_test_1.json"
)
NEUTRAL = "neutral_shared_standard_alpha0"
ADAPTED = "adapted_shared_standard_alpha1"
METRICS = [
    "micro_threat_validity_recall",
    "micro_end_to_end_defense_recall",
    "micro_end_to_end_pipeline_precision",
    "macro_end_to_end_defense_recall",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def digest(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def graph_summary(graph_dir: Path) -> dict[str, Any]:
    nodes = read_json(graph_dir / "graph_nodes.json")["nodes"]
    edges = read_json(graph_dir / "graph_edges.json")["edges"]
    metadata = read_json(graph_dir / "graph_metadata.json")
    feedback = read_jsonl(Path(metadata["feedback_jsonl"]))
    topology = {
        "nodes": sorted(node["node_id"] for node in nodes),
        "edges": sorted(
            (edge["edge_id"], edge["source"], edge["target"], edge["relation"])
            for edge in edges
        ),
    }
    weights = sorted((edge["edge_id"], float(edge["weight"])) for edge in edges)
    return {
        "path": str(graph_dir),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "topology_sha256": digest(topology),
        "weight_sha256": digest(weights),
        "feedback_alpha": metadata["feedback_weight_alpha"],
        "split_filter": metadata["edge_weight_update_policy"]["split_filter"],
        "feedback_event_count": len(feedback),
        "metadata_feedback_event_count": metadata["feedback_stats"][
            "input_event_count"
        ],
        "feedback_splits": sorted({event.get("split") for event in feedback}),
        "feedback_case_ids": sorted(
            {
                event["use_case_id"]
                for event in feedback
                if event.get("use_case_id")
            }
        ),
        "updated_edge_count": metadata["edge_weight_update_stats"][
            "updated_edge_count"
        ],
        "increased_edge_count": metadata["edge_weight_update_stats"][
            "increased_edge_count"
        ],
        "decreased_edge_count": metadata["edge_weight_update_stats"][
            "decreased_edge_count"
        ],
    }


def generation_summary(run: dict[str, Any], team: str) -> dict[str, Any]:
    results = run["results"]
    attempts = (
        results
        if team == "red"
        else [
            threat
            for result in results
            for threat in result.get("threat_generation_results", [])
        ]
    )
    retries = [
        {
            "use_case_id": result["use_case_id"],
            "threat_id": attempt.get("threat_id"),
            "retry_count": int(attempt.get("transport_retry_count") or 0),
            "first_error": attempt.get("transport_first_error"),
            "parse_valid": bool(attempt.get("parse_valid")),
            "schema_valid": bool(attempt.get("schema_valid")),
        }
        for result in results
        for attempt in (
            [result] if team == "red" else result.get("threat_generation_results", [])
        )
        if attempt.get("transport_retry_count")
    ]
    return {
        "case_count": len(results),
        "case_ids": [result["use_case_id"] for result in results],
        "record_splits": sorted({result.get("split") for result in results}),
        "attempt_count": len(attempts),
        "parse_valid_count": sum(bool(item.get("parse_valid")) for item in attempts),
        "schema_valid_count": sum(bool(item.get("schema_valid")) for item in attempts),
        "transport_retry_total": sum(
            int(item.get("transport_retry_count") or 0) for item in attempts
        ),
        "transport_retries": retries,
    }


def judge_summary(payload: dict[str, Any], team: str) -> dict[str, Any]:
    counts = payload["summary"]["counts"]
    if team == "red":
        call_count = counts["judge_pair_total"]
        model = payload["meta"]["model_name"]
    else:
        call_count = counts["judge_call_total"]
        model = payload["meta"]["judge_model_name"]
    return {
        "model": model,
        "prompt": payload["meta"]["prompt_path"],
        "call_count": call_count,
        "parse_valid_count": counts["judge_parse_valid_total"],
        "schema_valid_count": counts["judge_schema_valid_total"],
    }


def load_setting(name: str) -> dict[str, Any]:
    root = EXPERIMENT_ROOT / name
    red_run = read_json(root / "red_team" / "red_run.json")
    blue_run = read_json(root / "blue_team" / "blue_run.json")
    red_eval = read_json(root / "red_team" / "red_eval.json")
    blue_eval = read_json(root / "blue_team" / "blue_eval.json")
    return {
        "name": name,
        "red_meta": red_run["meta"],
        "blue_meta": blue_run["meta"],
        "red_generation": generation_summary(red_run, "red"),
        "blue_generation": generation_summary(blue_run, "blue"),
        "red_judge": judge_summary(red_eval, "red"),
        "blue_judge": judge_summary(blue_eval, "blue"),
        "aggregate": read_json(root / "aggregate_metrics.json"),
        "graph": graph_summary(Path(red_run["meta"]["kg_dir"])),
    }


def meta_matches(neutral: dict[str, Any], adapted: dict[str, Any], team: str) -> bool:
    keys = [
        "model_name",
        "model_env_var",
        "rag_top_k",
        "temperature",
        "request_timeout",
        "input_path",
        "prompt_path",
        "retriever_module",
        "transport_policy",
    ]
    if team == "red":
        keys.append("case_registry_path")
    return all(
        neutral[f"{team}_meta"].get(key) == adapted[f"{team}_meta"].get(key)
        for key in keys
    )


def metric_checks(aggregate: dict[str, Any]) -> dict[str, bool]:
    silver = aggregate["silver_threat_total"]
    predicted = aggregate["predicted_threat_total"]
    threats = aggregate["threat_validity_match_total"]
    defenses = aggregate["defense_valid_total"]
    return {
        "micro_threat_validity_recall": abs(
            aggregate["micro_threat_validity_recall"] - threats / silver
        )
        < 1e-6,
        "micro_end_to_end_defense_recall": abs(
            aggregate["micro_end_to_end_defense_recall"] - defenses / silver
        )
        < 1e-6,
        "micro_end_to_end_pipeline_precision": abs(
            aggregate["micro_end_to_end_pipeline_precision"] - defenses / predicted
        )
        < 1e-6,
    }


def retrieval_order(path: Path) -> dict[str, tuple[str, ...]]:
    return {
        item["use_case_id"]: tuple(
            evidence["metadata"]["id"] for evidence in item["retrieved_knowledge"]
        )
        for item in read_jsonl(path)
    }


def main() -> None:
    neutral = load_setting(NEUTRAL)
    adapted = load_setting(ADAPTED)
    registry = read_json(REGISTRY)["cases"]
    test_ids = {item["use_case_id"] for item in registry if item.get("split") == "test"}
    train_ids = {
        item["use_case_id"] for item in registry if item.get("split") == "train"
    }
    neutral_ids = neutral["red_generation"]["case_ids"]
    adapted_ids = adapted["red_generation"]["case_ids"]
    neutral_retrieval = retrieval_order(
        EXPERIMENT_ROOT / NEUTRAL / "red_team" / "red_trace.jsonl"
    )
    adapted_retrieval = retrieval_order(
        EXPERIMENT_ROOT / ADAPTED / "red_team" / "red_trace.jsonl"
    )
    changed_retrieval_cases = sorted(
        case_id
        for case_id in test_ids
        if neutral_retrieval.get(case_id) != adapted_retrieval.get(case_id)
    )
    adapted_feedback_ids = set(adapted["graph"]["feedback_case_ids"])

    generation_items = [neutral, adapted]
    retry_records = [
        retry
        for item in generation_items
        for team in ("red_generation", "blue_generation")
        for retry in item[team]["transport_retries"]
    ]
    retry_policy_compliant = all(
        retry["retry_count"] == 1
        and retry["parse_valid"]
        and retry["schema_valid"]
        and retry["first_error"]
        and (
            "ConnectionError" in retry["first_error"]
            or "502" in retry["first_error"]
            or "503" in retry["first_error"]
            or "504" in retry["first_error"]
            or "timeout" in retry["first_error"].lower()
        )
        for retry in retry_records
    )

    checks = {
        "red_fixed_parameters_matched": meta_matches(neutral, adapted, "red"),
        "blue_fixed_parameters_matched": meta_matches(neutral, adapted, "blue"),
        "expected_test_case_count_57": len(test_ids) == 57,
        "neutral_case_ids_equal_test_1_0": set(neutral_ids) == test_ids,
        "adapted_case_ids_equal_test_1_0": set(adapted_ids) == test_ids,
        "case_order_matched": neutral_ids == adapted_ids,
        "red_blue_case_order_matched": all(
            item["red_generation"]["case_ids"]
            == item["blue_generation"]["case_ids"]
            for item in generation_items
        ),
        "no_duplicate_case_ids": all(
            len(item["red_generation"]["case_ids"])
            == len(set(item["red_generation"]["case_ids"]))
            for item in generation_items
        ),
        "graph_topology_matched": neutral["graph"]["topology_sha256"]
        == adapted["graph"]["topology_sha256"],
        "graph_weights_changed": neutral["graph"]["weight_sha256"]
        != adapted["graph"]["weight_sha256"],
        "alpha_is_strict_0_vs_1": neutral["graph"]["feedback_alpha"] == 0.0
        and adapted["graph"]["feedback_alpha"] == 1.0,
        "feedback_split_filter_train_only": adapted["graph"]["split_filter"]
        == ["train"],
        "feedback_events_train_only": adapted["graph"]["feedback_splits"]
        == ["train"],
        "feedback_case_ids_subset_of_train": adapted_feedback_ids <= train_ids,
        "feedback_event_count_matched_metadata": adapted["graph"][
            "feedback_event_count"
        ]
        == adapted["graph"]["metadata_feedback_event_count"],
        "edge_weights_actually_updated": adapted["graph"]["updated_edge_count"] > 0,
        "retrieval_order_changed_for_test_cases": bool(changed_retrieval_cases),
        "all_generation_outputs_parse_valid": all(
            item[team]["parse_valid_count"] == item[team]["attempt_count"]
            for item in generation_items
            for team in ("red_generation", "blue_generation")
        ),
        "all_generation_outputs_schema_valid": all(
            item[team]["schema_valid_count"] == item[team]["attempt_count"]
            for item in generation_items
            for team in ("red_generation", "blue_generation")
        ),
        "transport_retries_policy_compliant": retry_policy_compliant,
        "judges_matched": all(
            neutral[judge][key] == adapted[judge][key]
            for judge in ("red_judge", "blue_judge")
            for key in ("model", "prompt")
        ),
        "all_judge_outputs_parse_valid": all(
            item[judge]["parse_valid_count"] == item[judge]["call_count"]
            for item in generation_items
            for judge in ("red_judge", "blue_judge")
        ),
        "all_judge_outputs_schema_valid": all(
            item[judge]["schema_valid_count"] == item[judge]["call_count"]
            for item in generation_items
            for judge in ("red_judge", "blue_judge")
        ),
        "neutral_micro_metrics_recomputed": all(
            metric_checks(neutral["aggregate"]).values()
        ),
        "adapted_micro_metrics_recomputed": all(
            metric_checks(adapted["aggregate"]).values()
        ),
    }
    pair = read_json(RESULT_ROOT / "test_pair.json")
    audit = {
        "protocol": "C2-StrictAlpha-GPT-SharedStandard-v1",
        "evidence_role": "post_hoc_diagnostic_replication",
        "neutral": neutral,
        "adapted": adapted,
        "relative_change_pct": pair["relative_change_pct"],
        "changed_red_retrieval_case_count": len(changed_retrieval_cases),
        "changed_red_retrieval_case_ids": changed_retrieval_cases,
        "transport_retry_records": retry_records,
        "checks": checks,
        "audit_passed": all(checks.values()),
    }
    write_json(RESULT_ROOT / "AUDIT.json", audit)

    lines = [
        "# GPT-5.2 Shared-Standard Strict-Alpha Audit",
        "",
        "This read-only audit verifies the isolated alpha=0/alpha=1 pair. It does not alter generated outputs or judgments.",
        "",
        f"- Overall audit: `{'PASS' if audit['audit_passed'] else 'FAIL'}`.",
        f"- Cases: `{len(test_ids)}` from `test_1.0`; identical order on both sides.",
        f"- Graph topology: `{neutral['graph']['topology_sha256']}` on both sides.",
        f"- Feedback: `{adapted['graph']['feedback_event_count']}` train-only events; `{adapted['graph']['updated_edge_count']}` updated edges (`{adapted['graph']['increased_edge_count']}` increased, `{adapted['graph']['decreased_edge_count']}` decreased).",
        f"- Red top-k order changed for `{len(changed_retrieval_cases)}/57` cases.",
        f"- Transport retries: `{len(retry_records)}`; all are policy-compliant and ended with valid parse/schema.",
        "",
        "| Metric | Alpha=0 | Alpha=1 | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {100 * neutral['aggregate'][metric]:.2f}% | "
            f"{100 * adapted['aggregate'][metric]:.2f}% | "
            f"{pair['relative_change_pct'][metric]:+.2f}% |"
        )
    lines.extend(["", "| Check | Result |", "|---|---|"])
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    )
    if retry_records:
        lines.extend(["", "## Transport Retry", ""])
        for retry in retry_records:
            lines.append(
                f"- `{retry['use_case_id']}/{retry['threat_id']}`: "
                f"`{retry['first_error']}`; one retry; final parse/schema valid."
            )
    (RESULT_ROOT / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit_passed": audit["audit_passed"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
