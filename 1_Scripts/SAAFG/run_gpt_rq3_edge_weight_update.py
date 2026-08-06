#!/usr/bin/env python3
"""Run the fixed GPT-5.2 shared-feedback, standard-prompt strict-alpha pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import run_rq3_edge_weight_update as c2


ROOT = Path(__file__).resolve().parents[2]
SAAFG_ROOT = ROOT / "0_Data" / "6_SAAFG"
EXPERIMENT_ROOT = (
    SAAFG_ROOT / "6_Experiment_Result" / "C2_StrictAlpha_GPT_SharedStandard_v1"
)
SUMMARY_ROOT = (
    SAAFG_ROOT / "9_result_test_split_2_0" / "C2_StrictAlpha_GPT_SharedStandard_v1"
)
GRAPH_ROOT = ROOT / "0_Data" / "5_Knowledge_Base" / "C2_strict_alpha_v1"
SHARED_ALPHA0 = GRAPH_ROOT / "deepseek-v32" / "shared_std" / "alpha_0p0"
SHARED_ALPHA1 = GRAPH_ROOT / "deepseek-v32" / "shared_std" / "alpha_1p0"
REGISTRY = (
    SAAFG_ROOT
    / "7_Benchmark_Package_v0_2"
    / "case_registry_test_1.json"
)
RED_PROMPT = ROOT / "3_Prompt" / "SAAFG" / "red_team_static_graph_rag.txt"
BLUE_PROMPT = ROOT / "3_Prompt" / "SAAFG" / "blue_team_static_graph_rag.txt"
METRICS = c2.METRIC_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["neutral", "adapted", "summarize", "all"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_base() -> None:
    c2.EXPERIMENT_ROOT = EXPERIMENT_ROOT
    c2.SUMMARY_ROOT = SUMMARY_ROOT
    c2.MODEL_CONFIGS["gpt52"] = {
        "model_name": "GPT-5.2",
        "registry": REGISTRY,
        "api_key_env": "gpt_API_KEY",
        "base_url_env": "gpt_BASE_URL",
        "model_env": "MODEL_GPT_52",
        "red_prompt": RED_PROMPT,
        "blue_prompt": BLUE_PROMPT,
        "candidates": ["shared_std_a1"],
    }


def runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model="gpt52",
        rag_top_k=3,
        skip_probe=args.skip_probe,
        resume=args.resume,
        dry_run=args.dry_run,
    )


def metric_path(setting: str) -> Path:
    return EXPERIMENT_ROOT / "gpt52" / "test" / setting / "aggregate_metrics.json"


def run_setting(args: argparse.Namespace, setting: str, graph: Path) -> dict[str, Any]:
    if not all((graph / name).exists() for name in ("graph_nodes.json", "graph_edges.json", "graph_metadata.json", "networkx_graph.pkl")):
        raise FileNotFoundError(f"Shared graph is incomplete: {graph}")
    return c2.run_setting(runner_args(args), "test", setting, graph)


def transport_retry_count(setting: str) -> int:
    root = EXPERIMENT_ROOT / "gpt52" / "test" / setting
    red = read_json(root / "red_team" / "red_run.json")
    blue = read_json(root / "blue_team" / "blue_run.json")
    red_retries = sum(int(item.get("transport_retry_count") or 0) for item in red["results"])
    blue_retries = sum(
        int(threat.get("transport_retry_count") or 0)
        for case in blue["results"]
        for threat in case.get("threat_generation_results", [])
    )
    return red_retries + blue_retries


def summarize() -> None:
    neutral = read_json(metric_path("neutral_shared_standard_alpha0"))
    adapted = read_json(metric_path("adapted_shared_standard_alpha1"))
    changes = c2.relative_changes(neutral, adapted)
    payload = {
        "protocol": "C2-StrictAlpha-GPT-SharedStandard-v1",
        "evidence_role": "post_hoc_diagnostic_replication",
        "model": "GPT-5.2",
        "split": "test_1.0",
        "case_count": neutral["case_count"],
        "fixed_configuration": {
            "red_prompt": str(RED_PROMPT),
            "blue_prompt": str(BLUE_PROMPT),
            "retrieval_top_k": 3,
            "neutral_graph": str(SHARED_ALPHA0),
            "adapted_graph": str(SHARED_ALPHA1),
            "feedback_source": "shared Qwen+DeepSeek train-only feedback",
        },
        "neutral_metrics": {key: neutral[key] for key in METRICS},
        "adapted_metrics": {key: adapted[key] for key in METRICS},
        "relative_change_pct": changes,
        "all_four_positive": c2.all_positive(changes),
        "transport_retry_count": {
            "neutral": transport_retry_count("neutral_shared_standard_alpha0"),
            "adapted": transport_retry_count("adapted_shared_standard_alpha1"),
        },
    }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    (SUMMARY_ROOT / "test_pair.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (SUMMARY_ROOT / "test_pair.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "split", "metric", "alpha0", "alpha1", "relative_change_pct"])
        for key in METRICS:
            writer.writerow(["GPT-5.2", "test_1.0", key, neutral[key], adapted[key], f"{changes[key]:.6f}"])
    lines = [
        "# GPT-5.2 Shared-Standard Strict-Alpha Pair",
        "",
        "Post-hoc diagnostic replication with fixed `test_1.0`, standard GraphAware/RSSG prompts, top-k=3, and the shared Qwen+DeepSeek train-feedback graph.",
        "",
        "| Metric | Alpha=0 | Alpha=1 | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for key in METRICS:
        lines.append(
            f"| {key} | {100 * neutral[key]:.2f}% | {100 * adapted[key]:.2f}% | {changes[key]:+.2f}% |"
        )
    lines.extend(
        [
            "",
            f"All four positive: `{'yes' if payload['all_four_positive'] else 'no'}`.",
            f"Transport retries (alpha0/alpha1): `{payload['transport_retry_count']['neutral']}/{payload['transport_retry_count']['adapted']}`.",
            "",
            "This result is diagnostic rather than a newly untouched held-out estimate.",
        ]
    )
    (SUMMARY_ROOT / "test_pair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    configure_base()
    if args.phase in ("neutral", "all"):
        run_setting(args, "neutral_shared_standard_alpha0", SHARED_ALPHA0)
    if args.phase in ("adapted", "all"):
        run_setting(args, "adapted_shared_standard_alpha1", SHARED_ALPHA1)
    if args.phase in ("summarize", "all") and not args.dry_run:
        summarize()


if __name__ == "__main__":
    main()
