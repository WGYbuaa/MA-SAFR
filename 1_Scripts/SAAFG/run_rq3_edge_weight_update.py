#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build, tune on dev, and test isolated strict-alpha C2 pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result" / "C2_StrictAlpha_v1"
SUMMARY_ROOT = SAAFG_ROOT / "9_result_test_split_2_0" / "C2_StrictAlpha_v1"
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
GRAPH_ROOT = KB_DIR / "C2_strict_alpha_v1"
STATIC_KG = KB_DIR / "recevograph_rag"
ENV_PATH = BASE_DIR / "1_Scripts" / ".env"

REGISTRY_V1 = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
REGISTRY_V2 = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_2.json"
SHARED_FEEDBACK = (
    KB_DIR / "recevograph_rag_feedback" / "v0_2" / "feedback_events.jsonl"
)
GPT_FEEDBACK = (
    KB_DIR
    / "recevograph_rag_feedback"
    / "v0_2"
    / "GPT52_train2_feedback_relaxed"
    / "feedback_events_GPT52_train2_feedback_relaxed.jsonl"
)
DEEP_FILTERED_FEEDBACK = GRAPH_ROOT / "feedback" / "deepseek-v32_train_only.jsonl"

RED_STANDARD = BASE_DIR / "3_Prompt" / "SAAFG" / "red_team_static_graph_rag.txt"
RED_GPT_RELAXED = BASE_DIR / "3_Prompt" / "SAAFG" / "red_team_gpt_with_rssg.txt"
BLUE_STANDARD = BASE_DIR / "3_Prompt" / "SAAFG" / "blue_team_static_graph_rag.txt"

METRIC_KEYS = [
    "micro_threat_validity_recall",
    "micro_end_to_end_defense_recall",
    "micro_end_to_end_pipeline_precision",
    "macro_end_to_end_defense_recall",
]


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-v32": {
        "model_name": "deepseek-v3.2",
        "registry": REGISTRY_V1,
        "api_key_env": "API_KEY",
        "base_url_env": "BASE_URL",
        "model_env": "MODEL_DEEPSEEK_V32",
        "red_prompt": RED_STANDARD,
        "blue_prompt": BLUE_STANDARD,
        "candidates": [
            "shared_std_a1",
            "model_std_a1",
            "model_conservative_a1",
            "model_conservative_a05",
        ],
    },
    "gpt52": {
        "model_name": "GPT-5.2",
        "registry": REGISTRY_V2,
        "api_key_env": "gpt_API_KEY",
        "base_url_env": "gpt_BASE_URL",
        "model_env": "MODEL_GPT_52",
        "red_prompt": RED_GPT_RELAXED,
        "blue_prompt": BLUE_STANDARD,
        "candidates": [
            "model_std_a1",
            "model_conservative_a1",
            "model_std_a05",
            "model_conservative_a05",
        ],
    },
}


CANDIDATES: Dict[str, Dict[str, Any]] = {
    "shared_std_a1": {"policy": "shared_std", "alpha": 1.0},
    "model_std_a1": {"policy": "model_std", "alpha": 1.0},
    "model_conservative_a1": {"policy": "model_conservative", "alpha": 1.0},
    "model_std_a05": {"policy": "model_std", "alpha": 0.5},
    "model_conservative_a05": {"policy": "model_conservative", "alpha": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated C2 strict-alpha experiments.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--phase", choices=["build", "dev-search", "test", "all"], required=True)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_case_ids(registry: Path, split: str) -> List[str]:
    return [
        row["use_case_id"]
        for row in read_json(registry).get("cases", [])
        if row.get("split") == split
    ]


def alpha_label(alpha: float) -> str:
    value = f"{alpha:.3f}".rstrip("0").rstrip(".")
    if "." not in value:
        value += ".0"
    return "alpha_" + value.replace(".", "p")


def graph_dir(model_key: str, policy: str, alpha: float) -> Path:
    return GRAPH_ROOT / model_key / policy / alpha_label(alpha)


def graph_ready(path: Path) -> bool:
    return all(
        (path / name).exists()
        for name in ["graph_nodes.json", "graph_edges.json", "graph_metadata.json", "networkx_graph.pkl"]
    )


def retrieval_graph_sha256(path: Path) -> str:
    """Hash only graph content that can affect retrieval at inference time."""
    nodes_payload = read_json(path / "graph_nodes.json")
    edges_payload = read_json(path / "graph_edges.json")
    nodes = sorted(nodes_payload.get("nodes", []), key=lambda item: item.get("node_id", ""))
    edges = []
    for edge in edges_payload.get("edges", []):
        # Feedback diagnostics differ across policies even when alpha=0. They are
        # not read by the retriever, so exclude them from the neutral-pair audit.
        edges.append({key: value for key, value in edge.items() if key != "feedback_alpha_update"})
    edges.sort(
        key=lambda item: (
            item.get("edge_id", ""),
            item.get("source", ""),
            item.get("relation", ""),
            item.get("target", ""),
        )
    )
    canonical = json.dumps(
        {"nodes": nodes, "edges": edges},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def child_env(model_key: str) -> Dict[str, str]:
    env = dict(os.environ)
    if model_key != "gpt52" or env.get("gpt_API_KEY"):
        return env
    if not ENV_PATH.exists():
        return env
    for raw_line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            stripped = stripped[1:].strip()
        if stripped.startswith("gpt_API_KEY="):
            value = stripped.split("=", 1)[1].strip()
            if value:
                env["gpt_API_KEY"] = value
            break
    return env


def run(command: Sequence[str], dry_run: bool, model_key: str | None = None) -> None:
    print("[Command] " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(
        list(command),
        cwd=BASE_DIR,
        check=True,
        env=child_env(model_key) if model_key else None,
    )


def filter_deep_feedback() -> None:
    DEEP_FILTERED_FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    kept: List[str] = []
    for line in SHARED_FEEDBACK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("run_tag") == "deepseek-v32" and event.get("split") == "train":
            kept.append(json.dumps(event, ensure_ascii=False))
    if not kept:
        raise ValueError("No DeepSeek train feedback events were selected.")
    DEEP_FILTERED_FEEDBACK.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"[Feedback] deepseek_train_event_count={len(kept)}", flush=True)


def graph_specs(model_key: str) -> Dict[str, Dict[str, Any]]:
    if model_key == "deepseek-v32":
        return {
            "shared_std": {"feedback": SHARED_FEEDBACK, "negative_scale": 0.7},
            "model_std": {"feedback": DEEP_FILTERED_FEEDBACK, "negative_scale": 0.7},
            "model_conservative": {"feedback": DEEP_FILTERED_FEEDBACK, "negative_scale": 0.25},
        }
    return {
        "model_std": {"feedback": GPT_FEEDBACK, "negative_scale": 0.7},
        "model_conservative": {"feedback": GPT_FEEDBACK, "negative_scale": 0.25},
    }


def build_graphs(model_key: str, dry_run: bool) -> None:
    if model_key == "deepseek-v32":
        filter_deep_feedback()
    for policy, spec in graph_specs(model_key).items():
        root = GRAPH_ROOT / model_key / policy
        alphas = [0.0, 0.5, 1.0]
        missing_alphas = [alpha for alpha in alphas if not graph_ready(root / alpha_label(alpha))]
        if not missing_alphas:
            print(f"[Graph] ready; skipping {model_key}/{policy}", flush=True)
            continue
        command = [
            sys.executable,
            str(SCRIPT_DIR / "feedback_weight_alpha" / "build_edge_weight_graphs.py"),
            "--kg-dir",
            str(STATIC_KG),
            "--feedback-jsonl",
            str(spec["feedback"]),
            "--output-root",
            str(root),
            "--alphas",
            *(str(alpha) for alpha in missing_alphas),
            "--split-filter",
            "train",
            "--negative-scale",
            str(spec["negative_scale"]),
        ]
        run(command, dry_run)


def assert_neutral_graphs_equal(model_key: str) -> Dict[str, str]:
    hashes = {
        policy: retrieval_graph_sha256(graph_dir(model_key, policy, 0.0))
        for policy in graph_specs(model_key)
    }
    if len(set(hashes.values())) != 1:
        raise ValueError(f"Neutral alpha=0 graphs differ: {hashes}")
    return hashes


def setting_paths(model_key: str, split: str, setting: str) -> Dict[str, Path]:
    root = EXPERIMENT_ROOT / model_key / split / setting
    red = root / "red_team"
    blue = root / "blue_team"
    return {
        "root": root,
        "red_dir": red,
        "red_run": red / "red_run.json",
        "red_artifact": red / "red_artifact.json",
        "red_trace": red / "red_trace.jsonl",
        "red_log": red / "red_run.log",
        "red_eval": red / "red_eval.json",
        "red_eval_csv": red / "red_eval.csv",
        "red_eval_log": red / "red_eval.log",
        "blue_dir": blue,
        "blue_run": blue / "blue_run.json",
        "blue_artifact": blue / "blue_artifact.json",
        "blue_trace": blue / "blue_trace.jsonl",
        "blue_log": blue / "blue_run.log",
        "blue_eval": blue / "blue_eval.json",
        "blue_eval_csv": blue / "blue_eval.csv",
        "blue_eval_log": blue / "blue_eval.log",
        "metrics": root / "aggregate_metrics.json",
    }


def add_common_flags(command: List[str], args: argparse.Namespace) -> None:
    if args.skip_probe:
        command.append("--skip-probe")
    if args.resume:
        command.append("--resume")


def run_setting(
    args: argparse.Namespace,
    split: str,
    setting: str,
    kg_dir: Path,
) -> Dict[str, Any]:
    model_key = args.model
    config = MODEL_CONFIGS[model_key]
    registry = Path(config["registry"])
    case_ids = split_case_ids(registry, split)
    paths = setting_paths(model_key, split, setting)
    run_tag = f"C2v1_{model_key}_{split}_{setting}"

    red_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_saafg_red_team_recevographrag_retry_v1.py"),
        "--run-tag",
        run_tag,
        "--api-key-env",
        config["api_key_env"],
        "--base-url-env",
        config["base_url_env"],
        "--model-env-var",
        config["model_env"],
        "--prompt-path",
        str(config["red_prompt"]),
        "--kg-dir",
        str(kg_dir),
        "--rag-top-k",
        str(args.rag_top_k),
        "--case-registry-path",
        str(registry),
        "--result-dir",
        str(paths["red_dir"]),
        "--output-path",
        str(paths["red_run"]),
        "--artifact-path",
        str(paths["red_artifact"]),
        "--retrieval-trace-path",
        str(paths["red_trace"]),
        "--log-path",
        str(paths["red_log"]),
        "--case-id",
        *case_ids,
    ]
    add_common_flags(red_command, args)
    run(red_command, args.dry_run, model_key)

    red_eval_command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_threat_validity_by_split.py"),
        "--split",
        split,
        "--run-tag",
        run_tag,
        "--case-registry-path",
        str(registry),
        "--predictions-path",
        str(paths["red_artifact"]),
        "--output-json",
        str(paths["red_eval"]),
        "--output-csv",
        str(paths["red_eval_csv"]),
        "--log-path",
        str(paths["red_eval_log"]),
    ]
    add_common_flags(red_eval_command, args)
    run(red_eval_command, args.dry_run, model_key)

    blue_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_saafg_blue_team_recevographrag_retry_v1.py"),
        "--run-tag",
        run_tag,
        "--source-run-tag",
        run_tag,
        "--api-key-env",
        config["api_key_env"],
        "--base-url-env",
        config["base_url_env"],
        "--model-env-var",
        config["model_env"],
        "--prompt-path",
        str(config["blue_prompt"]),
        "--kg-dir",
        str(kg_dir),
        "--rag-top-k",
        str(args.rag_top_k),
        "--result-dir",
        str(paths["blue_dir"]),
        "--source-run-output-path",
        str(paths["red_run"]),
        "--source-eval-output-path",
        str(paths["red_eval"]),
        "--output-path",
        str(paths["blue_run"]),
        "--artifact-path",
        str(paths["blue_artifact"]),
        "--retrieval-trace-path",
        str(paths["blue_trace"]),
        "--log-path",
        str(paths["blue_log"]),
        "--case-id",
        *case_ids,
    ]
    add_common_flags(blue_command, args)
    run(blue_command, args.dry_run, model_key)

    blue_eval_command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pipeline_validity_by_split.py"),
        "--split",
        split,
        "--run-tag",
        run_tag,
        "--source-run-tag",
        run_tag,
        "--case-registry-path",
        str(registry),
        "--blue-run-path",
        str(paths["blue_run"]),
        "--blue-artifact-path",
        str(paths["blue_artifact"]),
        "--red-run-path",
        str(paths["red_run"]),
        "--red-eval-path",
        str(paths["red_eval"]),
        "--output-json",
        str(paths["blue_eval"]),
        "--output-csv",
        str(paths["blue_eval_csv"]),
        "--log-path",
        str(paths["blue_eval_log"]),
    ]
    add_common_flags(blue_eval_command, args)
    run(blue_eval_command, args.dry_run, model_key)

    aggregate_command = [
        sys.executable,
        str(SCRIPT_DIR / "aggregate_metrics_by_split.py"),
        "--registry-path",
        str(registry),
        "--split",
        split,
        "--red-eval-path",
        str(paths["red_eval"]),
        "--blue-eval-path",
        str(paths["blue_eval"]),
        "--model",
        config["model_name"],
        "--setting",
        f"C2-StrictAlpha-v1 {setting}",
        "--run-tag",
        run_tag,
        "--output-dir",
        str(paths["root"]),
        "--output-prefix",
        "aggregate",
    ]
    run(aggregate_command, args.dry_run, model_key)
    return {} if args.dry_run else read_json(paths["metrics"])


def relative_changes(neutral: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in METRIC_KEYS:
        base = float(neutral[key])
        value = float(candidate[key])
        result[key] = 100.0 * (value - base) / base if base else 0.0
    return result


def all_positive(changes: Dict[str, float]) -> bool:
    return all(value > 1e-9 for value in changes.values())


def candidate_rank(changes: Dict[str, float]) -> tuple[Any, ...]:
    primary_positive = (
        changes["micro_end_to_end_defense_recall"] > 0
        and changes["micro_end_to_end_pipeline_precision"] > 0
    )
    values = list(changes.values())
    return (
        int(all_positive(changes)),
        int(primary_positive),
        sum(1 for value in values if value > 0),
        min(values),
        sum(values) / len(values),
    )


def selection_paths(model_key: str) -> Dict[str, Path]:
    root = SUMMARY_ROOT / model_key
    return {"root": root, "json": root / "dev_selection.json", "md": root / "dev_selection.md"}


def write_dev_selection(
    model_key: str,
    neutral: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    selected: Dict[str, Any],
    neutral_hashes: Dict[str, str],
) -> None:
    paths = selection_paths(model_key)
    payload = {
        "protocol": "C2-StrictAlpha-v1",
        "model": model_key,
        "selection_split": "dev",
        "selection_rule": (
            "Stop at the first candidate with four strictly positive relative changes; otherwise rank "
            "by primary E2E positivity, positive metric count, minimum relative change, then mean change."
        ),
        "neutral_setting": "neutral_alpha0",
        "neutral_metrics": {key: neutral[key] for key in METRIC_KEYS},
        "neutral_graph_edge_sha256": neutral_hashes,
        "candidate_records": list(records),
        "selected_candidate": selected,
        "test_not_used_for_selection": True,
    }
    write_json(paths["json"], payload)
    lines = [
        f"# {model_key} C2 Strict Alpha Dev Selection",
        "",
        "| Candidate | Threat | E2E recall | Precision | Macro E2E | All positive |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        c = record["relative_change_pct"]
        lines.append(
            "| {name} | {t:+.2f}% | {e:+.2f}% | {p:+.2f}% | {m:+.2f}% | {ok} |".format(
                name=record["candidate"],
                t=c[METRIC_KEYS[0]],
                e=c[METRIC_KEYS[1]],
                p=c[METRIC_KEYS[2]],
                m=c[METRIC_KEYS[3]],
                ok="yes" if record["all_positive"] else "no",
            )
        )
    lines.extend(
        [
            "",
            f"Selected: `{selected['candidate']}`.",
            "",
            "The test split was not read during selection.",
        ]
    )
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def dev_search(args: argparse.Namespace) -> None:
    model_key = args.model
    hashes = assert_neutral_graphs_equal(model_key)
    first_policy = CANDIDATES[MODEL_CONFIGS[model_key]["candidates"][0]]["policy"]
    neutral = run_setting(
        args,
        "dev",
        "neutral_alpha0",
        graph_dir(model_key, first_policy, 0.0),
    )
    records: List[Dict[str, Any]] = []
    selected: Dict[str, Any] | None = None
    for candidate_name in MODEL_CONFIGS[model_key]["candidates"]:
        spec = CANDIDATES[candidate_name]
        metrics = run_setting(
            args,
            "dev",
            candidate_name,
            graph_dir(model_key, spec["policy"], spec["alpha"]),
        )
        changes = relative_changes(neutral, metrics)
        record = {
            "candidate": candidate_name,
            "policy": spec["policy"],
            "alpha": spec["alpha"],
            "metrics": {key: metrics[key] for key in METRIC_KEYS},
            "relative_change_pct": changes,
            "all_positive": all_positive(changes),
            "rank": list(candidate_rank(changes)),
        }
        records.append(record)
        if record["all_positive"]:
            selected = record
            print(f"[Selection] four-positive stop at {candidate_name}", flush=True)
            break
    if selected is None:
        selected = max(records, key=lambda item: tuple(item["rank"]))
        print(f"[Selection] fallback best dev candidate={selected['candidate']}", flush=True)
    write_dev_selection(model_key, neutral, records, selected, hashes)


def write_test_pair(model_key: str, neutral: Dict[str, Any], candidate: Dict[str, Any], selected: Dict[str, Any]) -> None:
    root = SUMMARY_ROOT / model_key
    changes = relative_changes(neutral, candidate)
    payload = {
        "protocol": "C2-StrictAlpha-v1",
        "model": model_key,
        "split": "test",
        "selected_on_dev": selected["candidate"],
        "alpha": selected["alpha"],
        "policy": selected["policy"],
        "neutral_metrics": {key: neutral[key] for key in METRIC_KEYS},
        "adapted_metrics": {key: candidate[key] for key in METRIC_KEYS},
        "relative_change_pct": changes,
        "all_four_positive": all_positive(changes),
    }
    write_json(root / "test_pair.json", payload)
    with (root / "test_pair.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "metric", "alpha0", "adapted", "relative_change_pct"])
        for key in METRIC_KEYS:
            writer.writerow([model_key, key, neutral[key], candidate[key], f"{changes[key]:.6f}"])
    lines = [
        f"# {model_key} C2 Strict Alpha Test Pair",
        "",
        f"Dev-selected candidate: `{selected['candidate']}` (`alpha={selected['alpha']}`).",
        "",
        "| Metric | Alpha=0 | Adapted | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {100*float(neutral[key]):.2f}% | {100*float(candidate[key]):.2f}% | {changes[key]:+.2f}% |"
        )
    lines.extend(["", f"All four positive: `{'yes' if all_positive(changes) else 'no'}`."])
    (root / "test_pair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_test(args: argparse.Namespace) -> None:
    selection = read_json(selection_paths(args.model)["json"])
    selected = selection["selected_candidate"]
    policy = selected["policy"]
    neutral = run_setting(
        args,
        "test",
        "neutral_alpha0",
        graph_dir(args.model, policy, 0.0),
    )
    adapted = run_setting(
        args,
        "test",
        selected["candidate"],
        graph_dir(args.model, policy, float(selected["alpha"])),
    )
    write_test_pair(args.model, neutral, adapted, selected)


def main() -> None:
    args = parse_args()
    phases = [args.phase] if args.phase != "all" else ["build", "dev-search", "test"]
    for phase in phases:
        print(f"[Phase] model={args.model} phase={phase}", flush=True)
        if phase == "build":
            build_graphs(args.model, args.dry_run)
        elif phase == "dev-search":
            dev_search(args)
        elif phase == "test":
            run_test(args)


if __name__ == "__main__":
    main()
