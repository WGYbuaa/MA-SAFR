#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run GPT-5.2 feedback-weight iter_01 on SAAFG split_2.0.

Pipeline:
1. Run GPT-5.2 StaticGraphRAG-GraphAware on train_2.0 using the relaxed GPT Red prompt.
2. Evaluate train_2.0 and build feedback events from train_2.0 only.
3. Update graph edge weights into a split_2.0 GPT-specific iter_01 KG.
4. Run GPT-5.2 MA-SAFR on dev_2.0 and test_2.0 using the updated KG and relaxed GPT Red prompt.
5. Evaluate dev_2.0 and test_2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
REGISTRY_PATH = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_2.json"
RED_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "red_team_gpt_with_rssg.txt"

STATIC_RUN_TAG = "GPT52_train2_feedback_relaxed"
STATIC_METHOD_PREFIX = "ma_StaticGraphRAG_GraphAware"
STATIC_FILE_METHOD = "staticgraphrag_graphaware"

SAFR_METHOD_PREFIX = "ma_RecEvoGraphRAG"
SAFR_FILE_METHOD = "recevographrag"
SAFR_TAGS = {
    "dev": "GPT52_gptfb_iter01_2_0_dev_relaxed",
    "test": "GPT52_gptfb_iter01_2_0_test_relaxed",
}

GPT_MODEL_ENV_VAR = "MODEL_GPT_52"
GPT_BASE_URL_ENV = "gpt_BASE_URL"
GPT_API_KEY_ENV = "gpt_API_KEY"
JUDGE_MODEL_ENV_VAR = "MODEL_QWEN35_PLUS"

STATIC_KG_DIR = KB_DIR / "recevograph_rag"
GPT_FEEDBACK_DIR = KB_DIR / "recevograph_rag_feedback" / "v0_2" / STATIC_RUN_TAG
GPT_FEEDBACK_JSONL = GPT_FEEDBACK_DIR / f"feedback_events_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.jsonl"
GPT_FEEDBACK_SUMMARY = GPT_FEEDBACK_DIR / f"feedback_summary_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json"
GPT_EVO_KG_DIR = KB_DIR / "recevograph_rag_evo_v0_2" / "GPT52_iter_01_2_0_train_feedback_relaxed"

PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

STAGES = [
    "static-red",
    "static-red-eval",
    "static-blue",
    "static-blue-eval",
    "build-feedback",
    "update-kg",
    "safr-red-dev",
    "safr-red-eval-dev",
    "safr-blue-dev",
    "safr-blue-eval-dev",
    "safr-red-test",
    "safr-red-eval-test",
    "safr-blue-test",
    "safr-blue-eval-test",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-5.2 feedback iter_01 on split_2.0.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clear_dead_local_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value and "127.0.0.1:9" in value:
            os.environ.pop(key, None)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def split_case_ids(split: str) -> List[str]:
    cases = read_json(REGISTRY_PATH).get("cases") or []
    return [case["use_case_id"] for case in cases if case.get("split") == split]


def static_paths() -> Dict[str, Path]:
    root = EXPERIMENT_ROOT / f"{STATIC_METHOD_PREFIX}_{STATIC_RUN_TAG}"
    red_dir = root / "red_team"
    blue_dir = root / "blue_team"
    return {
        "root": root,
        "red_dir": red_dir,
        "red_run": red_dir / f"saafg_redteam_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "red_artifact": red_dir / f"saafg_threat_records_pred_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "red_trace": red_dir / f"saafg_redteam_retrieval_trace_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.jsonl",
        "red_log": red_dir / f"saafg_redteam_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.log",
        "red_eval_json": red_dir / f"saafg_redteam_task_a_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "red_eval_csv": red_dir / f"saafg_redteam_task_a_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.csv",
        "red_eval_log": red_dir / f"saafg_redteam_task_a_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.log",
        "blue_dir": blue_dir,
        "blue_run": blue_dir / f"saafg_blueteam_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "blue_artifact": blue_dir / f"saafg_security_augmented_flows_pred_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "blue_trace": blue_dir / f"saafg_blueteam_retrieval_trace_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.jsonl",
        "blue_log": blue_dir / f"saafg_blueteam_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.log",
        "blue_eval_json": blue_dir / f"saafg_blueteam_task_b_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "blue_eval_csv": blue_dir / f"saafg_blueteam_task_b_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.csv",
        "blue_eval_log": blue_dir / f"saafg_blueteam_task_b_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.log",
    }


def safr_paths(split: str) -> Dict[str, Path]:
    run_tag = SAFR_TAGS[split]
    root = EXPERIMENT_ROOT / f"{SAFR_METHOD_PREFIX}_{run_tag}"
    red_dir = root / "red_team"
    blue_dir = root / "blue_team"
    return {
        "root": root,
        "red_dir": red_dir,
        "red_run": red_dir / f"saafg_redteam_{SAFR_FILE_METHOD}_v0_2_{run_tag}.json",
        "red_artifact": red_dir / f"saafg_threat_records_pred_{SAFR_FILE_METHOD}_v0_2_{run_tag}.json",
        "red_trace": red_dir / f"saafg_redteam_retrieval_trace_{SAFR_FILE_METHOD}_v0_2_{run_tag}.jsonl",
        "red_log": red_dir / f"saafg_redteam_{SAFR_FILE_METHOD}_v0_2_{run_tag}.log",
        "red_eval_json": red_dir / "task_a_eval.json",
        "red_eval_csv": red_dir / "task_a_eval.csv",
        "red_eval_log": red_dir / "task_a_eval.log",
        "blue_dir": blue_dir,
        "blue_run": blue_dir / f"saafg_blueteam_{SAFR_FILE_METHOD}_v0_2_{run_tag}.json",
        "blue_artifact": blue_dir / f"saafg_security_augmented_flows_pred_{SAFR_FILE_METHOD}_v0_2_{run_tag}.json",
        "blue_trace": blue_dir / f"saafg_blueteam_retrieval_trace_{SAFR_FILE_METHOD}_v0_2_{run_tag}.jsonl",
        "blue_log": blue_dir / f"saafg_blueteam_{SAFR_FILE_METHOD}_v0_2_{run_tag}.log",
        "blue_eval_json": blue_dir / "task_b_eval.json",
        "blue_eval_csv": blue_dir / "task_b_eval.csv",
        "blue_eval_log": blue_dir / "task_b_eval.log",
    }


def add_generation_common(
    command: List[str],
    args: argparse.Namespace,
    case_ids: Sequence[str],
    *,
    include_registry: bool,
) -> None:
    if include_registry:
        command.extend(["--case-registry-path", str(REGISTRY_PATH)])
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")
    command.append("--case-id")
    command.extend(case_ids)


def add_eval_common(command: List[str], args: argparse.Namespace) -> None:
    command.extend(["--case-registry-path", str(REGISTRY_PATH)])
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")


def static_red_command(args: argparse.Namespace) -> List[str]:
    p = static_paths()
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_red_team_static_graph_rag.py"),
        "--run-tag",
        STATIC_RUN_TAG,
        "--api-key-env",
        GPT_API_KEY_ENV,
        "--base-url-env",
        GPT_BASE_URL_ENV,
        "--model-env-var",
        GPT_MODEL_ENV_VAR,
        "--prompt-path",
        str(RED_PROMPT_PATH),
        "--kg-dir",
        str(STATIC_KG_DIR),
        "--rag-top-k",
        str(args.rag_top_k),
        "--result-dir",
        str(p["red_dir"]),
        "--output-path",
        str(p["red_run"]),
        "--artifact-path",
        str(p["red_artifact"]),
        "--retrieval-trace-path",
        str(p["red_trace"]),
        "--log-path",
        str(p["red_log"]),
    ]
    add_generation_common(command, args, split_case_ids("train"), include_registry=True)
    return command


def static_red_eval_command(args: argparse.Namespace) -> List[str]:
    p = static_paths()
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_threat_validity_by_split.py"),
        "--split",
        "train",
        "--run-tag",
        STATIC_RUN_TAG,
        "--predictions-path",
        str(p["red_artifact"]),
        "--output-json",
        str(p["red_eval_json"]),
        "--output-csv",
        str(p["red_eval_csv"]),
        "--log-path",
        str(p["red_eval_log"]),
    ]
    add_eval_common(command, args)
    return command


def static_blue_command(args: argparse.Namespace) -> List[str]:
    p = static_paths()
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_blue_team_static_graph_rag.py"),
        "--run-tag",
        STATIC_RUN_TAG,
        "--source-run-tag",
        STATIC_RUN_TAG,
        "--api-key-env",
        GPT_API_KEY_ENV,
        "--base-url-env",
        GPT_BASE_URL_ENV,
        "--model-env-var",
        GPT_MODEL_ENV_VAR,
        "--kg-dir",
        str(STATIC_KG_DIR),
        "--rag-top-k",
        str(args.rag_top_k),
        "--result-dir",
        str(p["blue_dir"]),
        "--source-run-output-path",
        str(p["red_run"]),
        "--source-eval-output-path",
        str(p["red_eval_json"]),
        "--output-path",
        str(p["blue_run"]),
        "--artifact-path",
        str(p["blue_artifact"]),
        "--retrieval-trace-path",
        str(p["blue_trace"]),
        "--log-path",
        str(p["blue_log"]),
    ]
    add_generation_common(command, args, split_case_ids("train"), include_registry=False)
    return command


def static_blue_eval_command(args: argparse.Namespace) -> List[str]:
    p = static_paths()
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pipeline_validity_by_split.py"),
        "--split",
        "train",
        "--run-tag",
        STATIC_RUN_TAG,
        "--source-run-tag",
        STATIC_RUN_TAG,
        "--model-env-var",
        JUDGE_MODEL_ENV_VAR,
        "--blue-run-path",
        str(p["blue_run"]),
        "--blue-artifact-path",
        str(p["blue_artifact"]),
        "--red-run-path",
        str(p["red_run"]),
        "--red-eval-path",
        str(p["red_eval_json"]),
        "--output-json",
        str(p["blue_eval_json"]),
        "--output-csv",
        str(p["blue_eval_csv"]),
        "--log-path",
        str(p["blue_eval_log"]),
    ]
    add_eval_common(command, args)
    return command


def feedback_command() -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "build_feedback_events.py"),
        "--case-registry-path",
        str(REGISTRY_PATH),
        "--method-dir-prefix",
        STATIC_METHOD_PREFIX,
        "--file-method",
        STATIC_FILE_METHOD,
        "--source-method",
        "StaticGraphRAG-GraphAware-GPT52-RelaxedPrompt-Split2",
        "--run-tags",
        STATIC_RUN_TAG,
        "--split-filter",
        "train",
        "--kg-dir",
        str(STATIC_KG_DIR),
        "--output-dir",
        str(GPT_FEEDBACK_DIR),
        "--output-jsonl",
        str(GPT_FEEDBACK_JSONL),
        "--summary-json",
        str(GPT_FEEDBACK_SUMMARY),
    ]


def update_kg_command() -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "update_edge_weights.py"),
        "--kg-dir",
        str(STATIC_KG_DIR),
        "--feedback-jsonl",
        str(GPT_FEEDBACK_JSONL),
        "--output-dir",
        str(GPT_EVO_KG_DIR),
        "--split-filter",
        "train",
    ]


def safr_red_command(args: argparse.Namespace, split: str) -> List[str]:
    p = safr_paths(split)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_red_team_ma_safr.py"),
        "--run-tag",
        SAFR_TAGS[split],
        "--api-key-env",
        GPT_API_KEY_ENV,
        "--base-url-env",
        GPT_BASE_URL_ENV,
        "--model-env-var",
        GPT_MODEL_ENV_VAR,
        "--prompt-path",
        str(RED_PROMPT_PATH),
        "--kg-dir",
        str(GPT_EVO_KG_DIR),
        "--rag-top-k",
        str(args.rag_top_k),
        "--result-dir",
        str(p["red_dir"]),
        "--output-path",
        str(p["red_run"]),
        "--artifact-path",
        str(p["red_artifact"]),
        "--retrieval-trace-path",
        str(p["red_trace"]),
        "--log-path",
        str(p["red_log"]),
    ]
    add_generation_common(command, args, split_case_ids(split), include_registry=True)
    return command


def safr_red_eval_command(args: argparse.Namespace, split: str) -> List[str]:
    p = safr_paths(split)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_threat_validity_by_split.py"),
        "--split",
        split,
        "--run-tag",
        SAFR_TAGS[split],
        "--predictions-path",
        str(p["red_artifact"]),
        "--output-json",
        str(p["red_eval_json"]),
        "--output-csv",
        str(p["red_eval_csv"]),
        "--log-path",
        str(p["red_eval_log"]),
    ]
    add_eval_common(command, args)
    return command


def safr_blue_command(args: argparse.Namespace, split: str) -> List[str]:
    p = safr_paths(split)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_blue_team_ma_safr.py"),
        "--run-tag",
        SAFR_TAGS[split],
        "--source-run-tag",
        SAFR_TAGS[split],
        "--api-key-env",
        GPT_API_KEY_ENV,
        "--base-url-env",
        GPT_BASE_URL_ENV,
        "--model-env-var",
        GPT_MODEL_ENV_VAR,
        "--kg-dir",
        str(GPT_EVO_KG_DIR),
        "--rag-top-k",
        str(args.rag_top_k),
        "--result-dir",
        str(p["blue_dir"]),
        "--source-run-output-path",
        str(p["red_run"]),
        "--source-eval-output-path",
        str(p["red_eval_json"]),
        "--output-path",
        str(p["blue_run"]),
        "--artifact-path",
        str(p["blue_artifact"]),
        "--retrieval-trace-path",
        str(p["blue_trace"]),
        "--log-path",
        str(p["blue_log"]),
    ]
    add_generation_common(command, args, split_case_ids(split), include_registry=False)
    return command


def safr_blue_eval_command(args: argparse.Namespace, split: str) -> List[str]:
    p = safr_paths(split)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pipeline_validity_by_split.py"),
        "--split",
        split,
        "--run-tag",
        SAFR_TAGS[split],
        "--source-run-tag",
        SAFR_TAGS[split],
        "--model-env-var",
        JUDGE_MODEL_ENV_VAR,
        "--blue-run-path",
        str(p["blue_run"]),
        "--blue-artifact-path",
        str(p["blue_artifact"]),
        "--red-run-path",
        str(p["red_run"]),
        "--red-eval-path",
        str(p["red_eval_json"]),
        "--output-json",
        str(p["blue_eval_json"]),
        "--output-csv",
        str(p["blue_eval_csv"]),
        "--log-path",
        str(p["blue_eval_log"]),
    ]
    add_eval_common(command, args)
    return command


def command_for_stage(stage: str, args: argparse.Namespace) -> List[str]:
    if stage == "static-red":
        return static_red_command(args)
    if stage == "static-red-eval":
        return static_red_eval_command(args)
    if stage == "static-blue":
        return static_blue_command(args)
    if stage == "static-blue-eval":
        return static_blue_eval_command(args)
    if stage == "build-feedback":
        return feedback_command()
    if stage == "update-kg":
        return update_kg_command()
    if stage == "safr-red-dev":
        return safr_red_command(args, "dev")
    if stage == "safr-red-eval-dev":
        return safr_red_eval_command(args, "dev")
    if stage == "safr-blue-dev":
        return safr_blue_command(args, "dev")
    if stage == "safr-blue-eval-dev":
        return safr_blue_eval_command(args, "dev")
    if stage == "safr-red-test":
        return safr_red_command(args, "test")
    if stage == "safr-red-eval-test":
        return safr_red_eval_command(args, "test")
    if stage == "safr-blue-test":
        return safr_blue_command(args, "test")
    if stage == "safr-blue-eval-test":
        return safr_blue_eval_command(args, "test")
    raise ValueError(f"Unsupported stage: {stage}")


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=BASE_DIR, check=True)


def main() -> None:
    args = parse_args()
    clear_dead_local_proxy_env()
    for split in ["train", "dev", "test"]:
        print(f"[Config] split={split} case_count={len(split_case_ids(split))}", flush=True)
    print(f"[Config] registry={REGISTRY_PATH}", flush=True)
    print(f"[Config] red_prompt={RED_PROMPT_PATH}", flush=True)
    print(f"[Config] evo_kg={GPT_EVO_KG_DIR}", flush=True)
    for stage in args.stages:
        print(f"[Stage] {stage}", flush=True)
        run_command(command_for_stage(stage, args), args.dry_run)


if __name__ == "__main__":
    main()
