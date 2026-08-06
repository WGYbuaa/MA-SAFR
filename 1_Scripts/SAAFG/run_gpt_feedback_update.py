#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run GPT-5.2-specific feedback-weight iter_01 and test MA-SAFR.

Pipeline:
1. Run GPT-5.2 StaticGraphRAG-GraphAware on train split.
2. Evaluate train split and build feedback events from that GPT run only.
3. Update graph edge weights into a GPT-specific iter_01 KG.
4. Run GPT-5.2 MA-SAFR on test split using the GPT-specific KG.
5. Evaluate Red and Blue outputs on test split.

The script does not overwrite the existing qwen/deepseek iter_01 graph or
the existing RQ1 GPT MA-SAFR outputs.
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
REGISTRY_PATH = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"

STATIC_RUN_TAG = "GPT52_train_feedback"
STATIC_METHOD_PREFIX = "ma_StaticGraphRAG_GraphAware"
STATIC_FILE_METHOD = "staticgraphrag_graphaware"

SAFR_RUN_TAG = "GPT52_gptfb_iter01_test"
SAFR_METHOD_PREFIX = "ma_RecEvoGraphRAG"
SAFR_FILE_METHOD = "recevographrag"

GPT_MODEL_ENV_VAR = "MODEL_GPT_52"
GPT_BASE_URL_ENV = "gpt_BASE_URL"
GPT_API_KEY_ENV = "gpt_API_KEY"
JUDGE_MODEL_ENV_VAR = "MODEL_QWEN35_PLUS"

STATIC_KG_DIR = KB_DIR / "recevograph_rag"
GPT_FEEDBACK_DIR = KB_DIR / "recevograph_rag_feedback" / "v0_2" / "GPT52_train_feedback"
GPT_FEEDBACK_JSONL = GPT_FEEDBACK_DIR / "feedback_events_GPT52_train_feedback.jsonl"
GPT_FEEDBACK_SUMMARY = GPT_FEEDBACK_DIR / "feedback_summary_staticgraphrag_graphaware_v0_2_GPT52_train_feedback.json"
GPT_EVO_KG_DIR = KB_DIR / "recevograph_rag_evo_v0_2" / "GPT52_iter_01_train_feedback"

PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

STAGES = [
    "static-red",
    "static-red-eval",
    "static-blue",
    "static-blue-eval",
    "build-feedback",
    "update-kg",
    "safr-red",
    "safr-red-eval",
    "safr-blue",
    "safr-blue-eval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-5.2 feedback iter_01 MA-SAFR experiment.")
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
        "blue_artifact": blue_dir
        / f"saafg_security_augmented_flows_pred_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "blue_trace": blue_dir / f"saafg_blueteam_retrieval_trace_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.jsonl",
        "blue_log": blue_dir / f"saafg_blueteam_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.log",
        "blue_eval_json": blue_dir / f"saafg_blueteam_task_b_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.json",
        "blue_eval_csv": blue_dir / f"saafg_blueteam_task_b_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.csv",
        "blue_eval_log": blue_dir / f"saafg_blueteam_task_b_eval_{STATIC_FILE_METHOD}_v0_2_{STATIC_RUN_TAG}.log",
    }


def safr_paths() -> Dict[str, Path]:
    root = EXPERIMENT_ROOT / f"{SAFR_METHOD_PREFIX}_{SAFR_RUN_TAG}"
    red_dir = root / "red_team"
    blue_dir = root / "blue_team"
    return {
        "root": root,
        "red_dir": red_dir,
        "red_run": red_dir / f"saafg_redteam_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.json",
        "red_artifact": red_dir / f"saafg_threat_records_pred_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.json",
        "red_trace": red_dir / f"saafg_redteam_retrieval_trace_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.jsonl",
        "red_log": red_dir / f"saafg_redteam_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.log",
        "red_eval_json": red_dir / f"saafg_redteam_task_a_eval_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.json",
        "red_eval_csv": red_dir / f"saafg_redteam_task_a_eval_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.csv",
        "red_eval_log": red_dir / f"saafg_redteam_task_a_eval_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.log",
        "blue_dir": blue_dir,
        "blue_run": blue_dir / f"saafg_blueteam_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.json",
        "blue_artifact": blue_dir / f"saafg_security_augmented_flows_pred_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.json",
        "blue_trace": blue_dir / f"saafg_blueteam_retrieval_trace_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.jsonl",
        "blue_log": blue_dir / f"saafg_blueteam_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.log",
        "blue_eval_json": blue_dir / f"saafg_blueteam_task_b_eval_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.json",
        "blue_eval_csv": blue_dir / f"saafg_blueteam_task_b_eval_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.csv",
        "blue_eval_log": blue_dir / f"saafg_blueteam_task_b_eval_{SAFR_FILE_METHOD}_v0_2_{SAFR_RUN_TAG}.log",
    }


def add_common_generation_args(command: List[str], args: argparse.Namespace, case_ids: Sequence[str]) -> None:
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")
    command.append("--case-id")
    command.extend(case_ids)


def add_common_eval_args(command: List[str], args: argparse.Namespace) -> None:
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")


def command_for_stage(stage: str, args: argparse.Namespace) -> List[str]:
    sp = static_paths()
    mp = safr_paths()
    train_ids = split_case_ids("train")
    test_ids = split_case_ids("test")

    if stage == "static-red":
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
            "--kg-dir",
            str(STATIC_KG_DIR),
            "--rag-top-k",
            str(args.rag_top_k),
            "--result-dir",
            str(sp["red_dir"]),
            "--output-path",
            str(sp["red_run"]),
            "--artifact-path",
            str(sp["red_artifact"]),
            "--retrieval-trace-path",
            str(sp["red_trace"]),
            "--log-path",
            str(sp["red_log"]),
        ]
        add_common_generation_args(command, args, train_ids)
        return command

    if stage == "static-red-eval":
        command = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_threat_validity_by_split.py"),
            "--split",
            "train",
            "--run-tag",
            STATIC_RUN_TAG,
            "--predictions-path",
            str(sp["red_artifact"]),
            "--output-json",
            str(sp["red_eval_json"]),
            "--output-csv",
            str(sp["red_eval_csv"]),
            "--log-path",
            str(sp["red_eval_log"]),
        ]
        add_common_eval_args(command, args)
        return command

    if stage == "static-blue":
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
            str(sp["blue_dir"]),
            "--source-run-output-path",
            str(sp["red_run"]),
            "--source-eval-output-path",
            str(sp["red_eval_json"]),
            "--output-path",
            str(sp["blue_run"]),
            "--artifact-path",
            str(sp["blue_artifact"]),
            "--retrieval-trace-path",
            str(sp["blue_trace"]),
            "--log-path",
            str(sp["blue_log"]),
        ]
        add_common_generation_args(command, args, train_ids)
        return command

    if stage == "static-blue-eval":
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
            str(sp["blue_run"]),
            "--blue-artifact-path",
            str(sp["blue_artifact"]),
            "--red-run-path",
            str(sp["red_run"]),
            "--red-eval-path",
            str(sp["red_eval_json"]),
            "--output-json",
            str(sp["blue_eval_json"]),
            "--output-csv",
            str(sp["blue_eval_csv"]),
            "--log-path",
            str(sp["blue_eval_log"]),
        ]
        add_common_eval_args(command, args)
        return command

    if stage == "build-feedback":
        return [
            sys.executable,
            str(SCRIPT_DIR / "build_feedback_events.py"),
            "--method-dir-prefix",
            STATIC_METHOD_PREFIX,
            "--file-method",
            STATIC_FILE_METHOD,
            "--source-method",
            "StaticGraphRAG-GraphAware-GPT52",
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

    if stage == "update-kg":
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

    if stage == "safr-red":
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_red_team_ma_safr.py"),
            "--run-tag",
            SAFR_RUN_TAG,
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
            str(mp["red_dir"]),
            "--output-path",
            str(mp["red_run"]),
            "--artifact-path",
            str(mp["red_artifact"]),
            "--retrieval-trace-path",
            str(mp["red_trace"]),
            "--log-path",
            str(mp["red_log"]),
        ]
        add_common_generation_args(command, args, test_ids)
        return command

    if stage == "safr-red-eval":
        command = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_threat_validity_by_split.py"),
            "--split",
            "test",
            "--run-tag",
            SAFR_RUN_TAG,
            "--predictions-path",
            str(mp["red_artifact"]),
            "--output-json",
            str(mp["red_eval_json"]),
            "--output-csv",
            str(mp["red_eval_csv"]),
            "--log-path",
            str(mp["red_eval_log"]),
        ]
        add_common_eval_args(command, args)
        return command

    if stage == "safr-blue":
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_blue_team_ma_safr.py"),
            "--run-tag",
            SAFR_RUN_TAG,
            "--source-run-tag",
            SAFR_RUN_TAG,
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
            str(mp["blue_dir"]),
            "--source-run-output-path",
            str(mp["red_run"]),
            "--source-eval-output-path",
            str(mp["red_eval_json"]),
            "--output-path",
            str(mp["blue_run"]),
            "--artifact-path",
            str(mp["blue_artifact"]),
            "--retrieval-trace-path",
            str(mp["blue_trace"]),
            "--log-path",
            str(mp["blue_log"]),
        ]
        add_common_generation_args(command, args, test_ids)
        return command

    if stage == "safr-blue-eval":
        command = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_pipeline_validity_by_split.py"),
            "--split",
            "test",
            "--run-tag",
            SAFR_RUN_TAG,
            "--source-run-tag",
            SAFR_RUN_TAG,
            "--model-env-var",
            JUDGE_MODEL_ENV_VAR,
            "--blue-run-path",
            str(mp["blue_run"]),
            "--blue-artifact-path",
            str(mp["blue_artifact"]),
            "--red-run-path",
            str(mp["red_run"]),
            "--red-eval-path",
            str(mp["red_eval_json"]),
            "--output-json",
            str(mp["blue_eval_json"]),
            "--output-csv",
            str(mp["blue_eval_csv"]),
            "--log-path",
            str(mp["blue_eval_log"]),
        ]
        add_common_eval_args(command, args)
        return command

    raise ValueError(f"Unsupported stage: {stage}")


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=BASE_DIR, check=True)


def main() -> None:
    args = parse_args()
    clear_dead_local_proxy_env()
    for stage in args.stages:
        print(f"[Stage] {stage}", flush=True)
        run_command(command_for_stage(stage, args), args.dry_run)


if __name__ == "__main__":
    main()
