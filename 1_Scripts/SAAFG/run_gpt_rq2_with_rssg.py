#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run the GPT-5.2 relaxed RSSG pair on test_1.0.

Both settings use the GPT-specific train_1.0 feedback graph. The with-RSSG
setting mirrors the split_2.0 relaxed configuration: relaxed Red guidance and
the full RSSG Blue prompt. The paired without-RSSG setting replaces those
prompts with the frozen relaxed generic counterparts created for the GPT
ablation. Model, retrieval, split, generation policy, and judges are fixed.
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
REGISTRY_PATH = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
KG_DIR = (
    BASE_DIR
    / "0_Data"
    / "5_Knowledge_Base"
    / "recevograph_rag_evo_v0_2"
    / "GPT52_iter_01_train_feedback"
)

PROMPTS = {
    "with": {
        "red": BASE_DIR / "3_Prompt" / "SAAFG" / "red_team_gpt_with_rssg.txt",
        "blue": BASE_DIR / "3_Prompt" / "SAAFG" / "blue_team_static_graph_rag.txt",
    },
    "without": {
        "red": (
            BASE_DIR
            / "3_Prompt"
            / "SAAFG"
            / "red_team_gpt_without_rssg.txt"
        ),
        "blue": (
            BASE_DIR
            / "3_Prompt"
            / "SAAFG"
            / "blue_team_gpt_without_rssg.txt"
        ),
    },
}

RUN_TAGS = {
    "with": "GPT52_i01_s1_relaxed_with",
    "without": "GPT52_i01_s1_relaxed_woRSSG",
}
RESULT_ROOTS = {
    "with": EXPERIMENT_ROOT / "ma_SAFR_GPT52_i01_s1_relaxed_with",
    "without": EXPERIMENT_ROOT / "ma_SAFR_GPT52_i01_s1_relaxed_woRSSG",
}

GPT_MODEL_ENV_VAR = "MODEL_GPT_52"
GPT_BASE_URL_ENV = "gpt_BASE_URL"
GPT_API_KEY_ENV = "gpt_API_KEY"
JUDGE_MODEL_ENV_VAR = "MODEL_QWEN35_PLUS"
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

STAGES = [
    "with-red",
    "with-red-eval",
    "with-blue",
    "with-blue-eval",
    "without-red",
    "without-red-eval",
    "without-blue",
    "without-blue-eval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GPT relaxed RSSG pair on test_1.0.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clear_dead_local_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value and "127.0.0.1:9" in value:
            os.environ.pop(key, None)


def test_case_ids() -> List[str]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [case["use_case_id"] for case in payload.get("cases", []) if case.get("split") == "test"]


def paths(setting: str) -> Dict[str, Path]:
    root = RESULT_ROOTS[setting]
    red_dir = root / "red_team"
    blue_dir = root / "blue_team"
    return {
        "red_dir": red_dir,
        "red_run": red_dir / "red_run.json",
        "red_artifact": red_dir / "red_artifact.json",
        "red_trace": red_dir / "red_trace.jsonl",
        "red_log": red_dir / "red_run.log",
        "red_eval_json": red_dir / "task_a_eval.json",
        "red_eval_csv": red_dir / "task_a_eval.csv",
        "red_eval_log": red_dir / "task_a_eval.log",
        "blue_dir": blue_dir,
        "blue_run": blue_dir / "blue_run.json",
        "blue_artifact": blue_dir / "blue_artifact.json",
        "blue_trace": blue_dir / "blue_trace.jsonl",
        "blue_log": blue_dir / "blue_run.log",
        "blue_eval_json": blue_dir / "task_b_eval.json",
        "blue_eval_csv": blue_dir / "task_b_eval.csv",
        "blue_eval_log": blue_dir / "task_b_eval.log",
    }


def add_generation_args(command: List[str], args: argparse.Namespace) -> None:
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")
    command.append("--case-id")
    command.extend(test_case_ids())


def add_eval_args(command: List[str], args: argparse.Namespace) -> None:
    command.extend(["--case-registry-path", str(REGISTRY_PATH)])
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")


def red_command(setting: str, args: argparse.Namespace) -> List[str]:
    p = paths(setting)
    runner = (
        "run_red_team_ma_safr.py"
        if setting == "with"
        else "run_red_team_without_rssg.py"
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / runner),
        "--run-tag",
        RUN_TAGS[setting],
        "--api-key-env",
        GPT_API_KEY_ENV,
        "--base-url-env",
        GPT_BASE_URL_ENV,
        "--model-env-var",
        GPT_MODEL_ENV_VAR,
        "--prompt-path",
        str(PROMPTS[setting]["red"]),
        "--kg-dir",
        str(KG_DIR),
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
        "--case-registry-path",
        str(REGISTRY_PATH),
    ]
    add_generation_args(command, args)
    return command


def red_eval_command(setting: str, args: argparse.Namespace) -> List[str]:
    p = paths(setting)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_threat_validity_by_split.py"),
        "--split",
        "test",
        "--run-tag",
        RUN_TAGS[setting],
        "--predictions-path",
        str(p["red_artifact"]),
        "--output-json",
        str(p["red_eval_json"]),
        "--output-csv",
        str(p["red_eval_csv"]),
        "--log-path",
        str(p["red_eval_log"]),
    ]
    add_eval_args(command, args)
    return command


def blue_command(setting: str, args: argparse.Namespace) -> List[str]:
    p = paths(setting)
    runner = (
        "run_blue_team_ma_safr.py"
        if setting == "with"
        else "run_blue_team_without_rssg.py"
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / runner),
        "--run-tag",
        RUN_TAGS[setting],
        "--source-run-tag",
        RUN_TAGS[setting],
        "--api-key-env",
        GPT_API_KEY_ENV,
        "--base-url-env",
        GPT_BASE_URL_ENV,
        "--model-env-var",
        GPT_MODEL_ENV_VAR,
        "--prompt-path",
        str(PROMPTS[setting]["blue"]),
        "--kg-dir",
        str(KG_DIR),
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
    add_generation_args(command, args)
    return command


def blue_eval_command(setting: str, args: argparse.Namespace) -> List[str]:
    p = paths(setting)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pipeline_validity_by_split.py"),
        "--split",
        "test",
        "--run-tag",
        RUN_TAGS[setting],
        "--source-run-tag",
        RUN_TAGS[setting],
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
    add_eval_args(command, args)
    return command


def command_for_stage(stage: str, args: argparse.Namespace) -> List[str]:
    setting, operation = stage.split("-", 1)
    commands = {
        "red": red_command,
        "red-eval": red_eval_command,
        "blue": blue_command,
        "blue-eval": blue_eval_command,
    }
    return commands[operation](setting, args)


def run(command: Sequence[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=BASE_DIR, check=True)


def main() -> None:
    args = parse_args()
    clear_dead_local_proxy_env()
    print(f"[Config] split=test_1.0 case_count={len(test_case_ids())}", flush=True)
    print(f"[Config] kg_dir={KG_DIR}", flush=True)
    for setting in ("with", "without"):
        print(f"[Config] {setting}_red_prompt={PROMPTS[setting]['red']}", flush=True)
        print(f"[Config] {setting}_blue_prompt={PROMPTS[setting]['blue']}", flush=True)
    for stage in args.stages:
        print(f"[Stage] {stage}", flush=True)
        run(command_for_stage(stage, args), args.dry_run)


if __name__ == "__main__":
    main()
