#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch runner for RecEvoGraphRAG feedback_weight_alpha sensitivity experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from edge_weight_common import DEFAULT_ALPHAS, alpha_tag, feedback_alpha_kg_dir, validate_alphas


SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "qwen35plus": {
        "run_tag": "qwen35plus",
        "model_env_var": "MODEL_QWEN35_PLUS",
        "judge_model_env_var": "MODEL_QWEN35_PLUS",
    },
    "deepseek-v32": {
        "run_tag": "deepseek-v32",
        "model_env_var": "MODEL_DEEPSEEK_V32",
        "judge_model_env_var": "MODEL_QWEN35_PLUS",
    },
}

STAGE_SCRIPTS: Dict[str, str] = {
    "red": "run_red_team_by_edge_weight.py",
    "red-eval": "evaluate_threat_validity_by_edge_weight.py",
    "blue": "run_blue_team_by_edge_weight.py",
    "blue-eval": "evaluate_pipeline_validity_by_edge_weight.py",
}

BUILD_SCRIPT = "build_edge_weight_graphs.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RecEvoGraphRAG feedback_weight_alpha batches.")
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["qwen35plus"])
    parser.add_argument("--stages", nargs="+", choices=list(STAGE_SCRIPTS), default=list(STAGE_SCRIPTS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-kg-build", action="store_true")
    parser.add_argument("--overwrite-kg", action="store_true", help="Pass --overwrite when building missing alpha KGs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def alpha_arg(alpha: float) -> str:
    return f"{alpha:.6g}"


def kg_is_ready(alpha: float) -> bool:
    kg_dir = feedback_alpha_kg_dir(alpha)
    return all(
        (kg_dir / name).exists()
        for name in ["graph_nodes.json", "graph_edges.json", "graph_metadata.json", "networkx_graph.pkl"]
    )


def run_command(command: Sequence[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def ensure_alpha_kgs(alphas: Sequence[float], args: argparse.Namespace) -> None:
    if args.skip_kg_build:
        return
    missing = [alpha for alpha in alphas if not kg_is_ready(alpha)]
    if not missing:
        return
    command = [sys.executable, str(SCRIPT_DIR / BUILD_SCRIPT), "--alphas"]
    command.extend(alpha_arg(alpha) for alpha in missing)
    if args.overwrite_kg:
        command.append("--overwrite")
    run_command(command, args.dry_run)


def stage_command(stage: str, model_key: str, alpha: float, args: argparse.Namespace) -> List[str]:
    config = MODEL_CONFIGS[model_key]
    command = [
        sys.executable,
        str(SCRIPT_DIR / STAGE_SCRIPTS[stage]),
        "--feedback-weight-alpha",
        alpha_arg(alpha),
        "--run-tag",
        config["run_tag"],
    ]
    if stage in {"red", "blue"}:
        command.extend(["--model-env-var", config["model_env_var"]])
    if stage in {"red-eval", "blue-eval"}:
        command.extend(["--model-env-var", config["judge_model_env_var"]])
    if stage in {"blue", "blue-eval"}:
        command.extend(["--source-run-tag", config["run_tag"]])
    if args.resume:
        command.append("--resume")
    if args.skip_probe:
        command.append("--skip-probe")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.case_id:
        command.append("--case-id")
        command.extend(args.case_id)
    return command


def main() -> None:
    args = parse_args()
    alphas = validate_alphas(args.alphas)
    ensure_alpha_kgs(alphas, args)

    for alpha in alphas:
        for model_key in args.models:
            for stage in args.stages:
                print(f"[Batch] alpha={alpha_tag(alpha)} model={model_key} stage={stage}")
                run_command(stage_command(stage, model_key, alpha, args), args.dry_run)


if __name__ == "__main__":
    main()
