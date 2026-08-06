#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run RecEvoGraphRAG Blue Team generation for feedback_weight_alpha sensitivity."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from edge_weight_common import (
    alpha_tag,
    experiment_dir,
    feedback_alpha_kg_dir,
    parse_feedback_alpha_from_argv,
    require_feedback_alpha_kg,
    write_json_direct,
)

import run_blue_team_ma_safr as base_runner


_BASE_PATCH_META = base_runner.patch_meta
CURRENT_ALPHA = 0.0
CURRENT_ALPHA_TAG = alpha_tag(CURRENT_ALPHA)
CURRENT_KG_DIR: Path | None = None


def resolve_model_tag(args: Any, model_name: str) -> str:
    return args.run_tag or (
        "deepseek-v32"
        if model_name == os.getenv("MODEL_DEEPSEEK_V32")
        else ("qwen35plus" if model_name == os.getenv("MODEL_QWEN35_PLUS") else base_runner.model_slug(model_name))
    )


def resolve_paths(args: Any, model_name: str) -> Tuple[str, str]:
    blue_tag = resolve_model_tag(args, model_name)
    source_tag = args.source_run_tag or blue_tag
    red_dir = experiment_dir(CURRENT_ALPHA, source_tag, "red_team")
    blue_dir = experiment_dir(CURRENT_ALPHA, blue_tag, "blue_team")

    if args.result_dir is None:
        args.result_dir = blue_dir
    if args.source_run_output_path is None:
        args.source_run_output_path = red_dir / f"saafg_redteam_recevographrag_{CURRENT_ALPHA_TAG}_v0_2_{source_tag}.json"
    if args.source_eval_output_path is None:
        args.source_eval_output_path = (
            red_dir / f"saafg_redteam_task_a_eval_recevographrag_{CURRENT_ALPHA_TAG}_v0_2_{source_tag}.json"
        )
    if args.output_path is None:
        args.output_path = args.result_dir / f"saafg_blueteam_recevographrag_{CURRENT_ALPHA_TAG}_v0_2_{blue_tag}.json"
    if args.artifact_path is None:
        args.artifact_path = (
            args.result_dir / f"saafg_security_augmented_flows_pred_recevographrag_{CURRENT_ALPHA_TAG}_v0_2_{blue_tag}.json"
        )
    if args.retrieval_trace_path is None:
        args.retrieval_trace_path = (
            args.result_dir / f"saafg_blueteam_retrieval_trace_recevographrag_{CURRENT_ALPHA_TAG}_v0_2_{blue_tag}.jsonl"
        )
    if args.log_path is None:
        args.log_path = args.result_dir / f"saafg_blueteam_recevographrag_{CURRENT_ALPHA_TAG}_v0_2_{blue_tag}.log"
    return blue_tag, source_tag


def patch_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    patched = _BASE_PATCH_META(meta)
    patched.update(
        {
            "sensitivity_dimension": "feedback_weight_alpha",
            "sensitivity_value": CURRENT_ALPHA,
            "feedback_weight_alpha": CURRENT_ALPHA,
            "feedback_weight_alpha_label": CURRENT_ALPHA_TAG,
            "kg_dir": str(CURRENT_KG_DIR),
            "baseline_reference": "current ma_RecEvoGraphRAG iter_01 is approximately alpha=1.0",
            "output_isolation": "feedback-alpha-specific experiment directory and filenames",
        }
    )
    return patched


def main() -> None:
    global CURRENT_ALPHA, CURRENT_ALPHA_TAG, CURRENT_KG_DIR
    skip_kg_requirement = any(arg in {"-h", "--help", "--probe-only"} for arg in sys.argv[1:])
    CURRENT_ALPHA = parse_feedback_alpha_from_argv()
    CURRENT_ALPHA_TAG = alpha_tag(CURRENT_ALPHA)
    CURRENT_KG_DIR = feedback_alpha_kg_dir(CURRENT_ALPHA) if skip_kg_requirement else require_feedback_alpha_kg(CURRENT_ALPHA)

    base_runner.DEFAULT_EVO_KG_DIR = Path(CURRENT_KG_DIR)
    base_runner.static_runner.write_json_atomic = write_json_direct
    base_runner.resolve_paths = resolve_paths
    base_runner.patch_meta = patch_meta
    base_runner.main()


if __name__ == "__main__":
    main()
