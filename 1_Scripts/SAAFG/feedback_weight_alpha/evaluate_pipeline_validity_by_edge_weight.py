#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate RecEvoGraphRAG feedback_weight_alpha Blue Team Task B outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from edge_weight_common import alpha_tag, experiment_dir, parse_feedback_alpha_from_argv, write_json_direct

import evaluate_pipeline_validity as evaluator


_BASE_PARSE_ARGS = evaluator.parse_args
_BASE_WRITE_OUTPUTS = evaluator.write_outputs
CURRENT_ALPHA = 0.0
CURRENT_ALPHA_TAG = alpha_tag(CURRENT_ALPHA)


def parse_args_with_alpha(alpha: float) -> Any:
    args = _BASE_PARSE_ARGS()
    args.feedback_weight_alpha = alpha
    return args


def resolve_paths(args: Any) -> Tuple[str, str]:
    run_tag = args.run_tag or evaluator.DEFAULT_RUN_TAG
    source_tag = args.source_run_tag or run_tag
    label = alpha_tag(args.feedback_weight_alpha)
    blue_dir = experiment_dir(args.feedback_weight_alpha, run_tag, "blue_team")
    red_dir = experiment_dir(args.feedback_weight_alpha, source_tag, "red_team")

    if args.blue_run_path is None:
        args.blue_run_path = blue_dir / f"saafg_blueteam_recevographrag_{label}_v0_2_{run_tag}.json"
    if args.blue_artifact_path is None:
        args.blue_artifact_path = (
            blue_dir / f"saafg_security_augmented_flows_pred_recevographrag_{label}_v0_2_{run_tag}.json"
        )
    if args.red_run_path is None:
        args.red_run_path = red_dir / f"saafg_redteam_recevographrag_{label}_v0_2_{source_tag}.json"
    if args.red_eval_path is None:
        args.red_eval_path = red_dir / f"saafg_redteam_task_a_eval_recevographrag_{label}_v0_2_{source_tag}.json"
    if args.output_json is None:
        args.output_json = blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_{label}_v0_2_{run_tag}.json"
    if args.output_csv is None:
        args.output_csv = blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_{label}_v0_2_{run_tag}.csv"
    if args.log_path is None:
        args.log_path = blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_{label}_v0_2_{run_tag}.log"
    return run_tag, source_tag


def write_outputs(args: Any, meta: Dict[str, Any], case_reports: Sequence[Dict[str, Any]]) -> None:
    patched_meta = dict(meta)
    patched_meta.update(
        {
            "rag_method": "RecEvoGraphRAG",
            "sensitivity_dimension": "feedback_weight_alpha",
            "sensitivity_value": CURRENT_ALPHA,
            "feedback_weight_alpha": CURRENT_ALPHA,
            "feedback_weight_alpha_label": CURRENT_ALPHA_TAG,
            "baseline_reference": "current ma_RecEvoGraphRAG iter_01 is approximately alpha=1.0",
        }
    )
    _BASE_WRITE_OUTPUTS(args, patched_meta, case_reports)


def main() -> None:
    global CURRENT_ALPHA, CURRENT_ALPHA_TAG
    CURRENT_ALPHA = parse_feedback_alpha_from_argv()
    CURRENT_ALPHA_TAG = alpha_tag(CURRENT_ALPHA)

    evaluator.parse_args = lambda: parse_args_with_alpha(CURRENT_ALPHA)
    evaluator.resolve_paths = resolve_paths
    evaluator.write_json_atomic = write_json_direct
    evaluator.write_outputs = write_outputs
    evaluator.main()


if __name__ == "__main__":
    main()
