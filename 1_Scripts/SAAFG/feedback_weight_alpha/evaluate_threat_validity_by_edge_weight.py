#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate RecEvoGraphRAG feedback_weight_alpha Red Team Task A outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from edge_weight_common import alpha_tag, experiment_dir, parse_feedback_alpha_from_argv, write_json_direct

import evaluate_threat_validity as evaluator


_BASE_PARSE_ARGS = evaluator.parse_args
CURRENT_ALPHA = 0.0
CURRENT_ALPHA_TAG = alpha_tag(CURRENT_ALPHA)


def parse_args_with_alpha(alpha: float) -> Any:
    args = _BASE_PARSE_ARGS()
    args.feedback_weight_alpha = alpha
    return args


def resolve_paths(args: Any) -> None:
    run_tag = args.run_tag or evaluator.DEFAULT_RUN_TAG
    output_dir = experiment_dir(args.feedback_weight_alpha, run_tag, "red_team")
    label = alpha_tag(args.feedback_weight_alpha)
    if args.predictions_path is None:
        args.predictions_path = output_dir / f"saafg_threat_records_pred_recevographrag_{label}_v0_2_{run_tag}.json"
    if args.output_json is None:
        args.output_json = output_dir / f"saafg_redteam_task_a_eval_recevographrag_{label}_v0_2_{run_tag}.json"
    if args.output_csv is None:
        args.output_csv = output_dir / f"saafg_redteam_task_a_eval_recevographrag_{label}_v0_2_{run_tag}.csv"
    if args.log_path is None:
        args.log_path = output_dir / f"saafg_redteam_task_a_eval_recevographrag_{label}_v0_2_{run_tag}.log"


def write_json_atomic(path: Path, payload: Any) -> None:
    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
        meta: Dict[str, Any] = payload["meta"]
        if meta.get("task") == "SAAFG Task A evaluation":
            meta.update(
                {
                    "rag_method": "RecEvoGraphRAG",
                    "sensitivity_dimension": "feedback_weight_alpha",
                    "sensitivity_value": CURRENT_ALPHA,
                    "feedback_weight_alpha": CURRENT_ALPHA,
                    "feedback_weight_alpha_label": CURRENT_ALPHA_TAG,
                    "baseline_reference": "current ma_RecEvoGraphRAG iter_01 is approximately alpha=1.0",
                }
            )
    write_json_direct(path, payload)


def main() -> None:
    global CURRENT_ALPHA, CURRENT_ALPHA_TAG
    CURRENT_ALPHA = parse_feedback_alpha_from_argv()
    CURRENT_ALPHA_TAG = alpha_tag(CURRENT_ALPHA)

    evaluator.parse_args = lambda: parse_args_with_alpha(CURRENT_ALPHA)
    evaluator.resolve_paths = resolve_paths
    evaluator.write_json_atomic = write_json_atomic
    evaluator.main()


if __name__ == "__main__":
    main()
