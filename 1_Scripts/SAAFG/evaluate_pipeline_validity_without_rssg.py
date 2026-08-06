#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Task B evaluation wrapper for RecEvoGraphRAG-without-RSSG outputs."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Tuple

import evaluate_pipeline_validity as evaluator


BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"


def resolve_paths(args: Namespace) -> Tuple[str, str]:
    run_tag = args.run_tag or evaluator.DEFAULT_RUN_TAG
    source_tag = args.source_run_tag or run_tag
    blue_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_RecEvoGraphRAG_without_RSSG_{run_tag}" / "blue_team"
    red_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_RecEvoGraphRAG_without_RSSG_{source_tag}" / "red_team"

    if args.blue_run_path is None:
        args.blue_run_path = blue_dir / f"saafg_blueteam_recevographrag_without_rssg_v0_2_{run_tag}.json"
    if args.blue_artifact_path is None:
        args.blue_artifact_path = (
            blue_dir / f"saafg_security_augmented_flows_pred_recevographrag_without_rssg_v0_2_{run_tag}.json"
        )
    if args.red_run_path is None:
        args.red_run_path = red_dir / f"saafg_redteam_recevographrag_without_rssg_v0_2_{source_tag}.json"
    if args.red_eval_path is None:
        args.red_eval_path = red_dir / f"saafg_redteam_task_a_eval_recevographrag_without_rssg_v0_2_{source_tag}.json"
    if args.output_json is None:
        args.output_json = blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_without_rssg_v0_2_{run_tag}.json"
    if args.output_csv is None:
        args.output_csv = blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_without_rssg_v0_2_{run_tag}.csv"
    if args.log_path is None:
        args.log_path = blue_dir / f"saafg_blueteam_task_b_eval_recevographrag_without_rssg_v0_2_{run_tag}.log"
    return run_tag, source_tag


def main() -> None:
    evaluator.resolve_paths = resolve_paths
    evaluator.main()


if __name__ == "__main__":
    main()
