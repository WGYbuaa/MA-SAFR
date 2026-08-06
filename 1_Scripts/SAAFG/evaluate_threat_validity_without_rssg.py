#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Task A evaluation wrapper for RecEvoGraphRAG-without-RSSG outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import evaluate_threat_validity as evaluator


BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"


def build_default_paths(run_tag: str) -> Tuple[Path, Path, Path, Path]:
    output_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_RecEvoGraphRAG_without_RSSG_{run_tag}" / "red_team"
    predictions_path = output_dir / f"saafg_threat_records_pred_recevographrag_without_rssg_v0_2_{run_tag}.json"
    output_json = output_dir / f"saafg_redteam_task_a_eval_recevographrag_without_rssg_v0_2_{run_tag}.json"
    output_csv = output_dir / f"saafg_redteam_task_a_eval_recevographrag_without_rssg_v0_2_{run_tag}.csv"
    log_path = output_dir / f"saafg_redteam_task_a_eval_recevographrag_without_rssg_v0_2_{run_tag}.log"
    return predictions_path, output_json, output_csv, log_path


def main() -> None:
    evaluator.build_default_paths = build_default_paths
    evaluator.main()


if __name__ == "__main__":
    main()
