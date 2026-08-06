#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run SAAFG v0.2 Rec-EvoGraphRAG-without-RSSG Red Team generation.

This is a prompt-ablation wrapper around the verified RecEvoGraphRAG setup.
It keeps the evolved KG, retriever, model settings, and generation policy
unchanged while replacing only the Red Team prompt with the without-RSSG prompt
and writing to a separate experiment directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Sequence

import run_red_team_static_graph_rag as static_runner
from evographrag_weighted_retrieval import DEFAULT_EVO_KG_DIR, RecEvoGraphRetriever


BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
DEFAULT_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "red_team_without_rssg.txt"


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", model_name).lower()
    slug = slug.replace("qwen35plus", "qwen35plus")
    return slug or "model"


def resolve_paths(args: Any, model_name: str) -> None:
    tag = args.run_tag or (
        "deepseek-v32"
        if model_name == os.getenv("MODEL_DEEPSEEK_V32")
        else ("qwen35plus" if model_name == os.getenv("MODEL_QWEN35_PLUS") else model_slug(model_name))
    )
    if args.result_dir is None:
        args.result_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_RecEvoGraphRAG_without_RSSG_{tag}" / "red_team"
    if args.output_path is None:
        args.output_path = args.result_dir / f"saafg_redteam_recevographrag_without_rssg_v0_2_{tag}.json"
    if args.artifact_path is None:
        args.artifact_path = args.result_dir / f"saafg_threat_records_pred_recevographrag_without_rssg_v0_2_{tag}.json"
    if args.retrieval_trace_path is None:
        args.retrieval_trace_path = (
            args.result_dir / f"saafg_redteam_retrieval_trace_recevographrag_without_rssg_v0_2_{tag}.jsonl"
        )
    if args.log_path is None:
        args.log_path = args.result_dir / f"saafg_redteam_recevographrag_without_rssg_v0_2_{tag}.log"


def patch_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    patched = dict(meta)
    patched.update(
        {
            "task": "SAAFG Task A RecEvoGraphRAG without RSSG Red Team generation",
            "rag_method": "RecEvoGraphRAG_without_RSSG",
            "retriever_module": "evographrag_weighted_retrieval",
            "evo_graph_stage": "critic_feedback_edge_weight_update",
            "rssg_layer": "disabled_prompt_ablation",
            "generation_policy": (
                "RecEvoGraphRAG ranked graph retrieval over critic-evolved edge weights with "
                "graph-aware Risk -> AttackPattern -> Mitigation prompt guidance, RSSG Layer "
                "removed from the Red Team prompt, no retries, no JSON repair, no schema "
                "correction, no missing-field imputation"
            ),
        }
    )
    return patched


def install_patches() -> None:
    static_runner.DEFAULT_PROMPT_PATH = DEFAULT_PROMPT_PATH
    static_runner.DEFAULT_KG_DIR = DEFAULT_EVO_KG_DIR
    static_runner.RecEvoGraphRetriever = RecEvoGraphRetriever
    static_runner.resolve_paths = resolve_paths

    original_write_run_output = static_runner.write_run_output
    original_write_prediction_artifact = static_runner.write_prediction_artifact

    def write_run_output(
        output_path: Path,
        meta: Dict[str, Any],
        results: Sequence[Dict[str, Any]],
        selected_case_count: int,
        invocation_start: float,
    ) -> None:
        original_write_run_output(
            output_path,
            patch_meta(meta),
            results,
            selected_case_count,
            invocation_start,
        )

    def write_prediction_artifact(
        artifact_path: Path,
        meta: Dict[str, Any],
        results: Sequence[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        original_write_prediction_artifact(artifact_path, patch_meta(meta), results, output_path)
        payload = static_runner.read_json(artifact_path)
        payload.setdefault("meta", {})
        payload["meta"]["dataset_name"] = "SAAFG Red Team RecEvoGraphRAG without RSSG predicted threat records"
        payload["meta"]["rag_method"] = "RecEvoGraphRAG_without_RSSG"
        payload["meta"]["rssg_layer"] = "disabled_prompt_ablation"
        notes = payload["meta"].get("notes") or []
        if notes:
            notes[0] = (
                "Red Team receives ranked graph evidence from the critic-evolved Rec-EvoGraph-RAG "
                "knowledge graph; the RSSG Layer is removed from the prompt."
            )
        payload["meta"]["notes"] = notes
        static_runner.write_json_atomic(artifact_path, payload)

    static_runner.write_run_output = write_run_output
    static_runner.write_prediction_artifact = write_prediction_artifact


def main() -> None:
    install_patches()
    static_runner.main()


if __name__ == "__main__":
    main()
