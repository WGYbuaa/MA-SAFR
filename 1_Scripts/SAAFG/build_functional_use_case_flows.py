#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from safr_common import clean_flow_step, normalize_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PATH = (
    ROOT
    / "0_Data"
    / "3_Experiment_Result"
    / "baseline_MA_NoRAG_CriticEvalOnly-1Pass_qwen35plus"
    / "baseline_MA_NoRAG_CriticEvalOnly-1Pass_qwen35plus.json"
)
DEFAULT_OUTPUT_PATH = (
    ROOT
    / "0_Data"
    / "6_SAAFG"
    / "1_Input_Functional_Flows"
    / "input_functional_use_case_flows.json"
)


def load_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' to be a list in {path}")
    return results


def map_step_id_to_bf(step_index: int) -> str:
    return f"BF{step_index + 1}"


def convert_flow(result: Dict[str, Any], case_index: int) -> Dict[str, Any]:
    system_flow = result.get("system_flow", [])
    if not isinstance(system_flow, list):
        raise ValueError(f"Result index {case_index} has invalid system_flow")

    source_to_target_id: Dict[str, str] = {}
    for idx, step in enumerate(system_flow):
        source_to_target_id[str(step.get("step_id"))] = map_step_id_to_bf(idx)

    requirement_text = normalize_text(
        result.get("requirement_text") or result.get("raw_requirement") or ""
    )

    basic_flow: List[Dict[str, Any]] = []
    for idx, step in enumerate(system_flow):
        source_flow_from = step.get("flow_from")
        flow_from = None
        if source_flow_from is not None:
            flow_from = source_to_target_id.get(str(source_flow_from))
        basic_flow.append(
            clean_flow_step(
                step=step,
                requirement_text=requirement_text,
                bf_step_id=map_step_id_to_bf(idx),
                flow_from=flow_from,
            )
        )

    return {
        "use_case_id": f"UC{case_index + 1:04d}",
        "source_result_index": result.get("index", case_index),
        "source_requirement_text": requirement_text,
        "input_flow_version": "v2_bf_only_cleaned",
        "basic_flow": basic_flow,
        "alternative_flows": [],
    }


def build_output(results: List[Dict[str, Any]], source_path: Path) -> Dict[str, Any]:
    flows = [convert_flow(result, idx) for idx, result in enumerate(results)]
    return {
        "meta": {
            "task": "Security-Augmented Alternative Flow Generation",
            "dataset_name": "saafg_input_functional_use_case_flows_v2_cleaned",
            "source_result_file": str(source_path),
            "source_experiment_id": "baseline_MA_NoRAG_CriticEvalOnly-1Pass_qwen35plus",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_count": len(flows),
            "notes": [
                "This v2 input dataset is converted from parser-produced system_flow outputs.",
                "All imported steps are treated as Basic Flow (BF) steps.",
                "The v2 converter repairs nominalized parser objects using step-sentence and requirement-text context.",
                "The fields security_controls and flow_type are removed from the SAAFG input schema.",
            ],
            "schema": {
                "per_case_fields": [
                    "use_case_id",
                    "source_result_index",
                    "source_requirement_text",
                    "input_flow_version",
                    "basic_flow",
                    "alternative_flows",
                ],
                "basic_flow_step_fields": [
                    "step_id",
                    "source_step_id",
                    "step_sentence",
                    "subject",
                    "verb",
                    "object",
                    "flow_from",
                ],
            },
        },
        "use_case_flows": flows,
    }


def main() -> None:
    results = load_results(DEFAULT_SOURCE_PATH)
    output_payload = build_output(results, DEFAULT_SOURCE_PATH)
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(output_payload['use_case_flows'])} cleaned SAAFG input flows.")
    print(f"Output path: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
