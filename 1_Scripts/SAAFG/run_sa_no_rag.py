#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"

DEFAULT_INPUT_PATH = (
    SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
)
DEFAULT_REGISTRY_PATH = (
    SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
)
DEFAULT_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "sa_no_rag.txt"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"


class TeeStream:
    def __init__(self, original_stream, log_path: Path):
        self.original_stream = original_stream
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._line_buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        try:
            self.original_stream.write(text)
        except Exception:
            pass
        self._line_buffer += text
        with self.log_path.open("a", encoding="utf-8") as log_file:
            while "\n" in self._line_buffer:
                line, self._line_buffer = self._line_buffer.split("\n", 1)
                log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
            log_file.flush()
        return len(text)

    def flush(self) -> None:
        try:
            self.original_stream.flush()
        except Exception:
            pass
        if self._line_buffer:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self._line_buffer}\n")
                log_file.flush()
            self._line_buffer = ""

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_written = False
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_written = True
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            pass
    finally:
        if tmp_written:
            try:
                tmp_path.unlink()
            except (FileNotFoundError, PermissionError):
                pass
    path.write_text(text, encoding="utf-8")


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


_OUTER_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def normalize_model_output(content: str) -> Tuple[str, bool]:
    text = content.strip()
    match = _OUTER_CODE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip(), True
    return text, False


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", model_name).lower()
    slug = slug.replace("qwen35plus", "qwen35plus")
    return slug or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAAFG v0.2 NoRAG Single-Agent generation.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--case-registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--run-tag", default=None, help="Optional output tag override, e.g. qwen35plus or deepseek-v32.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--threat-artifact-path", type=Path, default=None)
    parser.add_argument("--flow-artifact-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N selected cases.")
    parser.add_argument("--case-id", nargs="*", default=None, help="Optional explicit use_case_id list.")
    parser.add_argument("--resume", action="store_true", help="Skip use_case_ids already present in output.")
    parser.add_argument("--probe-only", action="store_true", help="Only probe the configured model and exit.")
    parser.add_argument("--skip-probe", action="store_true", help="Skip startup model probe.")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--base-url-env", default="BASE_URL")
    parser.add_argument("--model-env-var", default="MODEL_QWEN35_PLUS")
    parser.add_argument("--model-name", default=None, help="Override model name directly.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Do not send Qwen thinking-control extra_body; useful for non-Qwen endpoints.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, model_name: str) -> str:
    tag = args.run_tag or ("qwen35plus" if model_name == os.getenv("MODEL_QWEN35_PLUS") else model_slug(model_name))
    if args.result_dir is None:
        args.result_dir = DEFAULT_EXPERIMENT_ROOT / f"sa_NoRAG_{tag}"
    if args.output_path is None:
        args.output_path = args.result_dir / f"saafg_singleagent_norag_v0_2_{tag}.json"
    if args.threat_artifact_path is None:
        args.threat_artifact_path = args.result_dir / f"saafg_threat_records_pred_norag_v0_2_{tag}.json"
    if args.flow_artifact_path is None:
        args.flow_artifact_path = args.result_dir / f"saafg_security_augmented_flows_pred_norag_v0_2_{tag}.json"
    if args.log_path is None:
        args.log_path = args.result_dir / f"saafg_singleagent_norag_v0_2_{tag}.log"
    return tag


def configure_stream_logging(log_path: Path) -> None:
    sys.stdout = TeeStream(sys.stdout, log_path)
    sys.stderr = TeeStream(sys.stderr, log_path)


def build_llm(args: argparse.Namespace, api_key: str, base_url: str, model_name: str) -> ChatOpenAI:
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
    }
    if not args.no_extra_body:
        kwargs["extra_body"] = {"enable_thinking": False, "thinking_budget": 0}
    return ChatOpenAI(**kwargs)


def run_model_probe(llm: ChatOpenAI, model_name: str) -> None:
    print("[Model Probe] Sending a short verification request...")
    response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    metadata = getattr(response, "response_metadata", {}) or {}
    response_model = metadata.get("model_name") or metadata.get("model") or metadata.get("model_id")
    response_content = str(getattr(response, "content", "") or "").strip()
    print(f"[Model Probe] Configured model: {model_name}")
    if response_model:
        print(f"[Model Probe] Response metadata model: {response_model}")
    print(f"[Model Probe] Response content: {response_content}")
    if not response_content:
        raise RuntimeError("[Model Probe] API returned an empty response.")


def load_functional_cases(path: Path) -> List[Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("use_case_flows")
    if not isinstance(cases, list):
        raise ValueError(f"Expected use_case_flows list in {path}")
    return cases


def load_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"Expected cases list in {path}")
    return {row["use_case_id"]: row for row in rows}


def select_cases(cases: Sequence[Dict[str, Any]], case_ids: Optional[List[str]], limit: Optional[int]) -> List[Dict[str, Any]]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case.get("use_case_id") in wanted]
        missing = sorted(wanted - {case.get("use_case_id") for case in selected})
        if missing:
            raise ValueError(f"Unknown case_id(s): {', '.join(missing)}")
    else:
        selected = list(cases)
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_model_input(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "use_case_id": case["use_case_id"],
        "basic_flow": case.get("basic_flow", []),
        "alternative_flows": case.get("alternative_flows", []),
    }


def strict_parse_json(content: str) -> Any:
    return json.loads(content.strip())


def validate_threat_records(
    parsed: Any,
    valid_bf_ids: Sequence[str],
) -> Tuple[bool, List[str], Dict[str, str]]:
    errors: List[str] = []
    threat_id_to_anchor: Dict[str, str] = {}
    valid_bf_id_set = set(valid_bf_ids)

    if not isinstance(parsed, dict):
        return False, ["Output root is not a JSON object."], threat_id_to_anchor

    threat_records = parsed.get("threat_records")
    if not isinstance(threat_records, list):
        return False, ["threat_records is missing or not a list."], threat_id_to_anchor

    expected_threat_fields = {
        "threat_id",
        "threat_name",
        "anchor_steps",
        "threat_mechanism",
        "security_impact",
    }
    for index, threat in enumerate(threat_records, start=1):
        context = f"threat_records[{index - 1}]"
        if not isinstance(threat, dict):
            errors.append(f"{context} is not an object.")
            continue

        extra_fields = sorted(set(threat.keys()) - expected_threat_fields)
        if extra_fields:
            errors.append(f"{context} has unexpected field(s): {', '.join(extra_fields)}.")

        for field in expected_threat_fields:
            if field not in threat:
                errors.append(f"{context} is missing {field}.")

        expected_threat_id = f"T{index}"
        threat_id = threat.get("threat_id")
        if threat_id != expected_threat_id:
            errors.append(f"{context}.threat_id should be {expected_threat_id}, got {threat_id}.")

        for field in ("threat_name", "threat_mechanism", "security_impact"):
            value = threat.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{context}.{field} must be a non-empty string.")

        anchors = threat.get("anchor_steps")
        if not isinstance(anchors, list) or len(anchors) != 1:
            errors.append(f"{context}.anchor_steps must contain exactly one BF step id.")
            continue
        anchor = anchors[0]
        if anchor not in valid_bf_id_set:
            errors.append(f"{context}.anchor_steps[0] is not a valid BF step id: {anchor}.")
            continue
        if isinstance(threat_id, str):
            threat_id_to_anchor[threat_id] = anchor

    return not errors, errors, threat_id_to_anchor


def validate_mitigates(
    item: Dict[str, Any],
    context: str,
    valid_threat_ids: Sequence[str],
) -> Tuple[Optional[str], List[str]]:
    errors: List[str] = []
    threat_id_set = set(valid_threat_ids)
    mitigates = item.get("mitigates")
    if not isinstance(mitigates, list) or len(mitigates) != 1:
        errors.append(f"{context}.mitigates must contain exactly one threat id.")
        return None, errors
    threat_id = mitigates[0]
    if threat_id not in threat_id_set:
        errors.append(f"{context}.mitigates[0] is not a valid threat id: {threat_id}.")
        return None, errors
    return threat_id, errors


def validate_security_augmented_flow(
    parsed: Any,
    threat_schema_valid: bool,
    threat_records: Sequence[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(parsed, dict):
        return False, ["Output root is not a JSON object."]
    if not threat_schema_valid:
        return False, ["Cannot validate security_augmented_flow without valid threat_records."]

    flow = parsed.get("security_augmented_flow")
    if not isinstance(flow, dict):
        return False, ["security_augmented_flow is missing or not an object."]

    expected_flow_fields = {"security_basic_flow", "security_alternative_flows"}
    extra_flow_fields = sorted(set(flow.keys()) - expected_flow_fields)
    if extra_flow_fields:
        errors.append(f"security_augmented_flow has unexpected field(s): {', '.join(extra_flow_fields)}.")

    basic_flow = flow.get("security_basic_flow")
    alternative_flows = flow.get("security_alternative_flows")
    if not isinstance(basic_flow, list):
        errors.append("security_augmented_flow.security_basic_flow is missing or not a list.")
        basic_flow = []
    if not isinstance(alternative_flows, list):
        errors.append("security_augmented_flow.security_alternative_flows is missing or not a list.")
        alternative_flows = []

    threat_ids = [str(threat.get("threat_id")) for threat in threat_records]
    if len(basic_flow) != len(threat_ids):
        errors.append("security_basic_flow length must equal threat_records length.")
    if len(alternative_flows) != len(threat_ids):
        errors.append("security_alternative_flows length must equal threat_records length.")

    basic_ids: List[str] = []
    expected_basic_fields = {"mitigates", "step_sentence"}
    for index, item in enumerate(basic_flow):
        context = f"security_basic_flow[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{context} is not an object.")
            continue
        extra_fields = sorted(set(item.keys()) - expected_basic_fields)
        if extra_fields:
            errors.append(f"{context} has unexpected field(s): {', '.join(extra_fields)}.")
        threat_id, mitigate_errors = validate_mitigates(item, context, threat_ids)
        errors.extend(mitigate_errors)
        if threat_id:
            basic_ids.append(threat_id)
        step_sentence = item.get("step_sentence")
        if not isinstance(step_sentence, str) or not step_sentence.strip():
            errors.append(f"{context}.step_sentence must be a non-empty string.")

    alt_ids: List[str] = []
    expected_alt_fields = {"mitigates", "entry_condition", "steps"}
    expected_step_fields = {"step_sentence"}
    for index, item in enumerate(alternative_flows):
        context = f"security_alternative_flows[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{context} is not an object.")
            continue
        extra_fields = sorted(set(item.keys()) - expected_alt_fields)
        if extra_fields:
            errors.append(f"{context} has unexpected field(s): {', '.join(extra_fields)}.")
        threat_id, mitigate_errors = validate_mitigates(item, context, threat_ids)
        errors.extend(mitigate_errors)
        if threat_id:
            alt_ids.append(threat_id)
        entry_condition = item.get("entry_condition")
        if not isinstance(entry_condition, str) or not entry_condition.strip():
            errors.append(f"{context}.entry_condition must be a non-empty string.")
        steps = item.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append(f"{context}.steps must be a non-empty list.")
            continue
        for step_index, step in enumerate(steps):
            step_context = f"{context}.steps[{step_index}]"
            if not isinstance(step, dict):
                errors.append(f"{step_context} is not an object.")
                continue
            extra_step_fields = sorted(set(step.keys()) - expected_step_fields)
            if extra_step_fields:
                errors.append(f"{step_context} has unexpected field(s): {', '.join(extra_step_fields)}.")
            step_sentence = step.get("step_sentence")
            if not isinstance(step_sentence, str) or not step_sentence.strip():
                errors.append(f"{step_context}.step_sentence must be a non-empty string.")

    for collection_name, ids in (("security_basic_flow", basic_ids), ("security_alternative_flows", alt_ids)):
        counts = Counter(ids)
        duplicates = sorted(threat_id for threat_id, count in counts.items() if count > 1)
        missing = sorted(set(threat_ids) - set(ids), key=lambda value: int(value[1:]) if value[1:].isdigit() else 10**9)
        if duplicates:
            errors.append(f"{collection_name} has duplicate mitigates threat id(s): {', '.join(duplicates)}.")
        if missing:
            errors.append(f"{collection_name} is missing defense item(s) for: {', '.join(missing)}.")

    return not errors, errors


def validate_single_agent_output(
    parsed: Any,
    valid_bf_ids: Sequence[str],
) -> Tuple[bool, bool, bool, List[str], List[str], List[str]]:
    overall_errors: List[str] = []
    if not isinstance(parsed, dict):
        return False, False, False, ["Output root is not a JSON object."], ["Output root is not a JSON object."], [
            "Output root is not a JSON object."
        ]

    expected_root_fields = {"threat_records", "security_augmented_flow"}
    extra_root_fields = sorted(set(parsed.keys()) - expected_root_fields)
    missing_root_fields = sorted(expected_root_fields - set(parsed.keys()))
    if extra_root_fields:
        overall_errors.append(f"Unexpected root field(s): {', '.join(extra_root_fields)}.")
    if missing_root_fields:
        overall_errors.append(f"Missing root field(s): {', '.join(missing_root_fields)}.")

    threat_schema_valid, threat_errors, _ = validate_threat_records(parsed, valid_bf_ids)
    threat_records = parsed.get("threat_records") if threat_schema_valid else []
    flow_schema_valid, flow_errors = validate_security_augmented_flow(
        parsed,
        threat_schema_valid,
        threat_records if isinstance(threat_records, list) else [],
    )

    overall_errors.extend(threat_errors)
    overall_errors.extend(flow_errors)
    return not overall_errors, threat_schema_valid, flow_schema_valid, overall_errors, threat_errors, flow_errors


def get_response_usage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return {"raw_usage": str(usage)}
    return usage


def jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def anchor_num_from_step_id(step_id: str) -> int:
    match = re.search(r"(\d+)$", step_id or "")
    return int(match.group(1)) if match else 1


def suffix_for_occurrence(occurrence_index: int) -> str:
    if occurrence_index <= 1:
        return ""
    return chr(ord("a") + occurrence_index - 2)


def empty_security_augmented_flow() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "security_basic_flow": [],
        "security_alternative_flows": [],
    }


def build_standardized_flow(
    threat_records: Sequence[Dict[str, Any]],
    raw_flow: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    basic_by_threat = {
        item["mitigates"][0]: item
        for item in raw_flow.get("security_basic_flow", [])
    }
    alternative_by_threat = {
        item["mitigates"][0]: item
        for item in raw_flow.get("security_alternative_flows", [])
    }
    suffix_counter: Dict[int, int] = defaultdict(int)
    security_basic_flow: List[Dict[str, Any]] = []
    security_alternative_flows: List[Dict[str, Any]] = []

    for threat in threat_records:
        threat_id = threat["threat_id"]
        anchor_step_id = str(threat["anchor_steps"][0]).strip()
        anchor_num = anchor_num_from_step_id(anchor_step_id)
        suffix_counter[anchor_num] += 1
        suffix = suffix_for_occurrence(suffix_counter[anchor_num])
        sbf_id = f"SBF{anchor_num}{suffix}"
        saf_id = f"SAF{anchor_num}{suffix}"

        basic_item = basic_by_threat[threat_id]
        alternative_item = alternative_by_threat[threat_id]
        security_basic_flow.append(
            {
                "step_id": sbf_id,
                "anchor_after": anchor_step_id,
                "step_sentence": basic_item["step_sentence"],
            }
        )
        security_alternative_flows.append(
            {
                "saf_id": saf_id,
                "mitigates": [threat_id],
                "entry_condition": alternative_item["entry_condition"],
                "steps": [
                    {
                        "step_id": f"{saf_id}.{step_index}",
                        "step_sentence": step["step_sentence"],
                    }
                    for step_index, step in enumerate(alternative_item.get("steps") or [], start=1)
                ],
            }
        )

    return {
        "security_basic_flow": security_basic_flow,
        "security_alternative_flows": security_alternative_flows,
    }


def run_single_agent_case(
    llm: ChatOpenAI,
    system_prompt: str,
    case: Dict[str, Any],
    registry_row: Dict[str, Any],
) -> Dict[str, Any]:
    case_id = case["use_case_id"]
    model_input = build_model_input(case)
    valid_bf_ids = [step.get("step_id") for step in case.get("basic_flow", []) if step.get("step_id")]
    started_at = now_utc()
    start_time = time.perf_counter()
    raw_output = ""
    parsed_output = None
    parse_valid = False
    schema_valid = False
    threat_schema_valid = False
    flow_schema_valid = False
    parse_repaired = False
    parse_error = None
    schema_errors: List[str] = []
    threat_schema_errors: List[str] = []
    flow_schema_errors: List[str] = []
    response_metadata: Dict[str, Any] = {}

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))),
            ]
        )
        raw_output = str(getattr(response, "content", "") or "")
        response_metadata = jsonable(getattr(response, "response_metadata", {}) or {})
        try:
            normalized_output, parse_repaired = normalize_model_output(raw_output)
            parsed_output = strict_parse_json(normalized_output)
            parse_valid = True
            (
                schema_valid,
                threat_schema_valid,
                flow_schema_valid,
                schema_errors,
                threat_schema_errors,
                flow_schema_errors,
            ) = validate_single_agent_output(parsed_output, valid_bf_ids)
        except Exception as exc:
            parse_error = str(exc)
    except Exception as exc:
        parse_error = f"Model invocation failed: {exc}"

    duration = time.perf_counter() - start_time
    threat_records: List[Dict[str, Any]] = []
    security_augmented_flow = empty_security_augmented_flow()
    if parse_valid and threat_schema_valid and isinstance(parsed_output, dict):
        threat_records = parsed_output.get("threat_records", [])
    if parse_valid and threat_schema_valid and flow_schema_valid and isinstance(parsed_output, dict):
        security_augmented_flow = build_standardized_flow(
            threat_records,
            parsed_output.get("security_augmented_flow") or {},
        )

    metadata = response_metadata if isinstance(response_metadata, dict) else {}
    return {
        "index": None,
        "use_case_id": case_id,
        "dataset": registry_row.get("dataset"),
        "split": registry_row.get("split"),
        "source_knowledge_id": registry_row.get("source_knowledge_id"),
        "model_input": model_input,
        "raw_model_output": raw_output,
        "parsed_output": parsed_output,
        "threat_records": threat_records,
        "security_augmented_flow": security_augmented_flow,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "threat_schema_valid": threat_schema_valid,
        "flow_schema_valid": flow_schema_valid,
        "case_parse_valid": parse_valid,
        "case_schema_valid": flow_schema_valid,
        "parse_repaired": parse_repaired,
        "parse_repair_note": "stripped outer markdown code fence" if parse_repaired else None,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "threat_schema_errors": threat_schema_errors,
        "flow_schema_errors": flow_schema_errors,
        "single_agent_duration_seconds": duration,
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "response_model": metadata.get("model_name") or metadata.get("model") or metadata.get("model_id"),
        "finish_reason": metadata.get("finish_reason"),
        "token_usage": get_response_usage(metadata),
    }


def summarize_results(results: Sequence[Dict[str, Any]], selected_case_count: int, invocation_start: float) -> Dict[str, Any]:
    durations = [float(item.get("single_agent_duration_seconds") or 0.0) for item in results]
    parse_valid_count = sum(1 for item in results if item.get("parse_valid"))
    full_schema_valid_count = sum(1 for item in results if item.get("schema_valid"))
    threat_schema_valid_count = sum(1 for item in results if item.get("threat_schema_valid"))
    flow_schema_valid_count = sum(1 for item in results if item.get("flow_schema_valid"))
    parse_repaired_count = sum(1 for item in results if item.get("parse_repaired"))
    predicted_threat_total = sum(len(item.get("threat_records") or []) for item in results)
    generated_defense_total = sum(
        len((item.get("security_augmented_flow") or {}).get("security_alternative_flows") or [])
        for item in results
    )

    usage_totals: Dict[str, float] = {}
    for item in results:
        usage = item.get("token_usage") or {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] = usage_totals.get(key, 0.0) + float(value)

    return {
        "selected_case_count": selected_case_count,
        "completed_case_count": len(results),
        "parse_valid_case_count": parse_valid_count,
        "full_schema_valid_case_count": full_schema_valid_count,
        "threat_schema_valid_case_count": threat_schema_valid_count,
        "flow_schema_valid_case_count": flow_schema_valid_count,
        "parse_repaired_case_count": parse_repaired_count,
        "failed_case_count": len(results) - full_schema_valid_count,
        "parse_valid_rate": parse_valid_count / len(results) if results else 0.0,
        "full_schema_valid_rate": full_schema_valid_count / len(results) if results else 0.0,
        "threat_schema_valid_rate": threat_schema_valid_count / len(results) if results else 0.0,
        "flow_schema_valid_rate": flow_schema_valid_count / len(results) if results else 0.0,
        "predicted_threat_total": predicted_threat_total,
        "generated_defense_total": generated_defense_total,
        "total_case_duration_seconds": sum(durations),
        "average_case_duration_seconds": sum(durations) / len(durations) if durations else 0.0,
        "current_invocation_wall_time_seconds": time.perf_counter() - invocation_start,
        "token_usage_totals": usage_totals,
    }


def write_run_output(
    output_path: Path,
    meta: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    selected_case_count: int,
    invocation_start: float,
) -> None:
    payload = {
        "meta": {**meta, "updated_at_utc": now_utc()},
        "results": list(results),
        "summary": summarize_results(results, selected_case_count, invocation_start),
    }
    write_json_atomic(output_path, payload)


def write_threat_artifact(
    artifact_path: Path,
    meta: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    invalid_case_ids = [item["use_case_id"] for item in results if not item.get("threat_schema_valid")]
    cases = []
    for item in results:
        cases.append(
            {
                "use_case_id": item["use_case_id"],
                "dataset": item.get("dataset"),
                "split": item.get("split"),
                "source_knowledge_id": item.get("source_knowledge_id"),
                "threat_records": item.get("threat_records") if item.get("threat_schema_valid") else [],
            }
        )

    payload = {
        "meta": {
            "dataset_name": "SAAFG Single-Agent NoRAG predicted threat records",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "source_run_output_path": str(output_path),
            "experiment_id": meta["experiment_id"],
            "case_count": len(cases),
            "invalid_case_count": len(invalid_case_ids),
            "invalid_case_ids": invalid_case_ids,
            "notes": [
                "Threat artifact is exported from section-level threat_records schema validity.",
                "No retries, JSON repair, schema correction, or missing-field imputation are applied.",
                "Invalid threat sections are represented as empty threat_records; raw failures remain in the run output.",
            ],
        },
        "threat_record_cases": cases,
    }
    write_json_atomic(artifact_path, payload)


def write_flow_artifact(
    artifact_path: Path,
    meta: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    invalid_case_ids = [item["use_case_id"] for item in results if not item.get("flow_schema_valid")]
    cases = []
    for item in results:
        cases.append(
            {
                "use_case_id": item["use_case_id"],
                "dataset": item.get("dataset"),
                "split": item.get("split"),
                "source_knowledge_id": item.get("source_knowledge_id"),
                "security_augmented_flow": item.get("security_augmented_flow")
                if item.get("flow_schema_valid")
                else empty_security_augmented_flow(),
            }
        )

    payload = {
        "meta": {
            "dataset_name": "SAAFG Single-Agent NoRAG predicted security augmented flows",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "source_run_output_path": str(output_path),
            "experiment_id": meta["experiment_id"],
            "case_count": len(cases),
            "invalid_case_count": len(invalid_case_ids),
            "invalid_case_ids": invalid_case_ids,
            "notes": [
                "Security-augmented flow artifact is exported from section-level flow schema validity.",
                "SBF/SAF ids and anchor_after are attached deterministically from mitigates -> threat_id -> anchor_steps.",
                "Invalid flow sections are represented as empty security_augmented_flow; raw failures remain in the run output.",
            ],
        },
        "security_augmented_flow_cases": cases,
    }
    write_json_atomic(artifact_path, payload)


def load_existing_results(output_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = read_json(output_path)
    meta = payload.get("meta") or {}
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise ValueError(f"Existing output has invalid results field: {output_path}")
    return meta, results


def main() -> None:
    args = parse_args()
    load_dotenv(BASE_DIR / "1_Scripts" / ".env", override=True)

    api_key = os.getenv(args.api_key_env)
    base_url = os.getenv(args.base_url_env)
    model_name = args.model_name or os.getenv(args.model_env_var)
    missing = [
        name
        for name, value in {
            args.api_key_env: api_key,
            args.base_url_env: base_url,
            args.model_env_var if not args.model_name else "model_name": model_name,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment/config values: {', '.join(missing)}")

    run_tag = resolve_paths(args, model_name)
    configure_stream_logging(args.log_path)

    print(f"[Config] run_tag={run_tag}")
    print(f"[Config] input_path={args.input_path}")
    print(f"[Config] case_registry_path={args.case_registry_path}")
    print(f"[Config] prompt_path={args.prompt_path}")
    print(f"[Config] output_path={args.output_path}")
    print(f"[Config] threat_artifact_path={args.threat_artifact_path}")
    print(f"[Config] flow_artifact_path={args.flow_artifact_path}")
    print(f"[Config] model={model_name}")
    print("[Config] policy=no retries, no repair, no imputation")
    print("[Config] parse_normalization=strip_outer_markdown_code_fence")

    llm = build_llm(args, api_key, base_url, model_name)
    if not args.skip_probe:
        run_model_probe(llm, model_name)
    if args.probe_only:
        return

    system_prompt = load_text(args.prompt_path)
    cases = load_functional_cases(args.input_path)
    registry_map = load_registry(args.case_registry_path)
    selected_cases = select_cases(cases, args.case_id, args.limit)

    existing_meta: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []
    if args.resume and args.output_path.exists():
        existing_meta, results = load_existing_results(args.output_path)
        print(f"[Resume] Loaded {len(results)} existing result(s).")

    completed_ids = {item.get("use_case_id") for item in results}
    experiment_id = args.output_path.stem
    meta = {
        **existing_meta,
        "experiment_id": existing_meta.get("experiment_id") or experiment_id,
        "task": "SAAFG Single-Agent NoRAG generation",
        "version": "v0.2",
        "no_rag": True,
        "run_tag": run_tag,
        "model_name": model_name,
        "model_env_var": args.model_env_var,
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
        "input_path": str(args.input_path),
        "case_registry_path": str(args.case_registry_path),
        "prompt_path": str(args.prompt_path),
        "started_at_utc": existing_meta.get("started_at_utc") or now_utc(),
        "generation_policy": "no retries, no JSON repair, no schema correction, no missing-field imputation",
        "parse_normalization": "strip_outer_markdown_code_fence_before_strict_json_parse",
        "artifact_policy": "Task A and Task B artifacts are exported from their section-level schema validity.",
    }

    invocation_start = time.perf_counter()
    total = len(selected_cases)
    print(f"[Run] Selected {total} case(s).")

    for position, case in enumerate(selected_cases, start=1):
        case_id = case["use_case_id"]
        if case_id in completed_ids:
            print(f"[Skip] {position}/{total} {case_id} already completed.")
            continue
        registry_row = registry_map.get(case_id)
        if not registry_row:
            raise ValueError(f"Missing case registry row for {case_id}")

        print(f"[Run] {position}/{total} {case_id}")
        result = run_single_agent_case(llm, system_prompt, case, registry_row)
        result["index"] = position - 1
        results.append(result)
        completed_ids.add(case_id)
        print(
            "[Done] {} full_schema={} threat_schema={} flow_schema={} duration={:.3f}s threats={} defenses={}".format(
                case_id,
                "valid" if result.get("schema_valid") else "invalid",
                "valid" if result.get("threat_schema_valid") else "invalid",
                "valid" if result.get("flow_schema_valid") else "invalid",
                float(result.get("single_agent_duration_seconds") or 0.0),
                len(result.get("threat_records") or []),
                len((result.get("security_augmented_flow") or {}).get("security_alternative_flows") or []),
            )
        )
        write_run_output(args.output_path, meta, results, total, invocation_start)
        write_threat_artifact(args.threat_artifact_path, meta, results, args.output_path)
        write_flow_artifact(args.flow_artifact_path, meta, results, args.output_path)

    write_run_output(args.output_path, meta, results, total, invocation_start)
    write_threat_artifact(args.threat_artifact_path, meta, results, args.output_path)
    write_flow_artifact(args.flow_artifact_path, meta, results, args.output_path)
    summary = summarize_results(results, total, invocation_start)
    print("[Summary] completed={completed_case_count}/{selected_case_count}".format(**summary))
    print("[Summary] full_schema_valid_rate={:.4f}".format(summary["full_schema_valid_rate"]))
    print("[Summary] threat_schema_valid_rate={:.4f}".format(summary["threat_schema_valid_rate"]))
    print("[Summary] flow_schema_valid_rate={:.4f}".format(summary["flow_schema_valid_rate"]))
    if summary.get("parse_repaired_case_count"):
        print("[Summary] parse_repaired_case_count={}".format(summary["parse_repaired_case_count"]))
    print("[Summary] predicted_threat_total={}".format(summary["predicted_threat_total"]))
    print("[Summary] generated_defense_total={}".format(summary["generated_defense_total"]))
    print("[Summary] total_case_duration_seconds={:.3f}".format(summary["total_case_duration_seconds"]))
    print("[Summary] average_case_duration_seconds={:.3f}".format(summary["average_case_duration_seconds"]))
    print(f"[Summary] output_path={args.output_path}")
    print(f"[Summary] threat_artifact_path={args.threat_artifact_path}")
    print(f"[Summary] flow_artifact_path={args.flow_artifact_path}")


if __name__ == "__main__":
    main()
