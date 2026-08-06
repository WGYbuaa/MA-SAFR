#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Task A evaluation for SAAFG v0.2 NoRAG Red Team outputs.

Metrics:
 - primary_anchor_precision / recall: exact anchor-step matching, one-to-one.
 - threat_validity_precision / recall: same-anchor pairwise proxy judge + one-to-one matching.
 - threat_f1: harmonic mean of threat_validity precision/recall.
Macro metrics are case averages. Micro metrics are global counts.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"

DEFAULT_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "threat_matching_judge.txt"
DEFAULT_SILVER_PATH = SAAFG_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"
DEFAULT_FLOW_PATH = SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
DEFAULT_REGISTRY_PATH = SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
DEFAULT_RUN_TAG = "qwen35plus"
ALLOWED_JUDGE_REASON_CODES = {
    "same_underlying_threat",
    "mechanism_mismatch",
    "target_mismatch",
    "impact_only_match",
    "too_generic",
    "non_security",
    "unclear",
}


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
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path.write_text(text, encoding="utf-8")
    last_error: Optional[OSError] = None
    for attempt in range(30):
        try:
            os.replace(tmp_path, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(min(1.0, 0.1 * (attempt + 1)))
    raise OSError(f"Failed to replace JSON output after retries: {path}") from last_error


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def threat_sort_key(threat: Dict[str, Any]) -> Tuple[int, str]:
    threat_id = str(threat.get("threat_id") or "")
    digits = "".join(ch for ch in threat_id if ch.isdigit())
    return (int(digits) if digits else 10**9, threat_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAAFG v0.2 NoRAG Red Team Task A.")
    parser.add_argument("--run-tag", default=None, help="Optional output tag override, e.g. qwen35plus or deepseek-v32.")
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--silver-path", type=Path, default=DEFAULT_SILVER_PATH)
    parser.add_argument("--flow-path", type=Path, default=DEFAULT_FLOW_PATH)
    parser.add_argument("--case-registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--case-id", nargs="*", default=None, help="Optional explicit use_case_id list.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N selected cases.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output if present.")
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


def build_default_paths(run_tag: str) -> Tuple[Path, Path, Path, Path]:
    output_dir = SAAFG_ROOT / "6_Experiment_Result" / f"ma_NoRAG_{run_tag}" / "red_team"
    predictions_path = output_dir / f"saafg_threat_records_pred_norag_v0_2_{run_tag}.json"
    output_json = output_dir / f"saafg_redteam_task_a_eval_norag_v0_2_{run_tag}.json"
    output_csv = output_dir / f"saafg_redteam_task_a_eval_norag_v0_2_{run_tag}.csv"
    log_path = output_dir / f"saafg_redteam_task_a_eval_norag_v0_2_{run_tag}.log"
    return predictions_path, output_json, output_csv, log_path


def resolve_paths(args: argparse.Namespace) -> None:
    run_tag = args.run_tag or DEFAULT_RUN_TAG
    default_predictions_path, default_output_json, default_output_csv, default_log_path = build_default_paths(run_tag)
    if args.predictions_path is None:
        args.predictions_path = default_predictions_path
    if args.output_json is None:
        args.output_json = default_output_json
    if args.output_csv is None:
        args.output_csv = default_output_csv
    if args.log_path is None:
        args.log_path = default_log_path


def resolve_model_name(args: argparse.Namespace) -> str:
    return args.model_name or os.getenv(args.model_env_var) or ""


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


def clean_json_content(content: str) -> str:
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    elif not text.startswith("{") and not text.startswith("["):
        decoder = json.JSONDecoder()
        positions = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
        for pos in sorted(positions):
            try:
                _, end = decoder.raw_decode(text[pos:])
                text = text[pos : pos + end]
                break
            except json.JSONDecodeError:
                continue
    return text.strip()


def try_repair_json(content: str) -> str:
    repaired = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", content)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r'("\s*)\n(\s*")', r'\1,\n\2', repaired)
    repaired = re.sub(r'(}\s*)\n(\s*")', r'\1,\n\2', repaired)
    repaired = re.sub(r'(]\s*)\n(\s*")', r'\1,\n\2', repaired)
    return repaired


def safe_json_loads(content: str) -> Tuple[Any, bool]:
    cleaned = clean_json_content(content)
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        repaired = try_repair_json(cleaned)
        if repaired != cleaned:
            try:
                return json.loads(repaired), True
            except json.JSONDecodeError:
                pass
        raise


def load_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("threat_record_cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid threat_record_cases in {path}")
    return {case["use_case_id"]: case for case in cases}


def load_silver_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("threat_record_cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid threat_record_cases in {path}")
    return {case["use_case_id"]: case for case in cases}


def load_flow_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("use_case_flows")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid use_case_flows in {path}")
    return {case["use_case_id"]: case for case in cases}


def load_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases in {path}")
    return {case["use_case_id"]: case for case in cases}


def select_case_ids(
    registry: Dict[str, Dict[str, Any]],
    case_ids: Optional[List[str]],
    limit: Optional[int],
) -> List[str]:
    ordered_case_ids = [row["use_case_id"] for row in registry.values()]
    if case_ids:
        wanted = set(case_ids)
        selected = [case_id for case_id in ordered_case_ids if case_id in wanted]
        missing = sorted(wanted - set(selected))
        if missing:
            raise ValueError(f"Unknown case_id(s): {', '.join(missing)}")
    else:
        selected = ordered_case_ids
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_flow_context(flow_case: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "step_id": step.get("step_id"),
            "step_sentence": step.get("step_sentence"),
        }
        for step in flow_case.get("basic_flow", [])
    ]


def group_by_anchor(threats: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for threat in threats:
        anchors = threat.get("anchor_steps") or []
        if not anchors:
            continue
        grouped[str(anchors[0])].append(threat)
    for anchor in grouped:
        grouped[anchor] = sorted(grouped[anchor], key=threat_sort_key)
    return dict(grouped)


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def harmonic_mean(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def maximum_bipartite_matching(edge_map: Dict[int, List[int]], right_size: int) -> Tuple[int, Dict[int, int]]:
    match_right: List[int] = [-1] * right_size

    def dfs(left_idx: int, seen: List[bool]) -> bool:
        for right_idx in edge_map.get(left_idx, []):
            if seen[right_idx]:
                continue
            seen[right_idx] = True
            if match_right[right_idx] == -1 or dfs(match_right[right_idx], seen):
                match_right[right_idx] = left_idx
                return True
        return False

    match_count = 0
    for left_idx in sorted(edge_map.keys()):
        seen = [False] * right_size
        if dfs(left_idx, seen):
            match_count += 1

    matches = {right_idx: left_idx for right_idx, left_idx in enumerate(match_right) if left_idx != -1}
    return match_count, matches


def build_judge_prompt(
    flow_context: List[Dict[str, str]],
    silver_threat: Dict[str, Any],
    predicted_threat: Dict[str, Any],
    use_case_id: str,
    anchor_step: str,
) -> List[Any]:
    payload = {
        "use_case_id": use_case_id,
        "anchor_step": anchor_step,
        "functional_flow_context": flow_context,
        "silver_threat": {
            "threat_id": silver_threat.get("threat_id"),
            "threat_name": silver_threat.get("threat_name"),
            "threat_mechanism": silver_threat.get("threat_mechanism"),
            "security_impact": silver_threat.get("security_impact"),
        },
        "predicted_threat": {
            "threat_id": predicted_threat.get("threat_id"),
            "threat_name": predicted_threat.get("threat_name"),
            "threat_mechanism": predicted_threat.get("threat_mechanism"),
            "security_impact": predicted_threat.get("security_impact"),
        },
    }
    return [HumanMessage(content=json.dumps(payload, ensure_ascii=False))]


def validate_judge_output(parsed: Any) -> Tuple[bool, List[str], bool, str, str]:
    errors: List[str] = []
    if not isinstance(parsed, dict):
        return False, ["Judge output root is not a JSON object."], False, "unclear", ""

    expected_fields = {"is_same_threat", "reason_code", "reasoning"}
    extra_fields = sorted(set(parsed.keys()) - expected_fields)
    if extra_fields:
        errors.append(f"Unexpected field(s): {', '.join(extra_fields)}.")

    missing_fields = sorted(expected_fields - set(parsed.keys()))
    if missing_fields:
        errors.append(f"Missing field(s): {', '.join(missing_fields)}.")

    is_same_threat = False
    if isinstance(parsed.get("is_same_threat"), bool):
        is_same_threat = parsed["is_same_threat"]
    else:
        errors.append("is_same_threat must be a boolean.")

    reason_code = "unclear"
    parsed_reason_code = parsed.get("reason_code")
    if isinstance(parsed_reason_code, str) and parsed_reason_code in ALLOWED_JUDGE_REASON_CODES:
        reason_code = parsed_reason_code
    else:
        errors.append("reason_code is missing or invalid.")

    reasoning = ""
    if isinstance(parsed.get("reasoning"), str):
        reasoning = parsed["reasoning"]
    else:
        errors.append("reasoning must be a string.")

    return not errors, errors, is_same_threat, reason_code, reasoning


def judge_same_threat(
    llm: ChatOpenAI,
    system_prompt: str,
    flow_context: List[Dict[str, str]],
    silver_threat: Dict[str, Any],
    predicted_threat: Dict[str, Any],
    use_case_id: str,
    anchor_step: str,
) -> Dict[str, Any]:
    started_at = now_utc()
    start_time = time.perf_counter()
    raw_output = ""
    parsed: Any = None
    parse_repaired = False
    parse_valid = False
    response_metadata: Dict[str, Any] = {}
    parse_error = None
    is_same_threat = False
    reason_code = "unclear"
    reasoning = ""
    schema_valid = False
    schema_errors: List[str] = []

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), *build_judge_prompt(flow_context, silver_threat, predicted_threat, use_case_id, anchor_step)]
        )
        raw_output = str(getattr(response, "content", "") or "")
        response_metadata = getattr(response, "response_metadata", {}) or {}
        try:
            parsed, parse_repaired = safe_json_loads(raw_output)
            parse_valid = True
            schema_valid, schema_errors, is_same_threat, reason_code, reasoning = validate_judge_output(parsed)
        except Exception as exc:
            parse_error = str(exc)
    except Exception as exc:
        parse_error = f"Model invocation failed: {exc}"

    duration = time.perf_counter() - start_time
    if not parse_valid:
        reasoning = parse_error or "No valid judge output."
    elif not schema_valid:
        is_same_threat = False
        if parse_error:
            schema_errors.append(parse_error)
        if not reasoning:
            reasoning = "Invalid judge schema."

    return {
        "use_case_id": use_case_id,
        "anchor_step": anchor_step,
        "silver_threat_id": silver_threat.get("threat_id"),
        "predicted_threat_id": predicted_threat.get("threat_id"),
        "is_same_threat": is_same_threat,
        "reason_code": reason_code,
        "reasoning": reasoning,
        "raw_output": raw_output,
        "parse_valid": parse_valid,
        "parse_repaired": parse_repaired,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "parse_error": parse_error,
        "judge_duration_seconds": duration,
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "response_model": response_metadata.get("model_name")
        or response_metadata.get("model")
        or response_metadata.get("model_id"),
        "finish_reason": response_metadata.get("finish_reason"),
        "token_usage": response_metadata.get("token_usage") or response_metadata.get("usage") or {},
    }


def load_existing_output(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid output payload in {path}")
    payload.setdefault("case_reports", [])
    payload.setdefault("pairwise_judgments", [])
    payload.setdefault("summary", {})
    return payload


def compute_case_metrics(
    predicted_case: Dict[str, Any],
    silver_case: Dict[str, Any],
    flow_case: Dict[str, Any],
    judge_cache: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    llm: ChatOpenAI,
    system_prompt: str,
) -> Dict[str, Any]:
    use_case_id = predicted_case["use_case_id"]
    predicted_threats = sorted(predicted_case.get("threat_records", []), key=threat_sort_key)
    silver_threats = sorted(silver_case.get("threat_records", []), key=threat_sort_key)
    flow_context = build_flow_context(flow_case)

    pred_by_anchor = group_by_anchor(predicted_threats)
    silver_by_anchor = group_by_anchor(silver_threats)
    anchors = sorted(set(pred_by_anchor) | set(silver_by_anchor))
    shared_anchors = sorted(set(pred_by_anchor) & set(silver_by_anchor))

    predicted_total = len(predicted_threats)
    silver_total = len(silver_threats)
    anchor_match_count = sum(min(len(pred_by_anchor.get(anchor, [])), len(silver_by_anchor.get(anchor, []))) for anchor in anchors)

    judge_pair_count = 0
    judge_positive_count = 0
    pairwise_judgments: List[Dict[str, Any]] = []
    judge_token_usage_totals: Dict[str, float] = {}
    judge_parse_valid_count = 0
    judge_schema_valid_count = 0

    semantic_match_total = 0
    semantic_match_pairs: List[Dict[str, str]] = []

    for anchor in shared_anchors:
        pred_list = pred_by_anchor[anchor]
        silver_list = silver_by_anchor[anchor]
        edge_map: Dict[int, List[int]] = defaultdict(list)
        judge_pair_count += len(pred_list) * len(silver_list)

        for p_idx, predicted_threat in enumerate(pred_list):
            for s_idx, silver_threat in enumerate(silver_list):
                cache_key = (use_case_id, anchor, str(silver_threat.get("threat_id")), str(predicted_threat.get("threat_id")))
                record = judge_cache.get(cache_key)
                if record is None:
                    record = judge_same_threat(
                        llm,
                        system_prompt,
                        flow_context,
                        silver_threat,
                        predicted_threat,
                        use_case_id,
                        anchor,
                    )
                    judge_cache[cache_key] = record
                pairwise_judgments.append(record)
                if record.get("is_same_threat"):
                    edge_map[p_idx].append(s_idx)
                    judge_positive_count += 1
                if record.get("parse_valid"):
                    judge_parse_valid_count += 1
                if record.get("schema_valid"):
                    judge_schema_valid_count += 1
                usage = record.get("token_usage") or {}
                if isinstance(usage, dict):
                    for key, value in usage.items():
                        if isinstance(value, (int, float)):
                            judge_token_usage_totals[key] = judge_token_usage_totals.get(key, 0.0) + float(value)

        match_count, matches = maximum_bipartite_matching(edge_map, len(silver_list))
        semantic_match_total += match_count
        for silver_idx, pred_idx in matches.items():
            semantic_match_pairs.append(
                {
                    "anchor_step": anchor,
                    "silver_threat_id": silver_list[silver_idx].get("threat_id"),
                    "predicted_threat_id": pred_list[pred_idx].get("threat_id"),
                }
            )

    primary_anchor_precision = ratio(anchor_match_count, predicted_total)
    primary_anchor_recall = ratio(anchor_match_count, silver_total)
    threat_validity_precision = ratio(semantic_match_total, predicted_total)
    threat_validity_recall = ratio(semantic_match_total, silver_total)
    threat_f1 = harmonic_mean(threat_validity_precision, threat_validity_recall)

    return {
        "use_case_id": use_case_id,
        "dataset": predicted_case.get("dataset"),
        "split": predicted_case.get("split"),
        "source_knowledge_id": predicted_case.get("source_knowledge_id"),
        "predicted_threat_total": predicted_total,
        "silver_threat_total": silver_total,
        "primary_anchor_match_count": anchor_match_count,
        "threat_validity_match_count": semantic_match_total,
        "primary_anchor_precision": primary_anchor_precision,
        "primary_anchor_recall": primary_anchor_recall,
        "threat_validity_precision": threat_validity_precision,
        "threat_validity_recall": threat_validity_recall,
        "threat_f1": threat_f1,
        "judge_pair_count": judge_pair_count,
        "judge_positive_count": judge_positive_count,
        "judge_parse_valid_count": judge_parse_valid_count,
        "judge_schema_valid_count": judge_schema_valid_count,
        "judge_token_usage_totals": judge_token_usage_totals,
        "judge_pairwise_records": pairwise_judgments,
        "semantic_match_pairs": semantic_match_pairs,
    }


def summarize_case_metrics(case_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not case_reports:
        zero_metrics = {
            "primary_anchor_precision": 0.0,
            "primary_anchor_recall": 0.0,
            "threat_validity_precision": 0.0,
            "threat_validity_recall": 0.0,
            "threat_f1": 0.0,
        }
        return {"macro": zero_metrics, "micro": zero_metrics, "counts": {}}

    macro = {
        "primary_anchor_precision": mean(case["primary_anchor_precision"] for case in case_reports),
        "primary_anchor_recall": mean(case["primary_anchor_recall"] for case in case_reports),
        "threat_validity_precision": mean(case["threat_validity_precision"] for case in case_reports),
        "threat_validity_recall": mean(case["threat_validity_recall"] for case in case_reports),
        "threat_f1": mean(case["threat_f1"] for case in case_reports),
    }

    predicted_total = sum(case["predicted_threat_total"] for case in case_reports)
    silver_total = sum(case["silver_threat_total"] for case in case_reports)
    anchor_match_total = sum(case["primary_anchor_match_count"] for case in case_reports)
    semantic_match_total = sum(case["threat_validity_match_count"] for case in case_reports)
    judge_pair_total = sum(case.get("judge_pair_count", 0) for case in case_reports)
    judge_positive_total = sum(case.get("judge_positive_count", 0) for case in case_reports)
    judge_parse_valid_total = sum(case.get("judge_parse_valid_count", 0) for case in case_reports)
    judge_schema_valid_total = sum(case.get("judge_schema_valid_count", 0) for case in case_reports)
    judge_token_usage_totals: Dict[str, float] = {}
    for case in case_reports:
        usage = case.get("judge_token_usage_totals") or {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    judge_token_usage_totals[key] = judge_token_usage_totals.get(key, 0.0) + float(value)

    micro_primary_anchor_precision = ratio(anchor_match_total, predicted_total)
    micro_primary_anchor_recall = ratio(anchor_match_total, silver_total)
    micro_threat_validity_precision = ratio(semantic_match_total, predicted_total)
    micro_threat_validity_recall = ratio(semantic_match_total, silver_total)
    micro = {
        "primary_anchor_precision": micro_primary_anchor_precision,
        "primary_anchor_recall": micro_primary_anchor_recall,
        "threat_validity_precision": micro_threat_validity_precision,
        "threat_validity_recall": micro_threat_validity_recall,
        "threat_f1": harmonic_mean(micro_threat_validity_precision, micro_threat_validity_recall),
    }

    counts = {
        "case_count": len(case_reports),
        "predicted_threat_total": predicted_total,
        "silver_threat_total": silver_total,
        "primary_anchor_match_total": anchor_match_total,
        "threat_validity_match_total": semantic_match_total,
        "judge_pair_total": judge_pair_total,
        "judge_positive_total": judge_positive_total,
        "judge_parse_valid_total": judge_parse_valid_total,
        "judge_schema_valid_total": judge_schema_valid_total,
        "judge_token_usage_totals": judge_token_usage_totals,
    }
    return {"macro": macro, "micro": micro, "counts": counts}


def build_summary_rows(case_reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for case in case_reports:
        rows.append(
            {
                "use_case_id": case["use_case_id"],
                "dataset": case.get("dataset"),
                "split": case.get("split"),
                "source_knowledge_id": case.get("source_knowledge_id"),
                "predicted_threat_total": case["predicted_threat_total"],
                "silver_threat_total": case["silver_threat_total"],
                "primary_anchor_match_count": case["primary_anchor_match_count"],
                "threat_validity_match_count": case["threat_validity_match_count"],
                "primary_anchor_precision": case["primary_anchor_precision"],
                "primary_anchor_recall": case["primary_anchor_recall"],
                "threat_validity_precision": case["threat_validity_precision"],
                "threat_validity_recall": case["threat_validity_recall"],
                "threat_f1": case["threat_f1"],
                "judge_pair_count": case["judge_pair_count"],
                "judge_positive_count": case["judge_positive_count"],
                "judge_parse_valid_count": case.get("judge_parse_valid_count"),
                "judge_schema_valid_count": case.get("judge_schema_valid_count"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    load_dotenv(BASE_DIR / "1_Scripts" / ".env", override=True)

    resolve_paths(args)

    api_key = os.getenv(args.api_key_env)
    base_url = os.getenv(args.base_url_env)
    model_name = resolve_model_name(args)
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

    configure_stream_logging(args.log_path)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Config] predictions_path={args.predictions_path}")
    print(f"[Config] silver_path={args.silver_path}")
    print(f"[Config] flow_path={args.flow_path}")
    print(f"[Config] case_registry_path={args.case_registry_path}")
    print(f"[Config] prompt_path={args.prompt_path}")
    print(f"[Config] output_json={args.output_json}")
    print(f"[Config] output_csv={args.output_csv}")
    print(f"[Config] model={model_name}")

    experiment_id = args.output_json.stem
    llm = build_llm(args, api_key, base_url, model_name)
    if not args.skip_probe:
        run_model_probe(llm, model_name)
    if args.probe_only:
        return

    system_prompt = load_text(args.prompt_path)
    predictions = load_predictions(args.predictions_path)
    silver_cases = load_silver_cases(args.silver_path)
    flow_cases = load_flow_cases(args.flow_path)
    registry = load_registry(args.case_registry_path)

    selected_case_ids = select_case_ids(registry, args.case_id, args.limit)

    existing_payload: Dict[str, Any] = {}
    case_reports: List[Dict[str, Any]] = []
    pairwise_cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    if args.resume and args.output_json.exists():
        existing_payload = load_existing_output(args.output_json)
        case_reports = list(existing_payload.get("case_reports", []))
        for record in existing_payload.get("pairwise_judgments", []):
            key = (
                record["use_case_id"],
                record["anchor_step"],
                str(record["silver_threat_id"]),
                str(record["predicted_threat_id"]),
            )
            pairwise_cache[key] = record
        print(f"[Resume] Loaded {len(case_reports)} case report(s) and {len(pairwise_cache)} cached pairwise judgment(s).")

    completed_case_ids = {report["use_case_id"] for report in case_reports}
    case_report_map = {report["use_case_id"]: report for report in case_reports}

    started_at = now_utc()
    wall_start = time.perf_counter()
    new_case_count = 0
    print(f"[Run] Selected {len(selected_case_ids)} case(s).")

    for position, case_id in enumerate(selected_case_ids, start=1):
        if case_id in completed_case_ids:
            print(f"[Skip] {position}/{len(selected_case_ids)} {case_id} already completed.")
            continue

        predicted_case = predictions.get(case_id)
        silver_case = silver_cases.get(case_id)
        flow_case = flow_cases.get(case_id)
        registry_row = registry.get(case_id)
        if not predicted_case:
            raise ValueError(f"Missing predicted case for {case_id}")
        if not silver_case:
            raise ValueError(f"Missing silver case for {case_id}")
        if not flow_case:
            raise ValueError(f"Missing functional flow case for {case_id}")
        if not registry_row:
            raise ValueError(f"Missing registry row for {case_id}")

        print(f"[Run] {position}/{len(selected_case_ids)} {case_id}")
        case_report = compute_case_metrics(
            predicted_case,
            silver_case,
            flow_case,
            pairwise_cache,
            llm,
            system_prompt,
        )
        case_reports.append(case_report)
        case_report_map[case_id] = case_report
        completed_case_ids.add(case_id)
        new_case_count += 1
        print(
            "[Done] {} anchor_precision={:.4f} anchor_recall={:.4f} validity_precision={:.4f} validity_recall={:.4f} f1={:.4f}".format(
                case_id,
                case_report["primary_anchor_precision"],
                case_report["primary_anchor_recall"],
                case_report["threat_validity_precision"],
                case_report["threat_validity_recall"],
                case_report["threat_f1"],
            )
        )

        summary = summarize_case_metrics(case_reports)
        payload = {
            "meta": {
                "experiment_id": experiment_id,
                "task": "SAAFG Task A evaluation",
                "version": "v0.2",
                "predictions_path": str(args.predictions_path),
                "silver_path": str(args.silver_path),
                "flow_path": str(args.flow_path),
                "case_registry_path": str(args.case_registry_path),
                "prompt_path": str(args.prompt_path),
                "model_name": model_name,
                "started_at_utc": existing_payload.get("meta", {}).get("started_at_utc") or started_at,
                "updated_at_utc": now_utc(),
                "notes": [
                    "primary_anchor_* is computed by exact anchor-step matching only.",
                    "threat_validity_* is computed by exact-anchor pairwise proxy judge plus one-to-one matching.",
                    "Macro metrics are case averages; micro metrics are global counts.",
                    "Case-level precision is 0.0 when no predicted threat is available.",
                ],
            },
            "case_reports": case_reports,
            "pairwise_judgments": list(pairwise_cache.values()),
            "summary": summary,
        }
        write_json_atomic(args.output_json, payload)
        write_csv(args.output_csv, build_summary_rows(case_reports), [
            "use_case_id",
            "dataset",
            "split",
            "source_knowledge_id",
            "predicted_threat_total",
            "silver_threat_total",
            "primary_anchor_match_count",
            "threat_validity_match_count",
            "primary_anchor_precision",
            "primary_anchor_recall",
            "threat_validity_precision",
            "threat_validity_recall",
            "threat_f1",
            "judge_pair_count",
            "judge_positive_count",
        ])

    summary = summarize_case_metrics(case_reports)
    previous_wall_time = existing_payload.get("meta", {}).get("wall_time_seconds")
    wall_time_seconds = (
        previous_wall_time
        if new_case_count == 0 and previous_wall_time is not None
        else time.perf_counter() - wall_start
    )
    payload = {
        "meta": {
            "experiment_id": experiment_id,
            "task": "SAAFG Task A evaluation",
            "version": "v0.2",
            "predictions_path": str(args.predictions_path),
            "silver_path": str(args.silver_path),
            "flow_path": str(args.flow_path),
            "case_registry_path": str(args.case_registry_path),
            "prompt_path": str(args.prompt_path),
            "model_name": model_name,
            "started_at_utc": existing_payload.get("meta", {}).get("started_at_utc") or started_at,
            "updated_at_utc": now_utc(),
            "wall_time_seconds": wall_time_seconds,
            "notes": [
                "primary_anchor_* is computed by exact anchor-step matching only.",
                "threat_validity_* is computed by exact-anchor pairwise proxy judge plus one-to-one matching.",
                "Macro metrics are case averages; micro metrics are global counts.",
                "Case-level precision is 0.0 when no predicted threat is available.",
            ],
        },
        "case_reports": case_reports,
        "pairwise_judgments": list(pairwise_cache.values()),
        "summary": summary,
    }
    write_json_atomic(args.output_json, payload)
    write_csv(args.output_csv, build_summary_rows(case_reports), [
        "use_case_id",
        "dataset",
        "split",
        "source_knowledge_id",
        "predicted_threat_total",
        "silver_threat_total",
        "primary_anchor_match_count",
        "threat_validity_match_count",
        "primary_anchor_precision",
        "primary_anchor_recall",
        "threat_validity_precision",
        "threat_validity_recall",
        "threat_f1",
        "judge_pair_count",
        "judge_positive_count",
    ])

    print("[Summary] cases={}".format(summary["counts"]["case_count"]))
    print("[Summary] macro={}".format(json.dumps(summary["macro"], ensure_ascii=False)))
    print("[Summary] micro={}".format(json.dumps(summary["micro"], ensure_ascii=False)))
    print("[Summary] counts={}".format(json.dumps(summary["counts"], ensure_ascii=False)))
    print(f"[Summary] output_json={args.output_json}")
    print(f"[Summary] output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
