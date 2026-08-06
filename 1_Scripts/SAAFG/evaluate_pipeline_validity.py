#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Task B evaluation for SAAFG v0.2 NoRAG Blue Team outputs.

The evaluator uses Task A semantic matches to map each predicted Red Team threat
    to one silver threat, then judges whether the generated Blue Team defense covers
    the corresponding silver defense.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"

DEFAULT_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "defense_validity_judge.txt"
DEFAULT_FLOW_PATH = SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
DEFAULT_SILVER_THREAT_PATH = SAAFG_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"
DEFAULT_SILVER_BLUE_PATH = SAAFG_ROOT / "3_BlueTeam_SA_Flows" / "security_augmented_use_case_flows.json"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
DEFAULT_RUN_TAG = "deepseek-v32" #"qwen35plus"

ALLOWED_JUDGE_REASON_CODES = {
    "same_defense",
    "weak_basic_flow",
    "weak_alternative_flow",
    "missing_detection",
    "missing_blocking",
    "wrong_threat",
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


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b) if (a + b) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAAFG v0.2 NoRAG Blue Team Task B.")
    parser.add_argument("--run-tag", default=None, help="Blue Team run tag, e.g. qwen35plus or deepseek-v32.")
    parser.add_argument("--source-run-tag", default=None, help="Red Team source tag. Defaults to --run-tag.")
    parser.add_argument("--blue-run-path", type=Path, default=None)
    parser.add_argument("--blue-artifact-path", type=Path, default=None)
    parser.add_argument("--red-run-path", type=Path, default=None)
    parser.add_argument("--red-eval-path", type=Path, default=None)
    parser.add_argument("--flow-path", type=Path, default=DEFAULT_FLOW_PATH)
    parser.add_argument("--silver-threat-path", type=Path, default=DEFAULT_SILVER_THREAT_PATH)
    parser.add_argument("--silver-blue-path", type=Path, default=DEFAULT_SILVER_BLUE_PATH)
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
    parser.add_argument("--model-name", default=None, help="Override judge model name directly.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Do not send Qwen thinking-control extra_body; useful for non-Qwen endpoints.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    run_tag = args.run_tag or DEFAULT_RUN_TAG
    source_tag = args.source_run_tag or run_tag
    blue_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_NoRAG_{run_tag}" / "blue_team"
    red_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_NoRAG_{source_tag}" / "red_team"

    if args.blue_run_path is None:
        args.blue_run_path = blue_dir / f"saafg_blueteam_norag_v0_2_{run_tag}.json"
    if args.blue_artifact_path is None:
        args.blue_artifact_path = blue_dir / f"saafg_security_augmented_flows_pred_norag_v0_2_{run_tag}.json"
    if args.red_run_path is None:
        args.red_run_path = red_dir / f"saafg_redteam_norag_v0_2_{source_tag}.json"
    if args.red_eval_path is None:
        args.red_eval_path = red_dir / f"saafg_redteam_task_a_eval_norag_v0_2_{source_tag}.json"
    if args.output_json is None:
        args.output_json = blue_dir / f"saafg_blueteam_task_b_eval_norag_v0_2_{run_tag}.json"
    if args.output_csv is None:
        args.output_csv = blue_dir / f"saafg_blueteam_task_b_eval_norag_v0_2_{run_tag}.csv"
    if args.log_path is None:
        args.log_path = blue_dir / f"saafg_blueteam_task_b_eval_norag_v0_2_{run_tag}.log"
    return run_tag, source_tag


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


_OUTER_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def safe_json_loads(content: str) -> Tuple[Any, bool]:
    text = content.strip()
    repaired = False
    match = _OUTER_CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
        repaired = True
    return json.loads(text), repaired


def get_response_usage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return {"raw_usage": str(usage)}
    return usage


def build_lookup(payload: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    items = payload.get(key)
    if not isinstance(items, list):
        raise ValueError(f"Expected {key} list in payload.")
    return {item["use_case_id"]: item for item in items}


def load_blue_run(path: Path) -> Dict[str, Dict[str, Any]]:
    return build_lookup(read_json(path), "results")


def load_red_run(path: Path) -> Dict[str, Dict[str, Any]]:
    return build_lookup(read_json(path), "results")


def load_red_eval(path: Path) -> Dict[str, Dict[str, Any]]:
    return build_lookup(read_json(path), "case_reports")


def load_flow_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    return build_lookup(read_json(path), "use_case_flows")


def load_silver_threat_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    return build_lookup(read_json(path), "threat_record_cases")


def load_sa_flow_cases(path: Path) -> Dict[str, Dict[str, Any]]:
    return build_lookup(read_json(path), "security_augmented_flow_cases")


def select_case_ids(flow_cases: Dict[str, Dict[str, Any]], case_ids: Optional[List[str]], limit: Optional[int]) -> List[str]:
    ordered = sorted(flow_cases.keys(), key=lambda cid: int(re.sub(r"\D", "", cid) or 0))
    if case_ids:
        wanted = set(case_ids)
        missing = sorted(wanted - set(flow_cases))
        if missing:
            raise ValueError(f"Unknown case_id(s): {', '.join(missing)}")
        ordered = [case_id for case_id in ordered if case_id in wanted]
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def build_flow_context(flow_case: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "step_id": str(step.get("step_id") or "").strip(),
            "step_sentence": str(step.get("step_sentence") or "").strip(),
        }
        for step in (flow_case.get("basic_flow") or [])
        if step.get("step_id") and step.get("step_sentence")
    ]


def find_defense(sa_case: Dict[str, Any], threat_id: str) -> Optional[Dict[str, Any]]:
    flow = sa_case.get("security_augmented_flow") or {}
    basic_flows = flow.get("security_basic_flow") or []
    alternative_flows = flow.get("security_alternative_flows") or []
    for index, alternative_flow in enumerate(alternative_flows):
        mitigates = [str(item).strip() for item in (alternative_flow.get("mitigates") or [])]
        if threat_id in mitigates:
            basic_flow = basic_flows[index] if index < len(basic_flows) else None
            return {
                "security_basic_flow": compact_basic_flow(basic_flow),
                "security_alternative_flow": compact_alternative_flow(alternative_flow),
            }
    return None


def compact_basic_flow(basic_flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not basic_flow:
        return {}
    return {
        "anchor_after": basic_flow.get("anchor_after"),
        "step_sentence": basic_flow.get("step_sentence"),
    }


def compact_alternative_flow(alternative_flow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not alternative_flow:
        return {}
    return {
        "entry_condition": alternative_flow.get("entry_condition"),
        "steps": [
            {
                "step_sentence": step.get("step_sentence"),
            }
            for step in (alternative_flow.get("steps") or [])
        ],
    }


def matched_pairs_from_case(case_report: Dict[str, Any]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for pair in case_report.get("semantic_match_pairs") or []:
        silver_threat_id = str(pair.get("silver_threat_id") or "").strip()
        predicted_threat_id = str(pair.get("predicted_threat_id") or "").strip()
        anchor_step = str(pair.get("anchor_step") or "").strip()
        if silver_threat_id and predicted_threat_id:
            pairs.append(
                {
                    "anchor_step": anchor_step,
                    "silver_threat_id": silver_threat_id,
                    "predicted_threat_id": predicted_threat_id,
                }
            )
    return pairs


def validate_judge_output(parsed: Any) -> Tuple[bool, List[str], bool, bool, bool, str, str]:
    errors: List[str] = []
    if not isinstance(parsed, dict):
        return False, ["Judge output root is not a JSON object."], False, False, False, "unclear", ""

    expected_fields = {
        "security_basic_flow_valid",
        "security_alternative_flow_valid",
        "overall_defense_valid",
        "reason_code",
        "reasoning",
    }
    extra_fields = sorted(set(parsed.keys()) - expected_fields)
    if extra_fields:
        errors.append(f"Unexpected field(s): {', '.join(extra_fields)}.")
    missing_fields = sorted(expected_fields - set(parsed.keys()))
    if missing_fields:
        errors.append(f"Missing field(s): {', '.join(missing_fields)}.")

    basic_valid = parsed.get("security_basic_flow_valid")
    alternative_valid = parsed.get("security_alternative_flow_valid")
    overall_valid = parsed.get("overall_defense_valid")
    if not isinstance(basic_valid, bool):
        errors.append("security_basic_flow_valid must be a boolean.")
        basic_valid = False
    if not isinstance(alternative_valid, bool):
        errors.append("security_alternative_flow_valid must be a boolean.")
        alternative_valid = False
    if not isinstance(overall_valid, bool):
        errors.append("overall_defense_valid must be a boolean.")
        overall_valid = False

    reason_code = parsed.get("reason_code")
    if not isinstance(reason_code, str) or reason_code not in ALLOWED_JUDGE_REASON_CODES:
        errors.append("reason_code is missing or invalid.")
        reason_code = "unclear"
    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str):
        errors.append("reasoning must be a string.")
        reasoning = ""

    return not errors, errors, basic_valid, alternative_valid, overall_valid, reason_code, reasoning


def build_judge_input(
    use_case_id: str,
    flow_context: List[Dict[str, str]],
    match_pair: Dict[str, str],
    silver_defense: Dict[str, Any],
    predicted_defense: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "use_case_id": use_case_id,
        "functional_flow_context": flow_context,
        "mapped_pair": match_pair,
        "silver_defense": silver_defense,
        "predicted_defense": predicted_defense,
    }


def judge_defense(
    llm: ChatOpenAI,
    system_prompt: str,
    judge_input: Dict[str, Any],
) -> Dict[str, Any]:
    started_at = now_utc()
    start_time = time.perf_counter()
    raw_output = ""
    parsed: Any = None
    parse_repaired = False
    parse_valid = False
    schema_valid = False
    schema_errors: List[str] = []
    parse_error = None
    response_metadata: Dict[str, Any] = {}
    basic_valid = False
    alternative_valid = False
    overall_valid = False
    reason_code = "unclear"
    reasoning = ""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(judge_input, ensure_ascii=False, separators=(",", ":"))),
            ]
        )
        raw_output = str(getattr(response, "content", "") or "")
        response_metadata = getattr(response, "response_metadata", {}) or {}
        try:
            parsed, parse_repaired = safe_json_loads(raw_output)
            parse_valid = True
            (
                schema_valid,
                schema_errors,
                basic_valid,
                alternative_valid,
                overall_valid,
                reason_code,
                reasoning,
            ) = validate_judge_output(parsed)
        except Exception as exc:
            parse_error = str(exc)
    except Exception as exc:
        parse_error = f"Model invocation failed: {exc}"

    duration = time.perf_counter() - start_time
    if not parse_valid:
        reasoning = parse_error or "No valid judge output."
    elif not schema_valid:
        overall_valid = False
        if not reasoning:
            reasoning = "Invalid judge schema."

    return {
        "security_basic_flow_valid": basic_valid,
        "security_alternative_flow_valid": alternative_valid,
        "overall_defense_valid": overall_valid,
        "reason_code": reason_code,
        "reasoning": reasoning,
        "raw_output": raw_output,
        "parsed_output": parsed,
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
        "token_usage": get_response_usage(response_metadata),
    }


def automatic_missing_record(
    reason_code: str,
    reasoning: str,
) -> Dict[str, Any]:
    return {
        "security_basic_flow_valid": False,
        "security_alternative_flow_valid": False,
        "overall_defense_valid": False,
        "reason_code": reason_code,
        "reasoning": reasoning,
        "raw_output": "",
        "parsed_output": None,
        "parse_valid": False,
        "parse_repaired": False,
        "schema_valid": False,
        "schema_errors": [],
        "parse_error": None,
        "judge_duration_seconds": 0.0,
        "started_at_utc": now_utc(),
        "finished_at_utc": now_utc(),
        "response_model": None,
        "finish_reason": None,
        "token_usage": {},
        "automatic_failure": True,
    }


def case_metric_report(
    case_id: str,
    flow_case: Dict[str, Any],
    silver_threat_case: Dict[str, Any],
    red_run_case: Dict[str, Any],
    red_eval_case: Dict[str, Any],
    silver_sa_case: Dict[str, Any],
    predicted_sa_case: Dict[str, Any],
    blue_run_case: Dict[str, Any],
    judge_cache: Dict[Tuple[str, str, str], Dict[str, Any]],
    llm: ChatOpenAI,
    system_prompt: str,
) -> Dict[str, Any]:
    flow_context = build_flow_context(flow_case)
    match_pairs = matched_pairs_from_case(red_eval_case)
    silver_total = int(red_eval_case.get("silver_threat_total") or len(silver_threat_case.get("threat_records") or []))
    raw_predicted_total = int(red_eval_case.get("predicted_threat_total") or len(red_run_case.get("threat_records") or []))
    generated_defense_total = 0
    missing_predicted_defense_count = 0
    missing_silver_defense_count = 0
    defense_judgments: List[Dict[str, Any]] = []
    token_usage_totals: Dict[str, float] = {}

    for match_pair in match_pairs:
        silver_threat_id = match_pair["silver_threat_id"]
        predicted_threat_id = match_pair["predicted_threat_id"]
        silver_defense = find_defense(silver_sa_case, silver_threat_id)
        predicted_defense = find_defense(predicted_sa_case, predicted_threat_id)

        cache_key = (case_id, silver_threat_id, predicted_threat_id)
        base_record = {
            "use_case_id": case_id,
            "anchor_step": match_pair.get("anchor_step"),
            "silver_threat_id": silver_threat_id,
            "predicted_threat_id": predicted_threat_id,
            "judge_input": None,
        }

        if predicted_defense:
            generated_defense_total += 1
        else:
            missing_predicted_defense_count += 1
            record = automatic_missing_record("missing_predicted_defense", "Predicted defense not found.")
            defense_judgments.append({**base_record, **record})
            continue

        if not silver_defense:
            missing_silver_defense_count += 1
            record = automatic_missing_record("missing_silver_defense", "Silver defense not found.")
            defense_judgments.append({**base_record, **record})
            continue

        judge_input = build_judge_input(
            case_id,
            flow_context,
            match_pair,
            silver_defense,
            predicted_defense,
        )
        cached_record = judge_cache.get(cache_key)
        if cached_record is None:
            cached_record = judge_defense(llm, system_prompt, judge_input)
            judge_cache[cache_key] = cached_record
        full_record = {**base_record, "judge_input": judge_input, **cached_record}
        defense_judgments.append(full_record)
        usage = cached_record.get("token_usage") or {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    token_usage_totals[key] = token_usage_totals.get(key, 0.0) + float(value)

    task_a_valid_total = len(match_pairs)
    basic_valid_count = sum(1 for item in defense_judgments if item.get("security_basic_flow_valid"))
    alternative_valid_count = sum(1 for item in defense_judgments if item.get("security_alternative_flow_valid"))
    defense_valid_count = sum(1 for item in defense_judgments if item.get("overall_defense_valid"))
    judge_call_count = sum(1 for item in defense_judgments if not item.get("automatic_failure"))
    judge_parse_valid_count = sum(1 for item in defense_judgments if item.get("parse_valid"))
    judge_schema_valid_count = sum(1 for item in defense_judgments if item.get("schema_valid"))
    judge_repaired_count = sum(1 for item in defense_judgments if item.get("parse_repaired"))

    defense_validity_precision = ratio(defense_valid_count, generated_defense_total)
    defense_validity_recall = ratio(defense_valid_count, task_a_valid_total)
    defense_f1 = harmonic_mean(defense_validity_precision, defense_validity_recall)
    end_to_end_precision = ratio(defense_valid_count, raw_predicted_total)
    end_to_end_recall = ratio(defense_valid_count, silver_total)
    end_to_end_f1 = harmonic_mean(end_to_end_precision, end_to_end_recall)

    return {
        "use_case_id": case_id,
        "dataset": red_eval_case.get("dataset") or predicted_sa_case.get("dataset"),
        "split": red_eval_case.get("split") or predicted_sa_case.get("split"),
        "source_knowledge_id": red_eval_case.get("source_knowledge_id") or predicted_sa_case.get("source_knowledge_id"),
        "raw_predicted_threat_total": raw_predicted_total,
        "silver_threat_total": silver_total,
        "task_a_valid_threat_total": task_a_valid_total,
        "generated_defense_total": generated_defense_total,
        "missing_predicted_defense_count": missing_predicted_defense_count,
        "missing_silver_defense_count": missing_silver_defense_count,
        "security_basic_flow_valid_count": basic_valid_count,
        "security_alternative_flow_valid_count": alternative_valid_count,
        "defense_valid_count": defense_valid_count,
        "defense_validity_precision": defense_validity_precision,
        "defense_validity_recall": defense_validity_recall,
        "defense_f1": defense_f1,
        "end_to_end_pipeline_precision": end_to_end_precision,
        "end_to_end_defense_recall": end_to_end_recall,
        "end_to_end_defense_f1": end_to_end_f1,
        "case_has_valid_defense": defense_valid_count > 0,
        "case_all_silver_threats_defended": silver_total > 0 and defense_valid_count >= silver_total,
        "judge_call_count": judge_call_count,
        "judge_parse_valid_count": judge_parse_valid_count,
        "judge_schema_valid_count": judge_schema_valid_count,
        "judge_repaired_count": judge_repaired_count,
        "judge_token_usage_totals": token_usage_totals,
        "source_blue_case_schema_valid": blue_run_case.get("case_schema_valid"),
        "defense_judgments": defense_judgments,
    }


def summarize_case_metrics(case_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    zero = {
        "defense_validity_precision": 0.0,
        "defense_validity_recall": 0.0,
        "defense_f1": 0.0,
        "end_to_end_pipeline_precision": 0.0,
        "end_to_end_defense_recall": 0.0,
        "end_to_end_defense_f1": 0.0,
        "end_to_end_case_recall": 0.0,
        "end_to_end_full_case_recall": 0.0,
    }
    if not case_reports:
        return {"macro": zero, "micro": zero, "counts": {}}

    macro = {
        "defense_validity_precision": mean(case["defense_validity_precision"] for case in case_reports),
        "defense_validity_recall": mean(case["defense_validity_recall"] for case in case_reports),
        "defense_f1": mean(case["defense_f1"] for case in case_reports),
        "end_to_end_pipeline_precision": mean(case["end_to_end_pipeline_precision"] for case in case_reports),
        "end_to_end_defense_recall": mean(case["end_to_end_defense_recall"] for case in case_reports),
        "end_to_end_defense_f1": mean(case["end_to_end_defense_f1"] for case in case_reports),
        "end_to_end_case_recall": mean(1.0 if case["case_has_valid_defense"] else 0.0 for case in case_reports),
        "end_to_end_full_case_recall": mean(
            1.0 if case["case_all_silver_threats_defended"] else 0.0 for case in case_reports
        ),
    }

    raw_predicted_total = sum(case["raw_predicted_threat_total"] for case in case_reports)
    silver_total = sum(case["silver_threat_total"] for case in case_reports)
    task_a_valid_total = sum(case["task_a_valid_threat_total"] for case in case_reports)
    generated_defense_total = sum(case["generated_defense_total"] for case in case_reports)
    defense_valid_total = sum(case["defense_valid_count"] for case in case_reports)
    basic_valid_total = sum(case["security_basic_flow_valid_count"] for case in case_reports)
    alternative_valid_total = sum(case["security_alternative_flow_valid_count"] for case in case_reports)
    missing_predicted_defense_total = sum(case["missing_predicted_defense_count"] for case in case_reports)
    missing_silver_defense_total = sum(case["missing_silver_defense_count"] for case in case_reports)
    judge_call_total = sum(case["judge_call_count"] for case in case_reports)
    judge_parse_valid_total = sum(case["judge_parse_valid_count"] for case in case_reports)
    judge_schema_valid_total = sum(case["judge_schema_valid_count"] for case in case_reports)
    judge_repaired_total = sum(case["judge_repaired_count"] for case in case_reports)
    case_hit_total = sum(1 for case in case_reports if case["case_has_valid_defense"])
    case_full_total = sum(1 for case in case_reports if case["case_all_silver_threats_defended"])

    micro_defense_precision = ratio(defense_valid_total, generated_defense_total)
    micro_defense_recall = ratio(defense_valid_total, task_a_valid_total)
    micro_end_to_end_precision = ratio(defense_valid_total, raw_predicted_total)
    micro_end_to_end_recall = ratio(defense_valid_total, silver_total)
    micro = {
        "defense_validity_precision": micro_defense_precision,
        "defense_validity_recall": micro_defense_recall,
        "defense_f1": harmonic_mean(micro_defense_precision, micro_defense_recall),
        "security_basic_flow_valid_rate": ratio(basic_valid_total, generated_defense_total),
        "security_alternative_flow_valid_rate": ratio(alternative_valid_total, generated_defense_total),
        "end_to_end_pipeline_precision": micro_end_to_end_precision,
        "end_to_end_defense_recall": micro_end_to_end_recall,
        "end_to_end_defense_f1": harmonic_mean(micro_end_to_end_precision, micro_end_to_end_recall),
        "end_to_end_case_recall": ratio(case_hit_total, len(case_reports)),
        "end_to_end_full_case_recall": ratio(case_full_total, len(case_reports)),
    }

    judge_token_usage_totals: Dict[str, float] = {}
    for case in case_reports:
        usage = case.get("judge_token_usage_totals") or {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    judge_token_usage_totals[key] = judge_token_usage_totals.get(key, 0.0) + float(value)

    counts = {
        "case_count": len(case_reports),
        "raw_predicted_threat_total": raw_predicted_total,
        "silver_threat_total": silver_total,
        "task_a_valid_threat_total": task_a_valid_total,
        "generated_defense_total": generated_defense_total,
        "defense_valid_total": defense_valid_total,
        "security_basic_flow_valid_total": basic_valid_total,
        "security_alternative_flow_valid_total": alternative_valid_total,
        "missing_predicted_defense_total": missing_predicted_defense_total,
        "missing_silver_defense_total": missing_silver_defense_total,
        "case_hit_total": case_hit_total,
        "case_full_defense_total": case_full_total,
        "judge_call_total": judge_call_total,
        "judge_parse_valid_total": judge_parse_valid_total,
        "judge_schema_valid_total": judge_schema_valid_total,
        "judge_repaired_total": judge_repaired_total,
        "judge_token_usage_totals": judge_token_usage_totals,
    }
    return {"macro": macro, "micro": micro, "counts": counts}


def build_summary_rows(case_reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in case_reports:
        rows.append(
            {
                "use_case_id": case["use_case_id"],
                "dataset": case.get("dataset"),
                "split": case.get("split"),
                "source_knowledge_id": case.get("source_knowledge_id"),
                "raw_predicted_threat_total": case["raw_predicted_threat_total"],
                "silver_threat_total": case["silver_threat_total"],
                "task_a_valid_threat_total": case["task_a_valid_threat_total"],
                "generated_defense_total": case["generated_defense_total"],
                "defense_valid_count": case["defense_valid_count"],
                "security_basic_flow_valid_count": case["security_basic_flow_valid_count"],
                "security_alternative_flow_valid_count": case["security_alternative_flow_valid_count"],
                "defense_validity_precision": case["defense_validity_precision"],
                "defense_validity_recall": case["defense_validity_recall"],
                "defense_f1": case["defense_f1"],
                "end_to_end_pipeline_precision": case["end_to_end_pipeline_precision"],
                "end_to_end_defense_recall": case["end_to_end_defense_recall"],
                "end_to_end_defense_f1": case["end_to_end_defense_f1"],
                "case_has_valid_defense": case["case_has_valid_defense"],
                "case_all_silver_threats_defended": case["case_all_silver_threats_defended"],
                "judge_call_count": case["judge_call_count"],
                "judge_parse_valid_count": case["judge_parse_valid_count"],
                "judge_schema_valid_count": case["judge_schema_valid_count"],
            }
        )
    return rows


CSV_FIELDS = [
    "use_case_id",
    "dataset",
    "split",
    "source_knowledge_id",
    "raw_predicted_threat_total",
    "silver_threat_total",
    "task_a_valid_threat_total",
    "generated_defense_total",
    "defense_valid_count",
    "security_basic_flow_valid_count",
    "security_alternative_flow_valid_count",
    "defense_validity_precision",
    "defense_validity_recall",
    "defense_f1",
    "end_to_end_pipeline_precision",
    "end_to_end_defense_recall",
    "end_to_end_defense_f1",
    "case_has_valid_defense",
    "case_all_silver_threats_defended",
    "judge_call_count",
    "judge_parse_valid_count",
    "judge_schema_valid_count",
]


def load_existing_output(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid output payload in {path}")
    payload.setdefault("case_reports", [])
    payload.setdefault("defense_judgments", [])
    payload.setdefault("summary", {})
    return payload


def write_outputs(
    args: argparse.Namespace,
    meta: Dict[str, Any],
    case_reports: Sequence[Dict[str, Any]],
) -> None:
    summary = summarize_case_metrics(case_reports)
    all_judgments = [
        judgment
        for case in case_reports
        for judgment in (case.get("defense_judgments") or [])
    ]
    payload = {
        "meta": {**meta, "updated_at_utc": now_utc()},
        "case_reports": list(case_reports),
        "defense_judgments": all_judgments,
        "summary": summary,
    }
    write_json_atomic(args.output_json, payload)
    write_csv(args.output_csv, build_summary_rows(case_reports), CSV_FIELDS)


def main() -> None:
    args = parse_args()
    load_dotenv(BASE_DIR / "1_Scripts" / ".env", override=True)
    run_tag, source_tag = resolve_paths(args)

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

    configure_stream_logging(args.log_path)
    print(f"[Config] run_tag={run_tag}")
    print(f"[Config] source_run_tag={source_tag}")
    print(f"[Config] blue_run_path={args.blue_run_path}")
    print(f"[Config] blue_artifact_path={args.blue_artifact_path}")
    print(f"[Config] red_run_path={args.red_run_path}")
    print(f"[Config] red_eval_path={args.red_eval_path}")
    print(f"[Config] silver_threat_path={args.silver_threat_path}")
    print(f"[Config] silver_blue_path={args.silver_blue_path}")
    print(f"[Config] prompt_path={args.prompt_path}")
    print(f"[Config] output_json={args.output_json}")
    print(f"[Config] output_csv={args.output_csv}")
    print(f"[Config] judge_model={model_name}")

    llm = build_llm(args, api_key, base_url, model_name)
    if not args.skip_probe:
        run_model_probe(llm, model_name)
    if args.probe_only:
        return

    system_prompt = load_text(args.prompt_path)
    flow_cases = load_flow_cases(args.flow_path)
    silver_threat_cases = load_silver_threat_cases(args.silver_threat_path)
    silver_sa_cases = load_sa_flow_cases(args.silver_blue_path)
    blue_run_cases = load_blue_run(args.blue_run_path)
    predicted_sa_cases = load_sa_flow_cases(args.blue_artifact_path)
    red_run_cases = load_red_run(args.red_run_path)
    red_eval_cases = load_red_eval(args.red_eval_path)

    selected_case_ids = select_case_ids(flow_cases, args.case_id, args.limit)

    existing_payload: Dict[str, Any] = {}
    case_reports: List[Dict[str, Any]] = []
    judge_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if args.resume and args.output_json.exists():
        existing_payload = load_existing_output(args.output_json)
        case_reports = list(existing_payload.get("case_reports") or [])
        for record in existing_payload.get("defense_judgments") or []:
            if record.get("automatic_failure"):
                continue
            key = (
                record["use_case_id"],
                str(record["silver_threat_id"]),
                str(record["predicted_threat_id"]),
            )
            judge_cache[key] = {
                name: value
                for name, value in record.items()
                if name
                not in {
                    "use_case_id",
                    "anchor_step",
                    "silver_threat_id",
                    "predicted_threat_id",
                    "judge_input",
                }
            }
        print(f"[Resume] Loaded {len(case_reports)} case report(s) and {len(judge_cache)} cached judgment(s).")

    completed_case_ids = {case["use_case_id"] for case in case_reports}
    started_at = existing_payload.get("meta", {}).get("started_at_utc") or now_utc()
    wall_start = time.perf_counter()
    experiment_id = args.output_json.stem
    meta = {
        "experiment_id": experiment_id,
        "task": "SAAFG Task B Blue Team evaluation",
        "version": "v0.2",
        "run_tag": run_tag,
        "source_run_tag": source_tag,
        "blue_run_path": str(args.blue_run_path),
        "blue_artifact_path": str(args.blue_artifact_path),
        "red_run_path": str(args.red_run_path),
        "red_eval_path": str(args.red_eval_path),
        "flow_path": str(args.flow_path),
        "silver_threat_path": str(args.silver_threat_path),
        "silver_blue_path": str(args.silver_blue_path),
        "prompt_path": str(args.prompt_path),
        "judge_model_name": model_name,
        "started_at_utc": started_at,
        "notes": [
            "Task B evaluates only Task A semantic_match_pairs.",
            "Blue defense validity is judged against mapped silver threat and silver security-augmented flow.",
            "end_to_end_* metrics use raw Red Team predictions and all silver threats as denominators.",
            "Missing predicted defenses are automatic failures without LLM judge calls.",
        ],
    }

    total = len(selected_case_ids)
    print(f"[Run] Selected {total} case(s).")
    for position, case_id in enumerate(selected_case_ids, start=1):
        if case_id in completed_case_ids:
            print(f"[Skip] {position}/{total} {case_id} already completed.")
            continue
        required = {
            "flow": flow_cases,
            "silver_threat": silver_threat_cases,
            "silver_sa": silver_sa_cases,
            "blue_run": blue_run_cases,
            "predicted_sa": predicted_sa_cases,
            "red_run": red_run_cases,
            "red_eval": red_eval_cases,
        }
        missing_cases = [name for name, lookup in required.items() if case_id not in lookup]
        if missing_cases:
            raise ValueError(f"Missing {', '.join(missing_cases)} for {case_id}")

        print(f"[Run] {position}/{total} {case_id}")
        case_report = case_metric_report(
            case_id,
            flow_cases[case_id],
            silver_threat_cases[case_id],
            red_run_cases[case_id],
            red_eval_cases[case_id],
            silver_sa_cases[case_id],
            predicted_sa_cases[case_id],
            blue_run_cases[case_id],
            judge_cache,
            llm,
            system_prompt,
        )
        case_reports.append(case_report)
        completed_case_ids.add(case_id)
        print(
            "[Done] {} task_a_valid={} generated={} defense_valid={} e2e_recall={:.4f}".format(
                case_id,
                case_report["task_a_valid_threat_total"],
                case_report["generated_defense_total"],
                case_report["defense_valid_count"],
                case_report["end_to_end_defense_recall"],
            )
        )
        write_outputs(args, meta, case_reports)

    meta["wall_time_seconds"] = time.perf_counter() - wall_start
    write_outputs(args, meta, case_reports)
    summary = summarize_case_metrics(case_reports)
    print("[Summary] cases={}".format(summary["counts"].get("case_count", 0)))
    print("[Summary] macro={}".format(json.dumps(summary["macro"], ensure_ascii=False)))
    print("[Summary] micro={}".format(json.dumps(summary["micro"], ensure_ascii=False)))
    print("[Summary] counts={}".format(json.dumps(summary["counts"], ensure_ascii=False)))
    print(f"[Summary] output_json={args.output_json}")
    print(f"[Summary] output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
