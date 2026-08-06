#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer

from evographrag_retrieval import RecEvoGraphRetriever


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"

DEFAULT_INPUT_PATH = (
    SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
)
DEFAULT_REGISTRY_PATH = (
    SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
)
DEFAULT_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "red_team_static_graph_rag.txt"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
DEFAULT_KG_DIR = KB_DIR / "recevograph_rag"
CHROMA_EXPORT_DIR = KB_DIR / "chroma_db" / "store"
CHROMA_RUNTIME_DIR = (
    Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / "AI_for_AI_Sec" / "chroma_db_runtime"
)
CHROMA_COLLECTION_NAME = "ai_sec_knowledge_base"
EMBEDDING_BACKEND_FILE = "embedding_backend.json"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
RAG_RUNTIME_READY = False


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
    # Windows antivirus/indexers can briefly lock the destination and make
    # os.replace fail repeatedly. Direct write keeps long API batches resumable.
    path.write_text(text, encoding="utf-8")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass


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
    parser = argparse.ArgumentParser(
        description="Run SAAFG v0.2 StaticGraphRAG-GraphAware Red Team generation."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--case-registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--run-tag", default=None, help="Optional output tag override, e.g. qwen35plus or deepseek-v32.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--retrieval-trace-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--rag-top-k", type=int, default=3, help="Number of StaticGraphRAG ranked items to retrieve.")
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N selected cases.")
    parser.add_argument("--case-id", nargs="*", default=None, help="Optional explicit use_case_id list.")
    parser.add_argument("--resume", action="store_true", help="Skip use_case_ids already present in output.")
    parser.add_argument("--probe-only", action="store_true", help="Only probe the configured model and exit.")
    parser.add_argument("--skip-probe", action="store_true", help="Skip startup model probe.")
    parser.add_argument("--api-key-env", default="API_KEY")
    parser.add_argument("--base-url-env", default="BASE_URL")
    parser.add_argument("--model-env-var", default="MODEL_DEEPSEEK_V32")
    parser.add_argument("--model-name", default=None, help="Override model name directly.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--no-extra-body",
        action="store_true",
        help="Do not send Qwen thinking-control extra_body; useful for non-Qwen endpoints.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, model_name: str) -> None:
    tag = args.run_tag or (
        "deepseek-v32"
        if model_name == os.getenv("MODEL_DEEPSEEK_V32")
        else ("qwen35plus" if model_name == os.getenv("MODEL_QWEN35_PLUS") else model_slug(model_name))
    )
    if args.result_dir is None:
        args.result_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_StaticGraphRAG_GraphAware_{tag}" / "red_team"
    if args.output_path is None:
        args.output_path = args.result_dir / f"saafg_redteam_staticgraphrag_graphaware_v0_2_{tag}.json"
    if args.artifact_path is None:
        args.artifact_path = args.result_dir / f"saafg_threat_records_pred_staticgraphrag_graphaware_v0_2_{tag}.json"
    if args.retrieval_trace_path is None:
        args.retrieval_trace_path = (
            args.result_dir / f"saafg_redteam_retrieval_trace_staticgraphrag_graphaware_v0_2_{tag}.jsonl"
        )
    if args.log_path is None:
        args.log_path = args.result_dir / f"saafg_redteam_staticgraphrag_graphaware_v0_2_{tag}.log"


def configure_stream_logging(log_path: Path) -> None:
    sys.stdout = TeeStream(sys.stdout, log_path)
    sys.stderr = TeeStream(sys.stderr, log_path)


def build_llm(args: argparse.Namespace, api_key: str, base_url: str, model_name: str) -> Any:
    from langchain_openai import ChatOpenAI

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
    }
    if not args.no_extra_body and "qwen" in model_name.lower():
        kwargs["extra_body"] = {"enable_thinking": False, "thinking_budget": 0}
    return ChatOpenAI(**kwargs)


def run_model_probe(llm: Any, model_name: str) -> None:
    from langchain_core.messages import HumanMessage

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


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_runtime_db_from_export() -> None:
    global RAG_RUNTIME_READY
    if RAG_RUNTIME_READY and CHROMA_RUNTIME_DIR.exists():
        return
    if not CHROMA_EXPORT_DIR.exists():
        raise FileNotFoundError(f"Chroma export DB not found at {CHROMA_EXPORT_DIR}")

    reset_directory(CHROMA_RUNTIME_DIR)
    for item in CHROMA_EXPORT_DIR.iterdir():
        destination = CHROMA_RUNTIME_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    RAG_RUNTIME_READY = True


def load_embedding_backend_artifacts() -> Dict[str, Any]:
    metadata_path = CHROMA_RUNTIME_DIR / EMBEDDING_BACKEND_FILE
    metadata = read_json(metadata_path)
    backend_name = metadata["backend"]
    vectorizer = None
    if backend_name == "tfidf":
        with (CHROMA_RUNTIME_DIR / TFIDF_VECTORIZER_FILE).open("rb") as f:
            vectorizer = pickle.load(f)
    return {"backend": backend_name, "vectorizer": vectorizer}


def sanitize_retrieved_document(document: str) -> str:
    text = document

    raw_json_marker = "\nraw_json:\n"
    if raw_json_marker in text:
        text = text.split(raw_json_marker, 1)[0].rstrip()

    source_summary_marker = "\nsource_summary:"
    if source_summary_marker in text:
        tail = text.split(source_summary_marker, 1)[1]
        next_markers = [
            "\nrequirement_text:",
            "\nbusiness_value:",
            "\nimplicit_risk_hints:",
            "\nprevention_and_mitigation_strategies:",
            "\nprocedure:",
        ]
        cut_positions = [tail.find(marker) for marker in next_markers if tail.find(marker) != -1]
        if cut_positions:
            cut_pos = min(cut_positions)
            text = text.split(source_summary_marker, 1)[0] + tail[cut_pos:]
        else:
            text = text.split(source_summary_marker, 1)[0].rstrip()

    return text.strip()


def build_rag_query(case: Dict[str, Any]) -> str:
    query_payload = {
        "basic_flow": case.get("basic_flow", []),
        "alternative_flows": case.get("alternative_flows", []),
    }
    return (
        "Find relevant security threat patterns, attack mechanisms, and impacts for this functional flow.\n"
        f"{json.dumps(query_payload, ensure_ascii=False)}"
    )


def _join_names(items: Sequence[Dict[str, Any]], limit: int = 4) -> str:
    names = [str(item.get("name") or "").strip() for item in items[:limit] if item.get("name")]
    return "; ".join(names)


def _first_name(items: Sequence[Dict[str, Any]]) -> str:
    for item in items:
        name = str(item.get("name") or "").strip()
        if name:
            return name
    return ""


def _format_graphaware_red_hint(item: Dict[str, Any]) -> str:
    focus = item.get("focus") or {}
    focus_type = focus.get("node_type")
    focus_name = str(focus.get("name") or "").strip()
    risk_name = focus_name if focus_type == "Risk" else _first_name(item.get("related_risks") or [])
    attack_name = (
        focus_name
        if focus_type == "AttackPattern"
        else _first_name(item.get("related_attack_patterns") or item.get("related_techniques") or [])
    )
    mitigation_name = _first_name(item.get("candidate_mitigations") or [])
    chain_parts = [
        part
        for part in [
            f"Risk={risk_name}" if risk_name else "",
            f"AttackPatternOrTechnique={attack_name}" if attack_name else "",
            f"MitigationSignal={mitigation_name}" if mitigation_name else "",
        ]
        if part
    ]
    chain = " -> ".join(chain_parts) if chain_parts else "No explicit graph chain; use as weak context only."
    return (
        "graph_reasoning_hint: First verify that the Risk fits the functional flow; "
        "then use the AttackPattern or Technique as the threat mechanism; treat MitigationSignal "
        "only as evidence of a possible missing control, not as a Red Team output.\n"
        f"candidate_graph_chain: {chain}"
    )


def _format_staticgraph_red_document(item: Dict[str, Any]) -> str:
    focus = item.get("focus") or {}
    sections = [
        _format_graphaware_red_hint(item),
        f"focus_type: {focus.get('node_type')}",
        f"focus_name: {focus.get('name')}",
        f"focus_description: {focus.get('description')}",
        f"why_relevant: {item.get('why_relevant')}",
        f"related_risks: {_join_names(item.get('related_risks') or [])}",
        f"related_attack_patterns: {_join_names(item.get('related_attack_patterns') or [])}",
        f"related_techniques: {_join_names(item.get('related_techniques') or [])}",
        f"candidate_mitigations: {_join_names(item.get('candidate_mitigations') or [])}",
    ]
    evidence = item.get("source_evidence") or []
    if evidence:
        sections.append("source_evidence:")
        for evidence_item in evidence[:3]:
            sections.append(
                "- "
                + "; ".join(
                    part
                    for part in [
                        f"name={evidence_item.get('name')}",
                        f"dataset={evidence_item.get('dataset')}",
                        f"description={evidence_item.get('description')}",
                    ]
                    if part
                )
            )
    return "\n".join(section for section in sections if section and not section.endswith(": "))


def retrieve_relevant_knowledge(
    retriever: RecEvoGraphRetriever,
    case: Dict[str, Any],
    n_results: int,
) -> List[Dict[str, Any]]:
    if n_results <= 0:
        return []
    result = retriever.retrieve_red(case, top_k=n_results)
    retrieved_items: List[Dict[str, Any]] = []
    for item in result.get("items", []):
        focus = item.get("focus") or {}
        retrieved_items.append(
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "score_breakdown": item.get("score_breakdown") or {},
                "metadata": {
                    "dataset": focus.get("dataset"),
                    "id": focus.get("node_id"),
                    "node_type": focus.get("node_type"),
                    "source_title": (focus.get("metadata") or {}).get("source_title"),
                    "source_name": (focus.get("metadata") or {}).get("source_name"),
                },
                "document": _format_staticgraph_red_document(item),
                "graph_item": item,
            }
        )
    return retrieved_items


def build_retrieved_knowledge_input(retrieved_items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    knowledge_items: List[Dict[str, Any]] = []
    for item in retrieved_items:
        metadata = item.get("metadata") or {}
        knowledge_items.append(
            {
                "rank": item.get("rank"),
                "source": {
                    "dataset": metadata.get("dataset"),
                    "id": metadata.get("id"),
                    "source_title": metadata.get("source_title"),
                    "source_name": metadata.get("source_name"),
                    "node_type": metadata.get("node_type"),
                },
                "content": str(item.get("document") or "")[:2600],
            }
        )
    return knowledge_items


def build_model_input(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "basic_flow": case.get("basic_flow", []),
        "alternative_flows": case.get("alternative_flows", []),
    }


def strict_parse_json(content: str) -> Any:
    return json.loads(content.strip())


def validate_red_team_output(parsed: Any, expected_case_id: str, valid_bf_ids: Sequence[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    valid_bf_id_set = set(valid_bf_ids)
    _ = expected_case_id

    if not isinstance(parsed, dict):
        return False, ["Output root is not a JSON object."]

    expected_root_fields = {"threat_records"}
    extra_root_fields = sorted(set(parsed.keys()) - expected_root_fields)
    if extra_root_fields:
        errors.append(f"Unexpected root field(s): {', '.join(extra_root_fields)}.")

    threat_records = parsed.get("threat_records")
    if not isinstance(threat_records, list):
        errors.append("threat_records is missing or not a list.")
        return False, errors

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
        if threat.get("threat_id") != expected_threat_id:
            errors.append(f"{context}.threat_id should be {expected_threat_id}, got {threat.get('threat_id')}.")

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

    return not errors, errors


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


def run_red_team_case(
    llm: Any,
    system_prompt: str,
    case: Dict[str, Any],
    registry_row: Dict[str, Any],
    rag_top_k: int,
    retrieved_knowledge: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    case_id = case["use_case_id"]
    model_input = build_model_input(case)
    augmented_model_input = {
        "functional_flow": model_input,
        "retrieved_knowledge": build_retrieved_knowledge_input(retrieved_knowledge),
    }
    valid_bf_ids = [step.get("step_id") for step in case.get("basic_flow", []) if step.get("step_id")]
    started_at = now_utc()
    start_time = time.perf_counter()
    raw_output = ""
    parsed_output = None
    parse_valid = False
    schema_valid = False
    parse_repaired = False
    parse_error = None
    schema_errors: List[str] = []
    response_metadata: Dict[str, Any] = {}

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(augmented_model_input, ensure_ascii=False)),
            ]
        )
        raw_output = str(getattr(response, "content", "") or "")
        response_metadata = jsonable(getattr(response, "response_metadata", {}) or {})
        try:
            normalized_output, parse_repaired = normalize_model_output(raw_output)
            parsed_output = strict_parse_json(normalized_output)
            parse_valid = True
            schema_valid, schema_errors = validate_red_team_output(parsed_output, case_id, valid_bf_ids)
            if schema_valid and isinstance(parsed_output, dict):
                parsed_output = {
                    "use_case_id": case_id,
                    "threat_records": parsed_output.get("threat_records", []),
                }
        except Exception as exc:
            parse_error = str(exc)
    except Exception as exc:
        parse_error = f"Model invocation failed: {exc}"

    duration = time.perf_counter() - start_time
    threat_records = []
    if parse_valid and schema_valid and isinstance(parsed_output, dict):
        threat_records = parsed_output.get("threat_records", [])

    metadata = response_metadata if isinstance(response_metadata, dict) else {}
    return {
        "index": None,
        "use_case_id": case_id,
        "dataset": registry_row.get("dataset"),
        "split": registry_row.get("split"),
        "source_knowledge_id": registry_row.get("source_knowledge_id"),
        "model_input": augmented_model_input,
        "retrieved_knowledge": retrieved_knowledge,
        "retrieval_top_k": rag_top_k,
        "retrieval_count": len(retrieved_knowledge),
        "raw_model_output": raw_output,
        "parsed_output": parsed_output,
        "threat_records": threat_records,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "parse_repaired": parse_repaired,
        "parse_repair_note": "stripped outer markdown code fence" if parse_repaired else None,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "red_team_duration_seconds": duration,
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "response_model": metadata.get("model_name") or metadata.get("model") or metadata.get("model_id"),
        "finish_reason": metadata.get("finish_reason"),
        "token_usage": get_response_usage(metadata),
    }


def summarize_results(results: Sequence[Dict[str, Any]], selected_case_count: int, invocation_start: float) -> Dict[str, Any]:
    durations = [float(item.get("red_team_duration_seconds") or 0.0) for item in results]
    schema_valid_count = sum(1 for item in results if item.get("schema_valid"))
    parse_valid_count = sum(1 for item in results if item.get("parse_valid"))
    parse_repaired_count = sum(1 for item in results if item.get("parse_repaired"))
    failed_count = sum(1 for item in results if not item.get("schema_valid"))
    retrieval_case_count = sum(1 for item in results if int(item.get("retrieval_count") or 0) > 0)

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
        "schema_valid_case_count": schema_valid_count,
        "parse_repaired_case_count": parse_repaired_count,
        "failed_case_count": failed_count,
        "schema_valid_rate": schema_valid_count / len(results) if results else 0.0,
        "retrieval_case_count": retrieval_case_count,
        "retrieval_case_rate": retrieval_case_count / len(results) if results else 0.0,
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


def write_prediction_artifact(
    artifact_path: Path,
    meta: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    invalid_case_ids = [item["use_case_id"] for item in results if not item.get("schema_valid")]
    cases = []
    for item in results:
        threat_records = item.get("threat_records") if item.get("schema_valid") else []
        cases.append(
            {
                "use_case_id": item["use_case_id"],
                "dataset": item.get("dataset"),
                "split": item.get("split"),
                "source_knowledge_id": item.get("source_knowledge_id"),
                "threat_records": threat_records,
            }
        )

    payload = {
        "meta": {
            "dataset_name": "SAAFG Red Team StaticGraphRAG-GraphAware predicted threat records",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "source_run_output_path": str(output_path),
            "experiment_id": meta["experiment_id"],
            "case_count": len(cases),
            "invalid_case_count": len(invalid_case_ids),
            "invalid_case_ids": invalid_case_ids,
            "notes": [
                "Red Team receives ranked graph evidence from the static Rec-EvoGraph-RAG knowledge graph.",
                "The model outputs threat_records only; use_case_id is attached deterministically by the experiment script.",
                "No retries, repair, or field imputation are applied.",
                "Invalid model outputs are represented as empty threat_records in this prediction artifact; raw failures remain in the run output.",
            ],
        },
        "threat_record_cases": cases,
    }
    write_json_atomic(artifact_path, payload)


def append_retrieval_trace(trace_path: Path, result: Dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "use_case_id": result.get("use_case_id"),
        "dataset": result.get("dataset"),
        "split": result.get("split"),
        "source_knowledge_id": result.get("source_knowledge_id"),
        "retrieval_top_k": result.get("retrieval_top_k"),
        "retrieval_count": result.get("retrieval_count"),
        "retrieved_knowledge": result.get("retrieved_knowledge") or [],
    }
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


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

    resolve_paths(args, model_name)
    configure_stream_logging(args.log_path)

    print(f"[Config] input_path={args.input_path}")
    print(f"[Config] case_registry_path={args.case_registry_path}")
    print(f"[Config] prompt_path={args.prompt_path}")
    print(f"[Config] output_path={args.output_path}")
    print(f"[Config] artifact_path={args.artifact_path}")
    print(f"[Config] retrieval_trace_path={args.retrieval_trace_path}")
    print(f"[Config] kg_dir={args.kg_dir}")
    print(f"[Config] rag_top_k={args.rag_top_k}")
    print(f"[Config] model={model_name}")
    print("[Config] policy=no retries, no repair, no imputation")
    print("[Config] parse_normalization=strip_outer_markdown_code_fence")

    if args.probe_only:
        llm = build_llm(args, api_key, base_url, model_name)
        if not args.skip_probe:
            run_model_probe(llm, model_name)
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
        "task": "SAAFG Task A StaticGraphRAG-GraphAware Red Team generation",
        "version": "v0.2",
        "no_rag": False,
        "rag_enabled": True,
        "rag_method": "StaticGraphRAG-GraphAware",
        "rag_top_k": args.rag_top_k,
        "kg_dir": str(args.kg_dir),
        "retrieval_trace_path": str(args.retrieval_trace_path),
        "model_name": model_name,
        "model_env_var": args.model_env_var,
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
        "input_path": str(args.input_path),
        "case_registry_path": str(args.case_registry_path),
        "prompt_path": str(args.prompt_path),
        "started_at_utc": existing_meta.get("started_at_utc") or now_utc(),
        "generation_policy": (
            "StaticGraphRAG ranked graph retrieval with graph-aware Risk -> AttackPattern -> "
            "Mitigation prompt guidance, no retries, no JSON repair, no schema correction, "
            "no missing-field imputation"
        ),
        "parse_normalization": "strip_outer_markdown_code_fence_before_strict_json_parse",
        "script_injected_fields": ["use_case_id"],
    }

    invocation_start = time.perf_counter()
    total = len(selected_cases)
    print(f"[Run] Selected {total} case(s).")
    if not args.resume:
        args.retrieval_trace_path.parent.mkdir(parents=True, exist_ok=True)
        args.retrieval_trace_path.write_text("", encoding="utf-8")

    retriever = RecEvoGraphRetriever(args.kg_dir)
    retrieval_by_case_id: Dict[str, List[Dict[str, Any]]] = {}
    for position, case in enumerate(selected_cases, start=1):
        case_id = case["use_case_id"]
        if case_id in completed_ids:
            continue
        print(f"[RAG] {position}/{total} retrieving knowledge for {case_id}")
        retrieved_knowledge = retrieve_relevant_knowledge(retriever, case, args.rag_top_k)
        retrieval_by_case_id[case_id] = retrieved_knowledge
        print(f"[RAG] {case_id} retrieved={len(retrieved_knowledge)}")

    llm = build_llm(args, api_key, base_url, model_name)
    if not args.skip_probe:
        run_model_probe(llm, model_name)

    for position, case in enumerate(selected_cases, start=1):
        case_id = case["use_case_id"]
        if case_id in completed_ids:
            print(f"[Skip] {position}/{total} {case_id} already completed.")
            continue
        registry_row = registry_map.get(case_id)
        if not registry_row:
            raise ValueError(f"Missing case registry row for {case_id}")

        print(f"[Run] {position}/{total} {case_id}")
        result = run_red_team_case(
            llm,
            system_prompt,
            case,
            registry_row,
            args.rag_top_k,
            retrieval_by_case_id[case_id],
        )
        result["index"] = position - 1
        results.append(result)
        completed_ids.add(case_id)
        append_retrieval_trace(args.retrieval_trace_path, result)
        status = "valid" if result.get("schema_valid") else "invalid"
        print(
            "[Done] {} status={} duration={:.3f}s retrieval={} threats={}".format(
                case_id,
                status,
                float(result.get("red_team_duration_seconds") or 0.0),
                int(result.get("retrieval_count") or 0),
                len(result.get("threat_records") or []),
            )
        )
        write_run_output(args.output_path, meta, results, total, invocation_start)
        write_prediction_artifact(args.artifact_path, meta, results, args.output_path)

    write_run_output(args.output_path, meta, results, total, invocation_start)
    write_prediction_artifact(args.artifact_path, meta, results, args.output_path)
    summary = summarize_results(results, total, invocation_start)
    print("[Summary] completed={completed_case_count}/{selected_case_count}".format(**summary))
    print("[Summary] schema_valid_rate={:.4f}".format(summary["schema_valid_rate"]))
    if summary.get("parse_repaired_case_count"):
        print("[Summary] parse_repaired_case_count={}".format(summary["parse_repaired_case_count"]))
    print("[Summary] total_case_duration_seconds={:.3f}".format(summary["total_case_duration_seconds"]))
    print("[Summary] average_case_duration_seconds={:.3f}".format(summary["average_case_duration_seconds"]))
    print(f"[Summary] output_path={args.output_path}")
    print(f"[Summary] artifact_path={args.artifact_path}")


if __name__ == "__main__":
    main()
