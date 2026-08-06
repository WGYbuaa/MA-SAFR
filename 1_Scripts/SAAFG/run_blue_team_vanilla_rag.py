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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"

DEFAULT_FLOW_PATH = (
    SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
)
DEFAULT_PROMPT_PATH = BASE_DIR / "3_Prompt" / "SAAFG" / "blue_team_vanilla_rag.txt"
DEFAULT_EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
DEFAULT_BLUE_RUN_TAG = "qwen35plus"
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"
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
    raise OSError(f"Failed to replace JSON output after retries: {path}") from last_error


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
    parser = argparse.ArgumentParser(description="Run SAAFG v0.2 VanillaRAG Blue Team generation.")
    parser.add_argument("--run-tag", default=None, help="Optional output tag override, e.g. qwen35plus.")
    parser.add_argument(
        "--source-run-tag",
        default=None,
        help="Optional red-team source tag override, e.g. qwen35plus or deepseek-v32.",
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_FLOW_PATH)
    parser.add_argument("--source-run-output-path", type=Path, default=None)
    parser.add_argument("--source-eval-output-path", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--retrieval-trace-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--rag-top-k", type=int, default=3, help="Number of Chroma knowledge chunks to retrieve.")
    parser.add_argument("--case-id", nargs="*", default=None, help="Optional explicit use_case_id list.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N selected cases.")
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


def resolve_paths(args: argparse.Namespace, model_name: str) -> Tuple[str, str]:
    blue_tag = args.run_tag or model_slug(model_name) or DEFAULT_BLUE_RUN_TAG
    source_tag = args.source_run_tag or blue_tag
    if args.result_dir is None:
        args.result_dir = DEFAULT_EXPERIMENT_ROOT / f"ma_VanillaRAG_{blue_tag}" / "blue_team"
    if args.source_run_output_path is None:
        args.source_run_output_path = (
            DEFAULT_EXPERIMENT_ROOT
            / f"ma_VanillaRAG_{source_tag}"
            / "red_team"
            / f"saafg_redteam_vanillarag_v0_2_{source_tag}.json"
        )
    if args.source_eval_output_path is None:
        args.source_eval_output_path = (
            DEFAULT_EXPERIMENT_ROOT
            / f"ma_VanillaRAG_{source_tag}"
            / "red_team"
            / f"saafg_redteam_task_a_eval_vanillarag_v0_2_{source_tag}.json"
        )
    if args.output_path is None:
        args.output_path = args.result_dir / f"saafg_blueteam_vanillarag_v0_2_{blue_tag}.json"
    if args.artifact_path is None:
        args.artifact_path = (
            args.result_dir / f"saafg_security_augmented_flows_pred_vanillarag_v0_2_{blue_tag}.json"
        )
    if args.retrieval_trace_path is None:
        args.retrieval_trace_path = (
            args.result_dir / f"saafg_blueteam_retrieval_trace_vanillarag_v0_2_{blue_tag}.jsonl"
        )
    if args.log_path is None:
        args.log_path = args.result_dir / f"saafg_blueteam_vanillarag_v0_2_{blue_tag}.log"
    return blue_tag, source_tag


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
    if not args.no_extra_body:
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


def load_red_team_results(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Expected results list in {path}")
    return payload


def load_red_team_eval_results(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    case_reports = payload.get("case_reports")
    if not isinstance(case_reports, list):
        raise ValueError(f"Expected case_reports list in {path}")
    return payload


def select_cases(
    cases: Sequence[Dict[str, Any]],
    case_ids: Optional[List[str]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
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


def simplify_basic_flow(case: Dict[str, Any]) -> List[Dict[str, str]]:
    simplified = []
    for step in case.get("basic_flow", []) or []:
        step_id = str(step.get("step_id") or "").strip()
        step_sentence = str(step.get("step_sentence") or "").strip()
        if not step_id or not step_sentence:
            raise ValueError(f"Invalid basic_flow step in {case.get('use_case_id')}")
        simplified.append({"step_id": step_id, "step_sentence": step_sentence})
    return simplified


def build_threat_input(flow_case: Dict[str, Any], threat: Dict[str, Any]) -> Dict[str, Any]:
    threat_input = {
        "threat_id": threat["threat_id"],
        "anchor_steps": threat.get("anchor_steps", []),
        "threat_name": threat.get("threat_name", ""),
        "threat_mechanism": threat.get("threat_mechanism", ""),
        "security_impact": threat.get("security_impact", ""),
    }
    return {
        "basic_flow": simplify_basic_flow(flow_case),
        "threat": threat_input,
    }


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


def build_rag_query(flow_case: Dict[str, Any], threat: Dict[str, Any]) -> str:
    query_payload = build_threat_input(flow_case, threat)
    return (
        "Find relevant security mitigations, validation points, guardrails, and recovery steps "
        "for this threat in this functional flow.\n"
        f"{json.dumps(query_payload, ensure_ascii=False)}"
    )


def retrieve_relevant_knowledge(
    flow_case: Dict[str, Any],
    threat: Dict[str, Any],
    n_results: int,
) -> List[Dict[str, Any]]:
    if n_results <= 0:
        return []

    global RAG_RUNTIME_READY
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            if attempt > 0:
                RAG_RUNTIME_READY = False
            ensure_runtime_db_from_export()
            client = chromadb.PersistentClient(path=str(CHROMA_RUNTIME_DIR))
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
            backend_info = load_embedding_backend_artifacts()
            query = build_rag_query(flow_case, threat)

            if backend_info["backend"] == "onnx_minilm_l6_v2":
                query_embedding = embedding_functions.ONNXMiniLM_L6_V2()([query])[0]
            else:
                vectorizer: TfidfVectorizer = backend_info["vectorizer"]
                query_embedding = vectorizer.transform([query]).astype("float32").toarray().tolist()[0]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            retrieved_items: List[Dict[str, Any]] = []
            for idx, document in enumerate(documents):
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                distance = distances[idx] if idx < len(distances) else None
                retrieved_items.append(
                    {
                        "rank": idx + 1,
                        "distance": distance,
                        "metadata": metadata,
                        "document": sanitize_retrieved_document(document),
                    }
                )
            return retrieved_items
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Chroma retrieval failed after retry: {last_error}")


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
                },
                "content": str(item.get("document") or "")[:2000],
            }
        )
    return knowledge_items


def load_eval_case_lookup(eval_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    case_reports = eval_payload.get("case_reports") or []
    lookup: Dict[str, Dict[str, Any]] = {}
    for case_report in case_reports:
        case_id = str(case_report.get("use_case_id") or "").strip()
        if case_id:
            lookup[case_id] = case_report
    return lookup


def extract_effective_threat_ids(eval_case: Dict[str, Any]) -> List[str]:
    matched_pairs = eval_case.get("semantic_match_pairs") or []
    threat_ids: List[str] = []
    seen: set[str] = set()
    for pair in matched_pairs:
        threat_id = str(pair.get("predicted_threat_id") or "").strip()
        if threat_id and threat_id not in seen:
            seen.add(threat_id)
            threat_ids.append(threat_id)
    return threat_ids


def select_effective_threat_records(
    source_case_result: Dict[str, Any],
    eval_case: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], bool, bool, List[str], List[str]]:
    raw_threat_records = source_case_result.get("threat_records") or []
    source_red_team_parse_valid = source_case_result.get("parse_valid") is True
    source_red_team_schema_valid = source_case_result.get("schema_valid") is True
    eval_match_order_threat_ids = extract_effective_threat_ids(eval_case)
    effective_threat_id_set = set(eval_match_order_threat_ids)
    threat_records = [
        threat
        for threat in raw_threat_records
        if source_red_team_schema_valid
        and str(threat.get("threat_id") or "").strip() in effective_threat_id_set
    ]
    effective_threat_ids = [
        str(threat.get("threat_id") or "").strip()
        for threat in threat_records
    ]
    return (
        threat_records,
        source_red_team_parse_valid,
        source_red_team_schema_valid,
        eval_match_order_threat_ids,
        effective_threat_ids,
    )


def strict_parse_json(content: str) -> Any:
    return json.loads(content.strip())


def validate_blue_team_output(parsed: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(parsed, dict):
        return False, ["Output root is not a JSON object."]

    expected_root_fields = {"security_basic_flow", "security_alternative_flow"}
    extra_root_fields = sorted(set(parsed.keys()) - expected_root_fields)
    if extra_root_fields:
        errors.append(f"Unexpected root field(s): {', '.join(extra_root_fields)}.")

    basic_flow = parsed.get("security_basic_flow")
    if not isinstance(basic_flow, dict):
        errors.append("security_basic_flow is missing or not an object.")
    else:
        extra_basic_fields = sorted(set(basic_flow.keys()) - {"step_sentence"})
        if extra_basic_fields:
            errors.append(f"security_basic_flow has unexpected field(s): {', '.join(extra_basic_fields)}.")
        step_sentence = basic_flow.get("step_sentence")
        if not isinstance(step_sentence, str) or not step_sentence.strip():
            errors.append("security_basic_flow.step_sentence must be a non-empty string.")

    alt_flow = parsed.get("security_alternative_flow")
    if not isinstance(alt_flow, dict):
        errors.append("security_alternative_flow is missing or not an object.")
    else:
        extra_alt_fields = sorted(set(alt_flow.keys()) - {"entry_condition", "steps"})
        if extra_alt_fields:
            errors.append(
                f"security_alternative_flow has unexpected field(s): {', '.join(extra_alt_fields)}."
            )
        entry_condition = alt_flow.get("entry_condition")
        if not isinstance(entry_condition, str) or not entry_condition.strip():
            errors.append("security_alternative_flow.entry_condition must be a non-empty string.")
        steps = alt_flow.get("steps")
        if not isinstance(steps, list) or not steps:
            errors.append("security_alternative_flow.steps must be a non-empty list.")
        else:
            expected_step_fields = {"step_id", "step_sentence"}
            for index, step in enumerate(steps, start=1):
                context = f"security_alternative_flow.steps[{index - 1}]"
                if not isinstance(step, dict):
                    errors.append(f"{context} is not an object.")
                    continue
                extra_step_fields = sorted(set(step.keys()) - expected_step_fields)
                if extra_step_fields:
                    errors.append(f"{context} has unexpected field(s): {', '.join(extra_step_fields)}.")
                step_id = step.get("step_id")
                step_sentence = step.get("step_sentence")
                if not isinstance(step_id, str) or not step_id.strip():
                    errors.append(f"{context}.step_id must be a non-empty string.")
                if not isinstance(step_sentence, str) or not step_sentence.strip():
                    errors.append(f"{context}.step_sentence must be a non-empty string.")

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


def normalize_anchor_step_id(anchor_steps: Sequence[Any]) -> str:
    if not anchor_steps:
        raise ValueError("Missing anchor_steps in source threat.")
    anchor_step_id = str(anchor_steps[0]).strip()
    if not anchor_step_id:
        raise ValueError("Empty anchor_step_id in source threat.")
    return anchor_step_id


def anchor_num_from_step_id(step_id: str) -> int:
    match = re.search(r"(\d+)$", step_id or "")
    return int(match.group(1)) if match else 1


def suffix_for_occurrence(occurrence_index: int) -> str:
    if occurrence_index <= 1:
        return ""
    return chr(ord("a") + occurrence_index - 2)


def run_blue_threat(
    llm: Any,
    system_prompt: str,
    flow_case: Dict[str, Any],
    threat: Dict[str, Any],
    rag_top_k: int,
    retrieved_knowledge: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    threat_id = threat.get("threat_id")
    anchor_step_id = normalize_anchor_step_id(threat.get("anchor_steps") or [])
    model_input = build_threat_input(flow_case, threat)
    augmented_model_input = {
        **model_input,
        "retrieved_knowledge": build_retrieved_knowledge_input(retrieved_knowledge),
    }
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
        print(
            "[Blue Input] threat={} anchor={} retrieval={}".format(
                threat_id,
                anchor_step_id,
                len(retrieved_knowledge),
            )
        )
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(augmented_model_input, ensure_ascii=False, separators=(",", ":"))),
            ]
        )
        raw_output = str(getattr(response, "content", "") or "")
        response_metadata = jsonable(getattr(response, "response_metadata", {}) or {})
        try:
            normalized_output, parse_repaired = normalize_model_output(raw_output)
            parsed_output = strict_parse_json(normalized_output)
            parse_valid = True
            schema_valid, schema_errors = validate_blue_team_output(parsed_output)
        except Exception as exc:
            parse_error = str(exc)
    except Exception as exc:
        parse_error = f"Model invocation failed: {exc}"

    duration = time.perf_counter() - start_time
    return {
        "threat_id": threat_id,
        "anchor_step_id": anchor_step_id,
        "model_input": augmented_model_input,
        "retrieved_knowledge": list(retrieved_knowledge),
        "retrieval_top_k": rag_top_k,
        "retrieval_count": len(retrieved_knowledge),
        "raw_model_output": raw_output,
        "parsed_output": parsed_output,
        "parse_valid": parse_valid,
        "schema_valid": schema_valid,
        "parse_repaired": parse_repaired,
        "parse_repair_note": "stripped outer markdown code fence" if parse_repaired else None,
        "parse_error": parse_error,
        "schema_errors": schema_errors,
        "blue_team_duration_seconds": duration,
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "response_model": response_metadata.get("model_name")
        or response_metadata.get("model")
        or response_metadata.get("model_id"),
        "finish_reason": response_metadata.get("finish_reason"),
        "token_usage": get_response_usage(response_metadata),
    }


def build_case_artifact(
    flow_case: Dict[str, Any],
    threat_generation_results: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    valid_results = [item for item in threat_generation_results if item.get("schema_valid")]
    suffix_counter: Dict[int, int] = defaultdict(int)
    security_basic_flow: List[Dict[str, Any]] = []
    security_alternative_flows: List[Dict[str, Any]] = []

    for result in valid_results:
        threat_input = result["model_input"]["threat"]
        threat_id = threat_input["threat_id"]
        anchor_step_id = result["anchor_step_id"]
        anchor_num = anchor_num_from_step_id(anchor_step_id)
        suffix_counter[anchor_num] += 1
        suffix = suffix_for_occurrence(suffix_counter[anchor_num])
        sbf_id = f"SBF{anchor_num}{suffix}"
        saf_id = f"SAF{anchor_num}{suffix}"

        basic_sentence = result["parsed_output"]["security_basic_flow"]["step_sentence"]
        alt_flow = result["parsed_output"]["security_alternative_flow"]
        alt_steps = []
        for index, step in enumerate(alt_flow.get("steps") or [], start=1):
            alt_steps.append(
                {
                    "step_id": f"{saf_id}.{index}",
                    "step_sentence": step["step_sentence"],
                }
            )

        security_basic_flow.append(
            {
                "step_id": sbf_id,
                "anchor_after": anchor_step_id,
                "step_sentence": basic_sentence,
            }
        )
        security_alternative_flows.append(
            {
                "saf_id": saf_id,
                "mitigates": [threat_id],
                "entry_condition": alt_flow["entry_condition"],
                "steps": alt_steps,
            }
        )

        result["sbf_step_id"] = sbf_id
        result["saf_id"] = saf_id

    aggregate = {
        "use_case_id": flow_case["use_case_id"],
        "security_augmented_flow": {
            "security_basic_flow": security_basic_flow,
            "security_alternative_flows": security_alternative_flows,
        },
    }
    return aggregate, {
        "valid_threat_count": len(valid_results),
        "invalid_threat_count": len(threat_generation_results) - len(valid_results),
        "security_basic_flow_count": len(security_basic_flow),
        "security_alternative_flow_count": len(security_alternative_flows),
    }


def summarize_results(results: Sequence[Dict[str, Any]], selected_case_count: int, invocation_start: float) -> Dict[str, Any]:
    case_durations = [float(item.get("blue_team_case_duration_seconds") or 0.0) for item in results]
    threat_results = [
        threat_result
        for case_result in results
        for threat_result in (case_result.get("threat_generation_results") or [])
    ]
    valid_threat_count = sum(1 for item in threat_results if item.get("schema_valid"))
    parse_valid_threat_count = sum(1 for item in threat_results if item.get("parse_valid"))
    repaired_threat_count = sum(1 for item in threat_results if item.get("parse_repaired"))
    case_schema_valid_count = sum(1 for item in results if item.get("case_schema_valid"))
    case_parse_valid_count = sum(1 for item in results if item.get("case_parse_valid"))
    empty_threat_case_count = sum(1 for item in results if not item.get("source_threat_count"))
    retrieval_threat_count = sum(1 for item in threat_results if int(item.get("retrieval_count") or 0) > 0)

    usage_totals: Dict[str, float] = {}
    for item in threat_results:
        usage = item.get("token_usage") or {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    usage_totals[key] = usage_totals.get(key, 0.0) + float(value)

    source_threat_count = sum(int(item.get("source_threat_count") or 0) for item in results)
    raw_source_threat_count = sum(int(item.get("raw_source_threat_count") or 0) for item in results)

    return {
        "selected_case_count": selected_case_count,
        "completed_case_count": len(results),
        "raw_source_threat_count": raw_source_threat_count,
        "source_threat_count": source_threat_count,
        "valid_threat_count": valid_threat_count,
        "parse_valid_threat_count": parse_valid_threat_count,
        "schema_valid_threat_count": valid_threat_count,
        "failed_threat_count": source_threat_count - valid_threat_count,
        "case_parse_valid_count": case_parse_valid_count,
        "case_schema_valid_count": case_schema_valid_count,
        "empty_threat_case_count": empty_threat_case_count,
        "case_parse_valid_rate": case_parse_valid_count / len(results) if results else 0.0,
        "case_schema_valid_rate": case_schema_valid_count / len(results) if results else 0.0,
        "threat_schema_valid_rate": valid_threat_count / source_threat_count if source_threat_count else 0.0,
        "retrieval_threat_count": retrieval_threat_count,
        "retrieval_threat_rate": retrieval_threat_count / source_threat_count if source_threat_count else 0.0,
        "parse_repaired_threat_count": repaired_threat_count,
        "total_case_duration_seconds": sum(case_durations),
        "average_case_duration_seconds": sum(case_durations) / len(case_durations) if case_durations else 0.0,
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


def write_artifact(
    artifact_path: Path,
    meta: Dict[str, Any],
    case_results: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    valid_threat_count = sum(int(item.get("valid_threat_count") or 0) for item in case_results)
    invalid_threat_count = sum(int(item.get("invalid_threat_count") or 0) for item in case_results)
    empty_case_count = sum(1 for item in case_results if not item.get("source_threat_count"))
    payload = {
        "meta": {
            "dataset_name": "SAAFG Blue Team VanillaRAG predicted security augmented flows",
            "version": "v0.2",
            "generated_at_utc": now_utc(),
            "source_run_output_path": str(meta["source_run_output_path"]),
            "experiment_id": meta["experiment_id"],
            "source_eval_output_path": str(meta["source_eval_output_path"]),
            "case_count": len(case_results),
            "valid_threat_count": valid_threat_count,
            "invalid_threat_count": invalid_threat_count,
            "empty_threat_case_count": empty_case_count,
            "notes": [
                "Blue Team receives retrieved external security knowledge from the local Chroma knowledge base.",
                "Security-augmented flows are generated from Task A evaluation-matched Red Team threats only.",
                "SBF/SAF ids, anchor_after, and mitigates are attached deterministically by the script.",
                "Invalid threat generations are omitted from the artifact; raw failures remain in the run output.",
            ],
        },
        "security_augmented_flow_cases": [
            {
                "use_case_id": item["use_case_id"],
                "dataset": item.get("dataset"),
                "split": item.get("split"),
                "source_knowledge_id": item.get("source_knowledge_id"),
                "security_augmented_flow": item["security_augmented_flow"],
            }
            for item in case_results
        ],
    }
    write_json_atomic(artifact_path, payload)


def append_retrieval_trace(
    trace_path: Path,
    case_id: str,
    case_metadata: Dict[str, Any],
    threat: Dict[str, Any],
    retrieved_knowledge: Sequence[Dict[str, Any]],
    rag_top_k: int,
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "use_case_id": case_id,
        "dataset": case_metadata.get("dataset"),
        "split": case_metadata.get("split"),
        "source_knowledge_id": case_metadata.get("source_knowledge_id"),
        "threat_id": threat.get("threat_id"),
        "anchor_steps": threat.get("anchor_steps") or [],
        "threat_name": threat.get("threat_name"),
        "retrieval_top_k": rag_top_k,
        "retrieval_count": len(retrieved_knowledge),
        "retrieved_knowledge": list(retrieved_knowledge),
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

    blue_tag, source_tag = resolve_paths(args, model_name)
    configure_stream_logging(args.log_path)

    print(f"[Config] input_path={args.input_path}")
    print(f"[Config] source_run_output_path={args.source_run_output_path}")
    print(f"[Config] source_eval_output_path={args.source_eval_output_path}")
    print(f"[Config] prompt_path={args.prompt_path}")
    print(f"[Config] output_path={args.output_path}")
    print(f"[Config] artifact_path={args.artifact_path}")
    print(f"[Config] retrieval_trace_path={args.retrieval_trace_path}")
    print(f"[Config] chroma_export_dir={CHROMA_EXPORT_DIR}")
    print(f"[Config] chroma_runtime_dir={CHROMA_RUNTIME_DIR}")
    print(f"[Config] rag_top_k={args.rag_top_k}")
    print(f"[Config] model={model_name}")
    print("[Config] policy=no retries, no repair, no imputation")
    print("[Config] parse_normalization=strip_outer_markdown_code_fence")
    print(f"[Config] run_tag={blue_tag}")
    print(f"[Config] source_run_tag={source_tag}")

    if args.probe_only:
        llm = build_llm(args, api_key, base_url, model_name)
        if not args.skip_probe:
            run_model_probe(llm, model_name)
        return

    system_prompt = load_text(args.prompt_path)
    functional_cases = load_functional_cases(args.input_path)
    flow_map = {case["use_case_id"]: case for case in functional_cases}
    source_payload = load_red_team_results(args.source_run_output_path)
    eval_payload = load_red_team_eval_results(args.source_eval_output_path)
    eval_lookup = load_eval_case_lookup(eval_payload)
    source_results = source_payload.get("results") or []
    if not isinstance(source_results, list):
        raise ValueError(f"Source red-team run output has invalid results field: {args.source_run_output_path}")

    selected_source_results = select_cases(source_results, args.case_id, args.limit)

    existing_meta: Dict[str, Any] = {}
    case_results: List[Dict[str, Any]] = []
    if args.resume and args.output_path.exists():
        existing_meta, case_results = load_existing_results(args.output_path)
        print(f"[Resume] Loaded {len(case_results)} existing case result(s).")

    completed_case_ids = {item.get("use_case_id") for item in case_results}
    experiment_id = args.output_path.stem
    meta = {
        **existing_meta,
        "experiment_id": existing_meta.get("experiment_id") or experiment_id,
        "task": "SAAFG Task B VanillaRAG Blue Team generation",
        "version": "v0.2",
        "no_rag": False,
        "rag_enabled": True,
        "rag_method": "VanillaRAG",
        "rag_top_k": args.rag_top_k,
        "chroma_export_dir": str(CHROMA_EXPORT_DIR),
        "chroma_collection_name": CHROMA_COLLECTION_NAME,
        "retrieval_trace_path": str(args.retrieval_trace_path),
        "model_name": model_name,
        "model_env_var": args.model_env_var,
        "source_run_tag": source_tag,
        "source_run_output_path": str(args.source_run_output_path),
        "source_eval_output_path": str(args.source_eval_output_path),
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
        "input_path": str(args.input_path),
        "prompt_path": str(args.prompt_path),
        "started_at_utc": existing_meta.get("started_at_utc") or now_utc(),
        "generation_policy": "VanillaRAG top-k retrieval, no retries, no JSON repair, no schema correction, no missing-field imputation",
        "parse_normalization": "strip_outer_markdown_code_fence_before_strict_json_parse",
    }

    invocation_start = time.perf_counter()
    total = len(selected_source_results)
    print(f"[Run] Selected {total} case(s).")
    if not args.resume:
        args.retrieval_trace_path.parent.mkdir(parents=True, exist_ok=True)
        args.retrieval_trace_path.write_text("", encoding="utf-8")

    retrieval_by_case_threat: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for position, source_case_result in enumerate(selected_source_results, start=1):
        case_id = source_case_result["use_case_id"]
        if case_id in completed_case_ids:
            continue

        flow_case = flow_map.get(case_id)
        if not flow_case:
            raise ValueError(f"Missing functional flow row for {case_id}")

        eval_case = eval_lookup.get(case_id)
        if eval_case is None:
            raise ValueError(f"Missing red-team Task A evaluation row for {case_id}")

        (
            threat_records,
            _source_red_team_parse_valid,
            _source_red_team_schema_valid,
            _eval_match_order_threat_ids,
            _effective_threat_ids,
        ) = select_effective_threat_records(source_case_result, eval_case)

        for threat_index, threat in enumerate(threat_records, start=1):
            threat_id = str(threat.get("threat_id") or "").strip()
            print(f"[RAG] {position}/{total} {case_id} threat={threat_id} ({threat_index}/{len(threat_records)})")
            retrieved_knowledge = retrieve_relevant_knowledge(flow_case, threat, args.rag_top_k)
            retrieval_by_case_threat[(case_id, threat_id)] = retrieved_knowledge
            append_retrieval_trace(
                args.retrieval_trace_path,
                case_id,
                source_case_result,
                threat,
                retrieved_knowledge,
                args.rag_top_k,
            )
            print(f"[RAG] {case_id} threat={threat_id} retrieved={len(retrieved_knowledge)}")

    llm = build_llm(args, api_key, base_url, model_name)
    if not args.skip_probe:
        run_model_probe(llm, model_name)

    for position, source_case_result in enumerate(selected_source_results, start=1):
        case_id = source_case_result["use_case_id"]
        if case_id in completed_case_ids:
            print(f"[Skip] {position}/{total} {case_id} already completed.")
            continue

        flow_case = flow_map.get(case_id)
        if not flow_case:
            raise ValueError(f"Missing functional flow row for {case_id}")

        eval_case = eval_lookup.get(case_id)
        if eval_case is None:
            raise ValueError(f"Missing red-team Task A evaluation row for {case_id}")

        raw_threat_records = source_case_result.get("threat_records") or []
        (
            threat_records,
            source_red_team_parse_valid,
            source_red_team_schema_valid,
            eval_match_order_threat_ids,
            effective_threat_ids,
        ) = select_effective_threat_records(source_case_result, eval_case)
        print(
            "[Run] {}/{} {} raw_threats={} eval_valid_threats={} eligible_threats={} source_schema_valid={}".format(
                position,
                total,
                case_id,
                len(raw_threat_records),
                len(eval_match_order_threat_ids),
                len(threat_records),
                source_red_team_schema_valid,
            )
        )

        case_started_at = now_utc()
        case_start_time = time.perf_counter()
        threat_generation_results: List[Dict[str, Any]] = []

        for threat_index, threat in enumerate(threat_records, start=1):
            threat_id = str(threat.get("threat_id") or "").strip()
            threat_result = run_blue_threat(
                llm,
                system_prompt,
                flow_case,
                threat,
                args.rag_top_k,
                retrieval_by_case_threat[(case_id, threat_id)],
            )
            threat_result["index"] = threat_index - 1
            threat_generation_results.append(threat_result)
            status = "valid" if threat_result.get("schema_valid") else "invalid"
            print(
                "[Done] {} threat={} status={} duration={:.3f}s retrieval={}".format(
                    case_id,
                    threat.get("threat_id"),
                    status,
                    float(threat_result.get("blue_team_duration_seconds") or 0.0),
                    int(threat_result.get("retrieval_count") or 0),
                )
            )

        case_artifact, case_counts = build_case_artifact(flow_case, threat_generation_results)
        case_duration = time.perf_counter() - case_start_time
        case_parse_valid = all(item.get("parse_valid") for item in threat_generation_results) if threat_generation_results else True
        case_schema_valid = all(item.get("schema_valid") for item in threat_generation_results) if threat_generation_results else True
        token_usage_totals: Dict[str, float] = {}
        for item in threat_generation_results:
            usage = item.get("token_usage") or {}
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, (int, float)):
                        token_usage_totals[key] = token_usage_totals.get(key, 0.0) + float(value)

        case_result = {
            "index": position - 1,
            "use_case_id": case_id,
            "dataset": source_case_result.get("dataset"),
            "split": source_case_result.get("split"),
            "source_knowledge_id": source_case_result.get("source_knowledge_id"),
            "input_basic_flow": simplify_basic_flow(flow_case),
            "source_red_team_parse_valid": source_red_team_parse_valid,
            "source_red_team_schema_valid": source_red_team_schema_valid,
            "source_red_team_eval_match_order_threat_ids": eval_match_order_threat_ids,
            "source_red_team_eval_valid_threat_ids": effective_threat_ids,
            "source_red_team_eval_semantic_match_pairs": eval_case.get("semantic_match_pairs") or [],
            "source_red_team_eval_threat_validity_match_count": eval_case.get("threat_validity_match_count"),
            "raw_source_threat_count": len(raw_threat_records),
            "source_threat_count": len(threat_records),
            "threat_generation_results": threat_generation_results,
            "security_augmented_flow": case_artifact["security_augmented_flow"],
            "valid_threat_count": case_counts["valid_threat_count"],
            "invalid_threat_count": case_counts["invalid_threat_count"],
            "case_parse_valid": case_parse_valid,
            "case_schema_valid": case_schema_valid,
            "blue_team_case_duration_seconds": case_duration,
            "started_at_utc": case_started_at,
            "finished_at_utc": now_utc(),
            "token_usage": token_usage_totals,
        }
        case_results.append(case_result)
        completed_case_ids.add(case_id)

        write_run_output(args.output_path, meta, case_results, total, invocation_start)
        write_artifact(args.artifact_path, meta, case_results, args.output_path)

    write_run_output(args.output_path, meta, case_results, total, invocation_start)
    write_artifact(args.artifact_path, meta, case_results, args.output_path)
    summary = summarize_results(case_results, total, invocation_start)
    print("[Summary] completed={completed_case_count}/{selected_case_count}".format(**summary))
    print("[Summary] source_threat_count={source_threat_count}".format(**summary))
    print("[Summary] valid_threat_count={valid_threat_count}".format(**summary))
    print("[Summary] retrieval_threat_count={retrieval_threat_count}".format(**summary))
    print("[Summary] case_schema_valid_rate={:.4f}".format(summary["case_schema_valid_rate"]))
    print("[Summary] threat_schema_valid_rate={:.4f}".format(summary["threat_schema_valid_rate"]))
    if summary.get("parse_repaired_threat_count"):
        print("[Summary] parse_repaired_threat_count={}".format(summary["parse_repaired_threat_count"]))
    print("[Summary] total_case_duration_seconds={:.3f}".format(summary["total_case_duration_seconds"]))
    print("[Summary] average_case_duration_seconds={:.3f}".format(summary["average_case_duration_seconds"]))
    print(f"[Summary] output_path={args.output_path}")
    print(f"[Summary] artifact_path={args.artifact_path}")


if __name__ == "__main__":
    main()
