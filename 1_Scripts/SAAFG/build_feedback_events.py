#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build critic feedback events for Rec-EvoGraph-RAG v0.2.

This script converts completed StaticGraphRAG-GraphAware Red/Blue runs and
their evaluator outputs into a JSONL feedback stream. The feedback stream is
used by update_edge_weights.py to evolve graph edge weights.

Design notes:
- Blue Team feedback is the primary signal because the Blue retrieval trace
  contains explicit Risk/AttackPattern/Technique -> Mitigation graph paths.
- Red Team feedback is kept as a weak auxiliary signal. Red retrieval evidence
  is case-level and less directly attributable to individual generated threats.
- The default split filter is "train" to avoid updating graph weights from test
  labels. Use --split-filter all only for diagnostic ablations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
EXPERIMENT_ROOT = SAAFG_ROOT / "6_Experiment_Result"
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"

DEFAULT_CASE_REGISTRY_PATH = (
    SAAFG_ROOT / "7_Benchmark_Package_v0_2" / "case_registry_test_1.json"
)
DEFAULT_KG_DIR = KB_DIR / "recevograph_rag"
DEFAULT_OUTPUT_DIR = KB_DIR / "recevograph_rag_feedback" / "v0_2"
DEFAULT_OUTPUT_JSONL = "feedback_events.jsonl"
DEFAULT_SUMMARY_JSON = "feedback_summary_staticgraphrag_graphaware_v0_2.json"

VERSION = "v0.2"
DEFAULT_FILE_METHOD = "staticgraphrag_graphaware"
DEFAULT_SOURCE_METHOD = "StaticGraphRAG-GraphAware"
METHOD_DIR_PREFIX = "ma_StaticGraphRAG_GraphAware"
DEFAULT_RUN_TAGS = ["qwen35plus", "deepseek-v32"]

RELATION_WEAK_SIGNAL_LIMIT = 8


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        records.append(value)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def compact_str(value: Any) -> str:
    return str(value or "").strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stable_hash(payload: Any, length: int = 12) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def normalize_split_filter(values: Sequence[str]) -> Optional[set[str]]:
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    if not normalized or "all" in normalized:
        return None
    return normalized


def split_allowed(split_value: Any, allowed_splits: Optional[set[str]]) -> bool:
    if allowed_splits is None:
        return True
    return compact_str(split_value).lower() in allowed_splits


def load_case_registry(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected cases list in {path}")
    registry: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        case_id = compact_str(case.get("use_case_id"))
        if case_id:
            registry[case_id] = case
    return registry


def load_graph_edges(kg_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    edge_payload = read_json(kg_dir / "graph_edges.json")
    edges = edge_payload.get("edges")
    if not isinstance(edges, list):
        raise ValueError(f"Expected edges list in {kg_dir / 'graph_edges.json'}")
    return {
        (
            compact_str(edge.get("source")),
            compact_str(edge.get("relation")),
            compact_str(edge.get("target")),
        ): edge
        for edge in edges
        if edge.get("source") and edge.get("relation") and edge.get("target")
    }


def build_pair_edge_index(
    edge_lookup: Dict[Tuple[str, str, str], Dict[str, Any]]
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (source, _relation, target), edge in edge_lookup.items():
        index[(source, target)].append(edge)
    return index


def experiment_paths(
    experiment_root: Path,
    method_dir_prefix: str,
    file_method: str,
    run_tag: str,
) -> Dict[str, Path]:
    root = experiment_root / f"{method_dir_prefix}_{run_tag}"
    return {
        "root": root,
        "red_run": root / "red_team" / f"saafg_redteam_{file_method}_v0_2_{run_tag}.json",
        "red_eval": root / "red_team" / f"saafg_redteam_task_a_eval_{file_method}_v0_2_{run_tag}.json",
        "red_trace": root / "red_team" / f"saafg_redteam_retrieval_trace_{file_method}_v0_2_{run_tag}.jsonl",
        "blue_run": root / "blue_team" / f"saafg_blueteam_{file_method}_v0_2_{run_tag}.json",
        "blue_eval": root / "blue_team" / f"saafg_blueteam_task_b_eval_{file_method}_v0_2_{run_tag}.json",
        "blue_trace": root / "blue_team" / f"saafg_blueteam_retrieval_trace_{file_method}_v0_2_{run_tag}.jsonl",
    }


def require_existing(paths: Dict[str, Path], keys: Iterable[str]) -> None:
    missing = [str(paths[key]) for key in keys if not paths[key].exists()]
    if missing:
        raise FileNotFoundError("Missing required experiment file(s):\n" + "\n".join(missing))


def case_report_lookup(eval_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reports = eval_payload.get("case_reports") or []
    if not isinstance(reports, list):
        return {}
    return {
        compact_str(report.get("use_case_id")): report
        for report in reports
        if compact_str(report.get("use_case_id"))
    }


def collect_blue_judgments(eval_payload: Dict[str, Any]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()

    def add_judgment(judgment: Dict[str, Any]) -> None:
        case_id = compact_str(judgment.get("use_case_id"))
        threat_id = compact_str(judgment.get("predicted_threat_id"))
        if not case_id or not threat_id:
            return
        dedupe_key = stable_hash(
            {
                "case_id": case_id,
                "threat_id": threat_id,
                "silver": judgment.get("silver_threat_id"),
                "overall": judgment.get("overall_defense_valid"),
                "reason": judgment.get("reason_code"),
                "raw": judgment.get("raw_output"),
            },
            length=20,
        )
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        lookup[(case_id, threat_id)].append(judgment)

    for judgment in eval_payload.get("defense_judgments") or []:
        if isinstance(judgment, dict):
            add_judgment(judgment)
    for case_report in eval_payload.get("case_reports") or []:
        for judgment in case_report.get("defense_judgments") or []:
            if isinstance(judgment, dict):
                add_judgment(judgment)
    return lookup


def summarize_blue_judgments(judgments: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not judgments:
        return {
            "has_judgment": False,
            "overall_defense_valid": False,
            "security_basic_flow_valid": False,
            "security_alternative_flow_valid": False,
            "parse_valid": False,
            "schema_valid": False,
            "reason_codes": [],
            "silver_threat_ids": [],
        }
    return {
        "has_judgment": True,
        "overall_defense_valid": any(j.get("overall_defense_valid") is True for j in judgments),
        "security_basic_flow_valid": any(j.get("security_basic_flow_valid") is True for j in judgments),
        "security_alternative_flow_valid": any(j.get("security_alternative_flow_valid") is True for j in judgments),
        "parse_valid": any(j.get("parse_valid") is True for j in judgments),
        "schema_valid": any(j.get("schema_valid") is True for j in judgments),
        "reason_codes": sorted({compact_str(j.get("reason_code")) for j in judgments if j.get("reason_code")}),
        "silver_threat_ids": sorted(
            {compact_str(j.get("silver_threat_id")) for j in judgments if j.get("silver_threat_id")}
        ),
        "judgment_count": len(judgments),
    }


def reward_from_blue_summary(
    summary: Dict[str, Any],
    positive_reward: float,
    negative_reward: float,
) -> Tuple[str, float]:
    if summary.get("overall_defense_valid"):
        return "positive", positive_reward
    partial = summary.get("security_basic_flow_valid") or summary.get("security_alternative_flow_valid")
    if partial:
        return "negative", negative_reward * 0.55
    return "negative", negative_reward


def edge_event_payload(edge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": compact_str(edge.get("source")),
        "relation": compact_str(edge.get("relation")),
        "target": compact_str(edge.get("target")),
        "source_name": edge.get("source_name"),
        "target_name": edge.get("target_name"),
        "weight_at_retrieval": safe_float(edge.get("weight"), 1.0),
        "source_ids": edge.get("source_ids") or [],
    }


def build_blue_events(
    run_tag: str,
    source_method: str,
    trace_records: Sequence[Dict[str, Any]],
    blue_eval_payload: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
    allowed_splits: Optional[set[str]],
    positive_reward: float,
    negative_reward: float,
    max_retrieved_rank: int,
    max_paths_per_item: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    judgment_lookup = collect_blue_judgments(blue_eval_payload)
    events: List[Dict[str, Any]] = []
    skipped_no_judgment = 0
    skipped_split = 0
    skipped_no_path = 0

    for trace_index, record in enumerate(trace_records):
        case_id = compact_str(record.get("use_case_id"))
        threat_id = compact_str(record.get("threat_id"))
        registry_case = registry.get(case_id, {})
        split = record.get("split") or registry_case.get("split")
        if not split_allowed(split, allowed_splits):
            skipped_split += 1
            continue

        judgments = judgment_lookup.get((case_id, threat_id), [])
        if not judgments:
            skipped_no_judgment += 1
            continue
        critic_summary = summarize_blue_judgments(judgments)
        outcome, reward = reward_from_blue_summary(critic_summary, positive_reward, negative_reward)

        for item in record.get("retrieved_knowledge") or []:
            rank = safe_int(item.get("rank"), default=0)
            if rank <= 0 or rank > max_retrieved_rank:
                continue
            graph_item = item.get("graph_item") or {}
            paths = graph_item.get("graph_paths") or []
            if not paths:
                skipped_no_path += 1
                continue
            for path_index, graph_path in enumerate(paths[:max_paths_per_item], start=1):
                raw_edges = [
                    edge
                    for edge in graph_path.get("edges") or []
                    if edge.get("source") and edge.get("target") and edge.get("relation")
                ]
                if not raw_edges:
                    skipped_no_path += 1
                    continue
                event_id = (
                    f"feedback::blue::{run_tag}::{case_id}::{threat_id}"
                    f"::r{rank:02d}::p{path_index:02d}"
                )
                event = {
                    "event_id": event_id,
                    "version": VERSION,
                    "team": "blue",
                    "source_method": source_method,
                    "run_tag": run_tag,
                    "use_case_id": case_id,
                    "dataset": record.get("dataset") or registry_case.get("dataset"),
                    "split": split,
                    "source_knowledge_id": record.get("source_knowledge_id")
                    or registry_case.get("source_knowledge_id"),
                    "threat_id": threat_id,
                    "anchor_steps": record.get("anchor_steps") or [],
                    "threat_name": record.get("threat_name"),
                    "outcome": outcome,
                    "reward": reward,
                    "critic": critic_summary,
                    "retrieval": {
                        "trace_index": trace_index,
                        "retrieved_rank": rank,
                        "item_score": safe_float(item.get("score"), 0.0),
                        "score_breakdown": item.get("score_breakdown") or {},
                        "path_rank": path_index,
                        "path_score": safe_float(graph_path.get("path_score"), 1.0),
                        "mitigation_id": (
                            (graph_item.get("mitigation") or {}).get("node_id")
                            or (item.get("metadata") or {}).get("id")
                        ),
                    },
                    "edges": [edge_event_payload(edge) for edge in raw_edges],
                    "nodes": {
                        "mitigation": graph_item.get("mitigation") or {},
                        "matched_sources": graph_item.get("matched_sources") or [],
                    },
                    "attribution": {
                        "source": "blue_retrieval_trace.graph_paths",
                        "confidence": 1.0,
                        "note": "Blue path-level feedback is directly attributable to retrieved graph edges.",
                    },
                }
                events.append(event)

    stats = {
        "blue_trace_record_count": len(trace_records),
        "blue_event_count": len(events),
        "skipped_split": skipped_split,
        "skipped_no_judgment": skipped_no_judgment,
        "skipped_no_path": skipped_no_path,
    }
    return events, stats


def red_case_outcome(eval_case: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    predicted_total = safe_int(eval_case.get("predicted_threat_total"), 0)
    valid_match_count = safe_int(eval_case.get("threat_validity_match_count"), 0)
    anchor_match_count = safe_int(eval_case.get("primary_anchor_match_count"), 0)
    semantic_pairs = eval_case.get("semantic_match_pairs") or []
    if valid_match_count > 0:
        outcome = "positive"
        reward = 0.35 + min(valid_match_count, 3) * 0.05
    elif predicted_total > 0:
        outcome = "negative"
        reward = -0.12
    else:
        outcome = "neutral"
        reward = 0.0
    critic = {
        "predicted_threat_total": predicted_total,
        "silver_threat_total": safe_int(eval_case.get("silver_threat_total"), 0),
        "primary_anchor_match_count": anchor_match_count,
        "threat_validity_match_count": valid_match_count,
        "threat_f1": safe_float(eval_case.get("threat_f1"), 0.0),
        "semantic_match_pairs": semantic_pairs,
    }
    return outcome, reward, critic


def node_id_list(values: Any) -> List[str]:
    result: List[str] = []
    for item in values or []:
        if isinstance(item, dict):
            node_id = compact_str(item.get("node_id"))
        else:
            node_id = compact_str(item)
        if node_id:
            result.append(node_id)
    return result


def infer_red_edges(
    graph_item: Dict[str, Any],
    pair_edge_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    focus = graph_item.get("focus") or {}
    focus_id = compact_str(focus.get("node_id"))
    if not focus_id:
        return []
    related_ids = (
        node_id_list(graph_item.get("related_risks"))
        + node_id_list(graph_item.get("related_attack_patterns"))
        + node_id_list(graph_item.get("related_techniques"))
        + node_id_list(graph_item.get("candidate_mitigations"))
    )
    edge_candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for related_id in related_ids:
        for source, target in ((focus_id, related_id), (related_id, focus_id)):
            for edge in pair_edge_index.get((source, target), []):
                key = (
                    compact_str(edge.get("source")),
                    compact_str(edge.get("relation")),
                    compact_str(edge.get("target")),
                )
                if key in seen:
                    continue
                seen.add(key)
                edge_candidates.append(edge)
                if len(edge_candidates) >= RELATION_WEAK_SIGNAL_LIMIT:
                    return [edge_event_payload(item) for item in edge_candidates]
    return [edge_event_payload(item) for item in edge_candidates]


def build_red_events(
    run_tag: str,
    source_method: str,
    trace_records: Sequence[Dict[str, Any]],
    red_eval_payload: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
    allowed_splits: Optional[set[str]],
    pair_edge_index: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_retrieved_rank: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eval_lookup = case_report_lookup(red_eval_payload)
    events: List[Dict[str, Any]] = []
    skipped_split = 0
    skipped_missing_eval = 0
    node_only_count = 0

    for trace_index, record in enumerate(trace_records):
        case_id = compact_str(record.get("use_case_id"))
        registry_case = registry.get(case_id, {})
        split = record.get("split") or registry_case.get("split")
        if not split_allowed(split, allowed_splits):
            skipped_split += 1
            continue

        eval_case = eval_lookup.get(case_id)
        if eval_case is None:
            skipped_missing_eval += 1
            continue
        outcome, reward, critic = red_case_outcome(eval_case)
        if outcome == "neutral":
            continue

        for item in record.get("retrieved_knowledge") or []:
            rank = safe_int(item.get("rank"), default=0)
            if rank <= 0 or rank > max_retrieved_rank:
                continue
            graph_item = item.get("graph_item") or {}
            focus = graph_item.get("focus") or {}
            focus_id = compact_str(focus.get("node_id") or (item.get("metadata") or {}).get("id"))
            inferred_edges = infer_red_edges(graph_item, pair_edge_index)
            if not inferred_edges:
                node_only_count += 1

            event_id = f"feedback::red::{run_tag}::{case_id}::r{rank:02d}::{stable_hash(focus_id)}"
            event = {
                "event_id": event_id,
                "version": VERSION,
                "team": "red",
                "source_method": source_method,
                "run_tag": run_tag,
                "use_case_id": case_id,
                "dataset": record.get("dataset") or registry_case.get("dataset"),
                "split": split,
                "source_knowledge_id": record.get("source_knowledge_id")
                or registry_case.get("source_knowledge_id"),
                "outcome": outcome,
                "reward": reward,
                "critic": critic,
                "retrieval": {
                    "trace_index": trace_index,
                    "retrieved_rank": rank,
                    "item_score": safe_float(item.get("score"), 0.0),
                    "score_breakdown": item.get("score_breakdown") or {},
                    "focus_node_id": focus_id,
                },
                "edges": inferred_edges,
                "nodes": {
                    "focus": focus,
                    "related_risks": graph_item.get("related_risks") or [],
                    "related_attack_patterns": graph_item.get("related_attack_patterns") or [],
                    "related_techniques": graph_item.get("related_techniques") or [],
                    "candidate_mitigations": graph_item.get("candidate_mitigations") or [],
                },
                "attribution": {
                    "source": "red_retrieval_trace.case_level_inferred_edges",
                    "confidence": 0.25 if inferred_edges else 0.0,
                    "note": (
                        "Red feedback is weak because Task A critic labels are case/threat-level, "
                        "not item-level; inferred edges should receive low update weight."
                    ),
                },
            }
            events.append(event)

    stats = {
        "red_trace_record_count": len(trace_records),
        "red_event_count": len(events),
        "skipped_split": skipped_split,
        "skipped_missing_eval": skipped_missing_eval,
        "node_only_event_count": node_only_count,
    }
    return events, stats


def summarize_events(
    events: Sequence[Dict[str, Any]],
    run_stats: Dict[str, Any],
    args: argparse.Namespace,
    output_jsonl: Path,
    summary_json: Path,
) -> Dict[str, Any]:
    by_team = Counter(event.get("team") for event in events)
    by_run_tag = Counter(event.get("run_tag") for event in events)
    by_outcome = Counter(event.get("outcome") for event in events)
    by_split = Counter(event.get("split") for event in events)
    by_relation: Counter[str] = Counter()
    edge_event_count = 0
    for event in events:
        edges = event.get("edges") or []
        if edges:
            edge_event_count += 1
        for edge in edges:
            by_relation[compact_str(edge.get("relation"))] += 1

    return {
        "version": VERSION,
        "generated_at_utc": now_utc(),
        "source_method": args.source_method,
        "file_method": args.file_method,
        "run_tags": args.run_tags,
        "split_filter": args.split_filter,
        "experiment_root": str(args.experiment_root),
        "kg_dir": str(args.kg_dir),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "event_count": len(events),
        "edge_attributed_event_count": edge_event_count,
        "node_only_event_count": len(events) - edge_event_count,
        "by_team": dict(sorted(by_team.items())),
        "by_run_tag": dict(sorted(by_run_tag.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_relation": dict(sorted(by_relation.items())),
        "run_stats": run_stats,
        "policy": {
            "default_split_filter": "train",
            "blue_feedback": "Primary path-level signal from defense critic judgments.",
            "red_feedback": "Weak case-level signal with low attribution confidence.",
            "no_silver_content_ingestion": (
                "Feedback events store critic outcomes and graph edge identifiers; they do not add "
                "SAAFG silver threat/defense text into graph nodes."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Rec-EvoGraph-RAG feedback events v0.2.")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--method-dir-prefix", default=METHOD_DIR_PREFIX)
    parser.add_argument(
        "--file-method",
        default=DEFAULT_FILE_METHOD,
        help="Filename method token, e.g. staticgraphrag_graphaware or recevographrag.",
    )
    parser.add_argument(
        "--source-method",
        default=DEFAULT_SOURCE_METHOD,
        help="Method label stored in feedback events and summary metadata.",
    )
    parser.add_argument("--run-tags", nargs="+", default=DEFAULT_RUN_TAGS)
    parser.add_argument("--case-registry-path", type=Path, default=DEFAULT_CASE_REGISTRY_PATH)
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=Path(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--summary-json", type=Path, default=Path(DEFAULT_SUMMARY_JSON))
    parser.add_argument(
        "--split-filter",
        nargs="+",
        default=["train"],
        help="Split(s) used for feedback. Use 'all' only for diagnostic ablations.",
    )
    parser.add_argument("--positive-reward", type=float, default=1.0)
    parser.add_argument("--negative-reward", type=float, default=-0.45)
    parser.add_argument("--max-retrieved-rank", type=int, default=3)
    parser.add_argument("--max-paths-per-item", type=int, default=5)
    parser.add_argument("--no-red-events", action="store_true")
    parser.add_argument("--no-blue-events", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = args.output_jsonl
    if not output_jsonl.is_absolute():
        output_jsonl = args.output_dir / output_jsonl
    summary_json = args.summary_json
    if not summary_json.is_absolute():
        summary_json = args.output_dir / summary_json

    registry = load_case_registry(args.case_registry_path)
    edge_lookup = load_graph_edges(args.kg_dir)
    pair_edge_index = build_pair_edge_index(edge_lookup)
    allowed_splits = normalize_split_filter(args.split_filter)

    all_events: List[Dict[str, Any]] = []
    run_stats: Dict[str, Any] = {}

    for run_tag in args.run_tags:
        paths = experiment_paths(
            args.experiment_root,
            args.method_dir_prefix,
            args.file_method,
            run_tag,
        )
        required_keys = []
        if not args.no_red_events:
            required_keys.extend(["red_eval", "red_trace"])
        if not args.no_blue_events:
            required_keys.extend(["blue_eval", "blue_trace"])
        require_existing(paths, required_keys)

        tag_stats: Dict[str, Any] = {"experiment_dir": str(paths["root"])}
        if not args.no_blue_events:
            blue_eval_payload = read_json(paths["blue_eval"])
            blue_trace_records = read_jsonl(paths["blue_trace"])
            blue_events, blue_stats = build_blue_events(
                run_tag=run_tag,
                source_method=args.source_method,
                trace_records=blue_trace_records,
                blue_eval_payload=blue_eval_payload,
                registry=registry,
                allowed_splits=allowed_splits,
                positive_reward=args.positive_reward,
                negative_reward=args.negative_reward,
                max_retrieved_rank=args.max_retrieved_rank,
                max_paths_per_item=args.max_paths_per_item,
            )
            all_events.extend(blue_events)
            tag_stats["blue"] = blue_stats

        if not args.no_red_events:
            red_eval_payload = read_json(paths["red_eval"])
            red_trace_records = read_jsonl(paths["red_trace"])
            red_events, red_stats = build_red_events(
                run_tag=run_tag,
                source_method=args.source_method,
                trace_records=red_trace_records,
                red_eval_payload=red_eval_payload,
                registry=registry,
                allowed_splits=allowed_splits,
                pair_edge_index=pair_edge_index,
                max_retrieved_rank=args.max_retrieved_rank,
            )
            all_events.extend(red_events)
            tag_stats["red"] = red_stats

        run_stats[run_tag] = tag_stats

    all_events.sort(
        key=lambda event: (
            str(event.get("split")),
            str(event.get("team")),
            str(event.get("run_tag")),
            str(event.get("use_case_id")),
            str(event.get("threat_id")),
            str(event.get("event_id")),
        )
    )
    summary = summarize_events(all_events, run_stats, args, output_jsonl, summary_json)

    write_jsonl(output_jsonl, all_events)
    write_json(summary_json, summary)

    print(f"[Done] feedback_events={len(all_events)}")
    print(f"[Done] edge_attributed_events={summary['edge_attributed_event_count']}")
    print(f"[Done] output_jsonl={output_jsonl}")
    print(f"[Done] summary_json={summary_json}")
    print(f"[Summary] by_team={summary['by_team']}")
    print(f"[Summary] by_outcome={summary['by_outcome']}")
    print(f"[Summary] by_split={summary['by_split']}")


if __name__ == "__main__":
    main()
