#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Update Rec-EvoGraph-RAG edge weights from critic feedback events v0.2.

The script keeps graph topology and node text fixed. It only changes edge
weights and records the update evidence for each changed edge. This makes the
second-stage Rec-EvoGraph experiment attributable to critic-driven graph
evolution rather than to a new knowledge base.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "0_Data" / "5_Knowledge_Base"

DEFAULT_KG_DIR = KB_DIR / "recevograph_rag"
DEFAULT_FEEDBACK_JSONL = (
    KB_DIR
    / "recevograph_rag_feedback"
    / "v0_2"
    / "feedback_events.jsonl"
)
DEFAULT_OUTPUT_DIR = KB_DIR / "recevograph_rag_evo_v0_2" / "iter_01"

VERSION = "v0.2"
GRAPH_NAME = "Rec-EvoGraph-RAG critic-evolved security knowledge graph"

RELATION_SIGNAL_WEIGHTS = {
    "mitigated_by": 1.0,
    "exploits": 0.65,
    "belongs_to": 0.55,
    "implements_or_examples": 0.55,
    "supported_by": 0.15,
}

TEAM_SIGNAL_WEIGHTS = {
    "blue": 1.0,
    "red": 0.25,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Feedback JSONL not found: {path}")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        records.append(payload)
    return records


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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_split_filter(values: Sequence[str]) -> Optional[set[str]]:
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    if not normalized or "all" in normalized:
        return None
    return normalized


def split_allowed(split_value: Any, allowed_splits: Optional[set[str]]) -> bool:
    if allowed_splits is None:
        return True
    return compact_str(split_value).lower() in allowed_splits


def edge_key_from_edge(edge: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        compact_str(edge.get("source")),
        compact_str(edge.get("relation")),
        compact_str(edge.get("target")),
    )


def build_edge_index(edges: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for edge in edges:
        key = edge_key_from_edge(edge)
        if all(key):
            index[key] = edge
    return index


def event_signal_for_edge(
    event: Dict[str, Any],
    event_edge: Dict[str, Any],
    args: argparse.Namespace,
) -> float:
    reward = safe_float(event.get("reward"), 0.0)
    if reward == 0.0:
        return 0.0

    team = compact_str(event.get("team")).lower()
    relation = compact_str(event_edge.get("relation"))
    retrieval = event.get("retrieval") or {}
    attribution = event.get("attribution") or {}

    rank = max(safe_int(retrieval.get("retrieved_rank"), 1), 1)
    path_rank = max(safe_int(retrieval.get("path_rank"), 1), 1)
    path_score = clamp(safe_float(retrieval.get("path_score"), 1.0), 0.0, 1.0)
    confidence = clamp(safe_float(attribution.get("confidence"), 1.0), 0.0, 1.0)

    rank_decay = 1.0 / math.sqrt(rank)
    path_decay = 1.0 / path_rank
    path_factor = 0.5 + 0.5 * path_score
    relation_weight = RELATION_SIGNAL_WEIGHTS.get(relation, 0.35)
    team_weight = TEAM_SIGNAL_WEIGHTS.get(team, 0.25)

    signal = reward * confidence * rank_decay * path_decay * path_factor * relation_weight * team_weight
    if signal < 0:
        signal *= args.negative_scale
    return signal


def aggregate_feedback(
    events: Sequence[Dict[str, Any]],
    edge_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[str, Any]]:
    allowed_splits = normalize_split_filter(args.split_filter)
    aggregates: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "raw_signal": 0.0,
            "event_count": 0,
            "positive_event_count": 0,
            "negative_event_count": 0,
            "teams": Counter(),
            "run_tags": Counter(),
            "outcomes": Counter(),
            "event_id_sample": [],
        }
    )
    missing_edges: Counter[Tuple[str, str, str]] = Counter()
    skipped_split = 0
    skipped_no_edges = 0
    skipped_low_confidence = 0

    for event in events:
        if not split_allowed(event.get("split"), allowed_splits):
            skipped_split += 1
            continue
        attribution = event.get("attribution") or {}
        confidence = clamp(safe_float(attribution.get("confidence"), 1.0), 0.0, 1.0)
        if confidence < args.min_confidence:
            skipped_low_confidence += 1
            continue
        edges = event.get("edges") or []
        if not edges:
            skipped_no_edges += 1
            continue

        for event_edge in edges:
            key = edge_key_from_edge(event_edge)
            if key not in edge_index:
                missing_edges[key] += 1
                continue
            signal = event_signal_for_edge(event, event_edge, args)
            if signal == 0.0:
                continue
            aggregate = aggregates[key]
            aggregate["raw_signal"] += signal
            aggregate["event_count"] += 1
            if signal > 0:
                aggregate["positive_event_count"] += 1
            elif signal < 0:
                aggregate["negative_event_count"] += 1
            aggregate["teams"][compact_str(event.get("team"))] += 1
            aggregate["run_tags"][compact_str(event.get("run_tag"))] += 1
            aggregate["outcomes"][compact_str(event.get("outcome"))] += 1
            if len(aggregate["event_id_sample"]) < 8:
                aggregate["event_id_sample"].append(event.get("event_id"))

    stats = {
        "input_event_count": len(events),
        "used_edge_count": len(aggregates),
        "skipped_split": skipped_split,
        "skipped_no_edges": skipped_no_edges,
        "skipped_low_confidence": skipped_low_confidence,
        "missing_edge_key_count": sum(missing_edges.values()),
        "missing_edge_key_sample": [
            {"source": key[0], "relation": key[1], "target": key[2], "count": count}
            for key, count in missing_edges.most_common(20)
        ],
    }
    return aggregates, stats


def serializable_aggregate(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "raw_signal": round(float(value["raw_signal"]), 8),
        "event_count": int(value["event_count"]),
        "positive_event_count": int(value["positive_event_count"]),
        "negative_event_count": int(value["negative_event_count"]),
        "teams": dict(sorted(value["teams"].items())),
        "run_tags": dict(sorted(value["run_tags"].items())),
        "outcomes": dict(sorted(value["outcomes"].items())),
        "event_id_sample": [item for item in value["event_id_sample"] if item],
    }


def update_edges(
    edges: Sequence[Dict[str, Any]],
    aggregates: Dict[Tuple[str, str, str], Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    updated_edges: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    weight_buckets = Counter()

    for edge in edges:
        updated = dict(edge)
        key = edge_key_from_edge(edge)
        old_weight = safe_float(edge.get("weight"), safe_float(edge.get("base_weight"), 1.0))
        base_weight = safe_float(edge.get("base_weight"), old_weight)
        updated["base_weight"] = base_weight

        aggregate = aggregates.get(key)
        if aggregate:
            raw_signal = float(aggregate["raw_signal"])
            clipped_signal = clamp(raw_signal, -args.max_abs_signal, args.max_abs_signal)
            multiplier = math.exp(args.eta * clipped_signal)
            new_weight = clamp(old_weight * multiplier, args.min_weight, args.max_weight)
            updated["weight"] = round(new_weight, 6)
            updated["evo_update"] = {
                "version": VERSION,
                "updated_at_utc": now_utc(),
                "policy": "multiplicative_exp_edge_weight_update",
                "eta": args.eta,
                "min_weight": args.min_weight,
                "max_weight": args.max_weight,
                "old_weight": round(old_weight, 6),
                "new_weight": round(new_weight, 6),
                "raw_signal": round(raw_signal, 8),
                "clipped_signal": round(clipped_signal, 8),
                **serializable_aggregate(aggregate),
            }
            delta = {
                "edge_id": edge.get("edge_id"),
                "source": key[0],
                "relation": key[1],
                "target": key[2],
                "old_weight": round(old_weight, 6),
                "new_weight": round(new_weight, 6),
                "delta": round(new_weight - old_weight, 6),
                "raw_signal": round(raw_signal, 8),
                "event_count": int(aggregate["event_count"]),
                "positive_event_count": int(aggregate["positive_event_count"]),
                "negative_event_count": int(aggregate["negative_event_count"]),
                "teams": dict(sorted(aggregate["teams"].items())),
                "run_tags": dict(sorted(aggregate["run_tags"].items())),
                "outcomes": dict(sorted(aggregate["outcomes"].items())),
                "event_id_sample": [item for item in aggregate["event_id_sample"] if item],
            }
            deltas.append(delta)
            if new_weight > old_weight:
                weight_buckets["increased"] += 1
            elif new_weight < old_weight:
                weight_buckets["decreased"] += 1
            else:
                weight_buckets["unchanged_after_clamp_or_rounding"] += 1
        else:
            updated["weight"] = round(old_weight, 6)
            weight_buckets["no_feedback"] += 1

        updated_edges.append(updated)

    deltas.sort(key=lambda item: (-abs(item["delta"]), -item["event_count"], item["edge_id"] or ""))
    stats = {
        "edge_count": len(edges),
        "updated_edge_count": len(deltas),
        "increased_edge_count": weight_buckets["increased"],
        "decreased_edge_count": weight_buckets["decreased"],
        "unchanged_after_clamp_or_rounding_count": weight_buckets["unchanged_after_clamp_or_rounding"],
        "no_feedback_edge_count": weight_buckets["no_feedback"],
        "max_abs_delta": max((abs(item["delta"]) for item in deltas), default=0.0),
        "mean_abs_delta": (
            sum(abs(item["delta"]) for item in deltas) / len(deltas) if deltas else 0.0
        ),
    }
    return updated_edges, deltas, stats


def validate_graph(nodes: Sequence[Dict[str, Any]], edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    node_ids = {node.get("node_id") for node in nodes}
    missing_endpoints = []
    for edge in edges:
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            missing_endpoints.append(
                {
                    "edge_id": edge.get("edge_id"),
                    "source": edge.get("source"),
                    "relation": edge.get("relation"),
                    "target": edge.get("target"),
                }
            )
    return {
        "missing_edge_endpoint_count": len(missing_endpoints),
        "missing_edge_endpoints_sample": missing_endpoints[:50],
    }


def build_networkx_graph(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(name=GRAPH_NAME, version=VERSION)
    for node in nodes:
        node_id = node["node_id"]
        attrs = {key: value for key, value in node.items() if key != "node_id"}
        graph.add_node(node_id, **attrs)
    for edge in edges:
        attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
        graph.add_edge(edge["source"], edge["target"], key=edge.get("edge_id"), **attrs)
    return graph


def relation_counts(edges: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(compact_str(edge.get("relation")) for edge in edges)
    return dict(sorted(counts.items()))


def node_type_counts(nodes: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(compact_str(node.get("node_type")) for node in nodes)
    return dict(sorted(counts.items()))


def build_metadata(
    base_metadata: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    feedback_stats: Dict[str, Any],
    update_stats: Dict[str, Any],
    validation: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metadata = dict(base_metadata)
    metadata.update(
        {
            "graph_name": GRAPH_NAME,
            "version": VERSION,
            "generated_at_utc": now_utc(),
            "created_by": "critic_feedback_edge_weight_update",
            "base_graph_dir": str(args.kg_dir),
            "feedback_jsonl": str(args.feedback_jsonl),
            "output_dir": str(args.output_dir),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_type_counts": node_type_counts(nodes),
            "edge_relation_counts": relation_counts(edges),
            "feedback_stats": feedback_stats,
            "edge_weight_update_stats": update_stats,
            "edge_weight_update_policy": {
                "formula": "new_weight = clip(old_weight * exp(eta * clipped_signal), min_weight, max_weight)",
                "eta": args.eta,
                "min_weight": args.min_weight,
                "max_weight": args.max_weight,
                "max_abs_signal": args.max_abs_signal,
                "negative_scale": args.negative_scale,
                "split_filter": args.split_filter,
                "min_confidence": args.min_confidence,
                "team_signal_weights": TEAM_SIGNAL_WEIGHTS,
                "relation_signal_weights": RELATION_SIGNAL_WEIGHTS,
            },
            "validation": validation,
            "leakage_policy": (
                "Node text and graph topology are unchanged from the static OWASP/ATLAS graph. "
                "Only edge weights are updated from split-filtered critic feedback events."
            ),
        }
    )
    return metadata


def build_report(metadata: Dict[str, Any], deltas: Sequence[Dict[str, Any]]) -> str:
    feedback = metadata["feedback_stats"]
    update = metadata["edge_weight_update_stats"]
    policy = metadata["edge_weight_update_policy"]
    lines = [
        "# Rec-EvoGraph-RAG Edge Weight Update Report",
        "",
        f"- graph_name: {metadata['graph_name']}",
        f"- version: {metadata['version']}",
        f"- generated_at_utc: {metadata['generated_at_utc']}",
        f"- base_graph_dir: `{metadata['base_graph_dir']}`",
        f"- feedback_jsonl: `{metadata['feedback_jsonl']}`",
        f"- output_dir: `{metadata['output_dir']}`",
        "",
        "## Leakage Policy",
        "",
        metadata["leakage_policy"],
        "",
        "## Feedback Usage",
        "",
        f"- input_event_count: {feedback['input_event_count']}",
        f"- used_edge_count: {feedback['used_edge_count']}",
        f"- skipped_split: {feedback['skipped_split']}",
        f"- skipped_no_edges: {feedback['skipped_no_edges']}",
        f"- skipped_low_confidence: {feedback['skipped_low_confidence']}",
        f"- missing_edge_key_count: {feedback['missing_edge_key_count']}",
        "",
        "## Update Policy",
        "",
        f"- formula: `{policy['formula']}`",
        f"- eta: {policy['eta']}",
        f"- min_weight: {policy['min_weight']}",
        f"- max_weight: {policy['max_weight']}",
        f"- max_abs_signal: {policy['max_abs_signal']}",
        f"- negative_scale: {policy['negative_scale']}",
        f"- split_filter: {', '.join(map(str, policy['split_filter']))}",
        "",
        "## Edge Weight Changes",
        "",
        f"- edge_count: {update['edge_count']}",
        f"- updated_edge_count: {update['updated_edge_count']}",
        f"- increased_edge_count: {update['increased_edge_count']}",
        f"- decreased_edge_count: {update['decreased_edge_count']}",
        f"- unchanged_after_clamp_or_rounding_count: {update['unchanged_after_clamp_or_rounding_count']}",
        f"- no_feedback_edge_count: {update['no_feedback_edge_count']}",
        f"- max_abs_delta: {update['max_abs_delta']:.6f}",
        f"- mean_abs_delta: {update['mean_abs_delta']:.6f}",
        "",
        "## Top Absolute Deltas",
        "",
    ]
    for item in deltas[:30]:
        lines.append(
            "- {edge_id}: {relation} {old_weight:.6f}->{new_weight:.6f} "
            "delta={delta:.6f} events={event_count} source={source} target={target}".format(**item)
        )
    if not deltas:
        lines.append("- No edge was updated.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Rec-EvoGraph-RAG edge weights v0.2.")
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-filter",
        nargs="+",
        default=["train"],
        help="Feedback split(s) used for updates. Use 'all' only for diagnostic ablations.",
    )
    parser.add_argument("--eta", type=float, default=0.08)
    parser.add_argument("--min-weight", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=3.0)
    parser.add_argument("--max-abs-signal", type=float, default=5.0)
    parser.add_argument("--negative-scale", type=float, default=0.7)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes_payload = read_json(args.kg_dir / "graph_nodes.json")
    edges_payload = read_json(args.kg_dir / "graph_edges.json")
    base_metadata = read_json(args.kg_dir / "graph_metadata.json")
    events = read_jsonl(args.feedback_jsonl)

    nodes = list(nodes_payload.get("nodes") or [])
    edges = list(edges_payload.get("edges") or [])
    if not nodes:
        raise ValueError(f"No nodes found in {args.kg_dir / 'graph_nodes.json'}")
    if not edges:
        raise ValueError(f"No edges found in {args.kg_dir / 'graph_edges.json'}")

    edge_index = build_edge_index(edges)
    aggregates, feedback_stats = aggregate_feedback(events, edge_index, args)
    updated_edges, deltas, update_stats = update_edges(edges, aggregates, args)
    validation = validate_graph(nodes, updated_edges)
    metadata = build_metadata(
        base_metadata=base_metadata,
        nodes=nodes,
        edges=updated_edges,
        feedback_stats=feedback_stats,
        update_stats=update_stats,
        validation=validation,
        args=args,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "graph_nodes.json", {"meta": metadata, "nodes": nodes})
    write_json(output_dir / "graph_edges.json", {"meta": metadata, "edges": updated_edges})
    write_json(output_dir / "graph_metadata.json", metadata)
    write_json(output_dir / "edge_weight_deltas.json", {"meta": metadata, "deltas": deltas})
    (output_dir / "edge_weight_report.md").write_text(
        build_report(metadata, deltas),
        encoding="utf-8",
    )

    graph = build_networkx_graph(nodes, updated_edges)
    with (output_dir / "networkx_graph.pkl").open("wb") as f:
        pickle.dump(graph, f)

    print(f"[Done] updated_edge_count={update_stats['updated_edge_count']}")
    print(f"[Done] increased_edge_count={update_stats['increased_edge_count']}")
    print(f"[Done] decreased_edge_count={update_stats['decreased_edge_count']}")
    print(f"[Done] output_dir={output_dir}")
    print(f"[Done] report={output_dir / 'edge_weight_report.md'}")


if __name__ == "__main__":
    main()
