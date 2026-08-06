#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build Rec-EvoGraph-RAG KG variants for feedback weight alpha sensitivity.

This script creates multiple graph copies that share the same node text and
topology as the static Rec-EvoGraph-RAG KG. Only edge weights change.

For each edge with critic feedback:

    feedback_weight = clip(base_weight * exp(eta * clipped_signal), min_weight, max_weight)
    fused_weight = clip((1 - alpha) * base_weight + alpha * feedback_weight, min_weight, max_weight)

The current iter_01 graph is left untouched. By default the output folders are:

    0_Data/5_Knowledge_Base/recevograph_rag_evo_v0_2/alpha_0p0
    0_Data/5_Knowledge_Base/recevograph_rag_evo_v0_2/alpha_0p3
    0_Data/5_Knowledge_Base/recevograph_rag_evo_v0_2/alpha_0p5
    0_Data/5_Knowledge_Base/recevograph_rag_evo_v0_2/alpha_0p8
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SAAFG_SCRIPT_DIR = SCRIPT_DIR.parent
if str(SAAFG_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SAAFG_SCRIPT_DIR))

from update_edge_weights import (  # noqa: E402
    DEFAULT_FEEDBACK_JSONL,
    DEFAULT_KG_DIR,
    DEFAULT_OUTPUT_DIR as CURRENT_ITER_OUTPUT_DIR,
    KB_DIR,
    RELATION_SIGNAL_WEIGHTS,
    TEAM_SIGNAL_WEIGHTS,
    VERSION,
    aggregate_feedback,
    build_edge_index,
    build_networkx_graph,
    clamp,
    edge_key_from_edge,
    node_type_counts,
    now_utc,
    read_json,
    read_jsonl,
    relation_counts,
    safe_float,
    serializable_aggregate,
    validate_graph,
    write_json,
)


ALPHA_VERSION = f"{VERSION}-feedback-alpha"
GRAPH_NAME = "Rec-EvoGraph-RAG feedback-alpha security knowledge graph"
DEFAULT_OUTPUT_ROOT = KB_DIR / "recevograph_rag_evo_v0_2"
DEFAULT_ALPHAS = [0.0, 0.3, 0.5, 0.8]


def alpha_label(alpha: float) -> str:
    value = f"{alpha:.3f}".rstrip("0").rstrip(".")
    if "." not in value:
        value = f"{value}.0"
    return "alpha_" + value.replace("-", "m").replace(".", "p")


def validate_alpha_values(alphas: Sequence[float]) -> List[float]:
    result: List[float] = []
    seen: set[str] = set()
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"feedback_weight_alpha must be in [0.0, 1.0], got {alpha}")
        key = alpha_label(alpha)
        if key in seen:
            continue
        seen.add(key)
        result.append(float(alpha))
    if not result:
        raise ValueError("At least one feedback_weight_alpha value is required.")
    return result


def feedback_weight_from_signal(base_weight: float, raw_signal: float, args: argparse.Namespace) -> Tuple[float, float]:
    clipped_signal = clamp(raw_signal, -args.max_abs_signal, args.max_abs_signal)
    feedback_weight = clamp(
        base_weight * math.exp(args.eta * clipped_signal),
        args.min_weight,
        args.max_weight,
    )
    return feedback_weight, clipped_signal


def build_alpha_edges(
    edges: Sequence[Dict[str, Any]],
    aggregates: Dict[Tuple[str, str, str], Dict[str, Any]],
    alpha: float,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    updated_edges: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    increased_count = 0
    decreased_count = 0
    unchanged_count = 0
    no_feedback_count = 0
    alpha_abs_delta_sum = 0.0
    feedback_abs_delta_sum = 0.0

    for edge in edges:
        updated = dict(edge)
        key = edge_key_from_edge(edge)
        base_weight = safe_float(edge.get("base_weight"), safe_float(edge.get("weight"), 1.0))
        updated["base_weight"] = round(base_weight, 6)

        aggregate = aggregates.get(key)
        if aggregate:
            raw_signal = float(aggregate["raw_signal"])
            feedback_weight, clipped_signal = feedback_weight_from_signal(base_weight, raw_signal, args)
            fused_weight = clamp(
                (1.0 - alpha) * base_weight + alpha * feedback_weight,
                args.min_weight,
                args.max_weight,
            )
            rounded_fused = round(fused_weight, 6)
            rounded_base = round(base_weight, 6)
            feedback_delta = feedback_weight - base_weight
            alpha_delta = fused_weight - base_weight
            alpha_abs_delta_sum += abs(alpha_delta)
            feedback_abs_delta_sum += abs(feedback_delta)

            updated["weight"] = rounded_fused
            updated["feedback_alpha_update"] = {
                "version": ALPHA_VERSION,
                "updated_at_utc": now_utc(),
                "policy": "linear_fusion_between_base_and_feedback_weight",
                "feedback_weight_alpha": alpha,
                "eta": args.eta,
                "min_weight": args.min_weight,
                "max_weight": args.max_weight,
                "base_weight": rounded_base,
                "feedback_weight": round(feedback_weight, 6),
                "fused_weight": rounded_fused,
                "raw_signal": round(raw_signal, 8),
                "clipped_signal": round(clipped_signal, 8),
                **serializable_aggregate(aggregate),
            }

            delta = {
                "edge_id": edge.get("edge_id"),
                "source": key[0],
                "relation": key[1],
                "target": key[2],
                "base_weight": rounded_base,
                "feedback_weight": round(feedback_weight, 6),
                "fused_weight": rounded_fused,
                "feedback_delta": round(feedback_delta, 6),
                "alpha_delta": round(alpha_delta, 6),
                "raw_signal": round(raw_signal, 8),
                "clipped_signal": round(clipped_signal, 8),
                "event_count": int(aggregate["event_count"]),
                "positive_event_count": int(aggregate["positive_event_count"]),
                "negative_event_count": int(aggregate["negative_event_count"]),
                "teams": dict(sorted(aggregate["teams"].items())),
                "run_tags": dict(sorted(aggregate["run_tags"].items())),
                "outcomes": dict(sorted(aggregate["outcomes"].items())),
                "event_id_sample": [item for item in aggregate["event_id_sample"] if item],
            }
            deltas.append(delta)

            if rounded_fused > rounded_base:
                increased_count += 1
            elif rounded_fused < rounded_base:
                decreased_count += 1
            else:
                unchanged_count += 1
        else:
            updated["weight"] = round(base_weight, 6)
            no_feedback_count += 1

        updated_edges.append(updated)

    deltas.sort(
        key=lambda item: (
            -abs(item["alpha_delta"]),
            -abs(item["feedback_delta"]),
            -item["event_count"],
            item["edge_id"] or "",
        )
    )
    feedback_eligible_count = len(deltas)
    alpha_changed_count = increased_count + decreased_count
    stats = {
        "edge_count": len(edges),
        "feedback_eligible_edge_count": feedback_eligible_count,
        "updated_edge_count": alpha_changed_count,
        "increased_edge_count": increased_count,
        "decreased_edge_count": decreased_count,
        "unchanged_after_alpha_fusion_count": unchanged_count,
        "no_feedback_edge_count": no_feedback_count,
        "max_abs_feedback_delta": max((abs(item["feedback_delta"]) for item in deltas), default=0.0),
        "mean_abs_feedback_delta": (
            feedback_abs_delta_sum / feedback_eligible_count if feedback_eligible_count else 0.0
        ),
        "max_abs_alpha_delta": max((abs(item["alpha_delta"]) for item in deltas), default=0.0),
        "mean_abs_alpha_delta": alpha_abs_delta_sum / feedback_eligible_count if feedback_eligible_count else 0.0,
    }
    return updated_edges, deltas, stats


def build_metadata(
    base_metadata: Dict[str, Any],
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    feedback_stats: Dict[str, Any],
    update_stats: Dict[str, Any],
    validation: Dict[str, Any],
    alpha: float,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metadata = dict(base_metadata)
    metadata.update(
        {
            "graph_name": GRAPH_NAME,
            "version": ALPHA_VERSION,
            "generated_at_utc": now_utc(),
            "created_by": "feedback_weight_alpha_edge_weight_fusion",
            "base_graph_dir": str(args.kg_dir),
            "feedback_jsonl": str(args.feedback_jsonl),
            "output_dir": str(output_dir),
            "feedback_weight_alpha": alpha,
            "alpha_grid": list(args.alphas),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_type_counts": node_type_counts(nodes),
            "edge_relation_counts": relation_counts(edges),
            "feedback_stats": feedback_stats,
            "edge_weight_update_stats": update_stats,
            "edge_weight_update_policy": {
                "formula": (
                    "feedback_weight = clip(base_weight * exp(eta * clipped_signal), min_weight, max_weight); "
                    "fused_weight = clip((1 - feedback_weight_alpha) * base_weight + "
                    "feedback_weight_alpha * feedback_weight, min_weight, max_weight)"
                ),
                "feedback_weight_alpha": alpha,
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
                "Only edge weights are changed by split-filtered critic feedback and "
                "feedback_weight_alpha linear fusion."
            ),
        }
    )
    return metadata


def build_report(metadata: Dict[str, Any], deltas: Sequence[Dict[str, Any]]) -> str:
    feedback = metadata["feedback_stats"]
    update = metadata["edge_weight_update_stats"]
    policy = metadata["edge_weight_update_policy"]
    lines = [
        "# Rec-EvoGraph-RAG Feedback Weight Alpha KG Report",
        "",
        f"- graph_name: {metadata['graph_name']}",
        f"- version: {metadata['version']}",
        f"- generated_at_utc: {metadata['generated_at_utc']}",
        f"- feedback_weight_alpha: {metadata['feedback_weight_alpha']}",
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
        "## Alpha Fusion Policy",
        "",
        f"- formula: `{policy['formula']}`",
        f"- feedback_weight_alpha: {policy['feedback_weight_alpha']}",
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
        f"- feedback_eligible_edge_count: {update['feedback_eligible_edge_count']}",
        f"- updated_edge_count: {update['updated_edge_count']}",
        f"- increased_edge_count: {update['increased_edge_count']}",
        f"- decreased_edge_count: {update['decreased_edge_count']}",
        f"- unchanged_after_alpha_fusion_count: {update['unchanged_after_alpha_fusion_count']}",
        f"- no_feedback_edge_count: {update['no_feedback_edge_count']}",
        f"- max_abs_feedback_delta: {update['max_abs_feedback_delta']:.6f}",
        f"- mean_abs_feedback_delta: {update['mean_abs_feedback_delta']:.6f}",
        f"- max_abs_alpha_delta: {update['max_abs_alpha_delta']:.6f}",
        f"- mean_abs_alpha_delta: {update['mean_abs_alpha_delta']:.6f}",
        "",
        "## Top Absolute Alpha Deltas",
        "",
    ]
    for item in deltas[:30]:
        lines.append(
            "- {edge_id}: {relation} base={base_weight:.6f} feedback={feedback_weight:.6f} "
            "fused={fused_weight:.6f} alpha_delta={alpha_delta:.6f} events={event_count} "
            "source={source} target={target}".format(**item)
        )
    if not deltas:
        lines.append("- No edge had feedback.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Rec-EvoGraph-RAG feedback_weight_alpha KG variants v0.2."
    )
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def assert_output_dir_available(output_dir: Path, overwrite: bool) -> None:
    if output_dir.resolve() == CURRENT_ITER_OUTPUT_DIR.resolve():
        raise ValueError(f"Refusing to write over current iter_01 KG: {output_dir}")
    existing_outputs = [
        output_dir / "graph_nodes.json",
        output_dir / "graph_edges.json",
        output_dir / "graph_metadata.json",
        output_dir / "networkx_graph.pkl",
    ]
    if not overwrite and any(path.exists() for path in existing_outputs):
        raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")


def write_alpha_graph(
    output_dir: Path,
    nodes: Sequence[Dict[str, Any]],
    updated_edges: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    deltas: Sequence[Dict[str, Any]],
) -> None:
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


def main() -> None:
    args = parse_args()
    args.alphas = validate_alpha_values(args.alphas)

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

    for alpha in args.alphas:
        output_dir = args.output_root / alpha_label(alpha)
        assert_output_dir_available(output_dir, args.overwrite or args.dry_run)

        updated_edges, deltas, update_stats = build_alpha_edges(edges, aggregates, alpha, args)
        validation = validate_graph(nodes, updated_edges)
        metadata = build_metadata(
            base_metadata=base_metadata,
            nodes=nodes,
            edges=updated_edges,
            feedback_stats=feedback_stats,
            update_stats=update_stats,
            validation=validation,
            alpha=alpha,
            output_dir=output_dir,
            args=args,
        )

        if args.dry_run:
            print(
                "[DryRun] alpha={alpha} output_dir={output_dir} "
                "updated_edge_count={updated} max_abs_alpha_delta={max_delta:.6f}".format(
                    alpha=alpha,
                    output_dir=output_dir,
                    updated=update_stats["updated_edge_count"],
                    max_delta=update_stats["max_abs_alpha_delta"],
                )
            )
            continue

        write_alpha_graph(output_dir, nodes, updated_edges, metadata, deltas)
        print(
            "[Done] alpha={alpha} output_dir={output_dir} "
            "updated_edge_count={updated} increased={increased} decreased={decreased}".format(
                alpha=alpha,
                output_dir=output_dir,
                updated=update_stats["updated_edge_count"],
                increased=update_stats["increased_edge_count"],
                decreased=update_stats["decreased_edge_count"],
            )
        )


if __name__ == "__main__":
    main()
