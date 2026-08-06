#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rec-EvoGraph-RAG retrieval utilities with critic-evolved edge weights.

This module intentionally leaves evographrag_retrieval.py unchanged.
It subclasses the static retriever and makes edge weights more visible in
Red/Blue reranking so that the second-stage experiment tests critic-updated
graph structure rather than only a different output directory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from evographrag_retrieval import (
    BLUE_SOURCE_NODE_TYPES,
    RED_SEARCH_NODE_TYPES,
    RecEvoGraphRetriever as StaticRecEvoGraphRetriever,
    build_flow_query,
    build_threat_query,
    edge_weight,
    get_anchor_step_text,
    load_functional_cases,
    load_threat_cases,
    now_utc,
    safe_float,
    select_threat,
    token_overlap_score,
    truncate,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
SAAFG_ROOT = BASE_DIR / "0_Data" / "6_SAAFG"
DEFAULT_EVO_KG_DIR = (
    BASE_DIR / "0_Data" / "5_Knowledge_Base" / "recevograph_rag_evo_v0_2" / "iter_01"
)
DEFAULT_FLOW_PATH = (
    SAAFG_ROOT / "1_Input_Functional_Flows" / "functional_use_case_flows.json"
)
DEFAULT_THREAT_PATH = (
    SAAFG_ROOT / "2_RedTeam_Threat_Records" / "threat_records.json"
)

VERSION = "v0.2-evo"
EDGE_WEIGHT_MIN = 0.35
EDGE_WEIGHT_MAX = 3.0
STATIC_WEIGHT_NORM = 1.0 / EDGE_WEIGHT_MAX


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized_edge_weight(weight: float) -> float:
    return clamp(weight, EDGE_WEIGHT_MIN, EDGE_WEIGHT_MAX) / EDGE_WEIGHT_MAX


class RecEvoGraphRetriever(StaticRecEvoGraphRetriever):
    """Graph retriever that uses critic-evolved edge weights in reranking."""

    def __init__(self, kg_dir: Path = DEFAULT_EVO_KG_DIR) -> None:
        super().__init__(kg_dir=kg_dir)

    def edge_weight_score(self, edge: Dict[str, Any]) -> float:
        return normalized_edge_weight(edge_weight(edge))

    def path_weight_score(self, path_edges: Sequence[Dict[str, Any]]) -> float:
        if not path_edges:
            return STATIC_WEIGHT_NORM
        scores = [self.edge_weight_score(edge) for edge in path_edges]
        return sum(scores) / len(scores)

    def local_edge_weight_score(self, node_id: str) -> float:
        weights: List[float] = []
        for relation_edges in self.out_edges.get(node_id, {}).values():
            weights.extend(self.edge_weight_score(edge) for edge in relation_edges)
        for relation_edges in self.in_edges.get(node_id, {}).values():
            weights.extend(self.edge_weight_score(edge) for edge in relation_edges)
        if not weights:
            return STATIC_WEIGHT_NORM
        weights.sort(reverse=True)
        return sum(weights[:8]) / min(len(weights), 8)

    def graph_support_score(self, node_id: str) -> float:
        base_score = super().graph_support_score(node_id)
        weight_score = self.local_edge_weight_score(node_id)
        return clamp(0.65 * base_score + 0.35 * weight_score, 0.0, 1.0)

    def expand_red_candidates(self, hits: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        candidates: Dict[str, float] = {}
        for hit in hits:
            node_id = hit["node_id"]
            similarity = hit["similarity"]
            candidates[node_id] = max(candidates.get(node_id, 0.0), similarity)
            for related_id in (
                self.related_risks(node_id, limit=4)
                + self.related_attack_patterns(node_id, limit=4)
                + self.related_techniques(node_id, limit=4)
            ):
                related_type = self.nodes[related_id].get("node_type")
                if related_type not in RED_SEARCH_NODE_TYPES:
                    continue
                weight_lift = 0.80 + 0.18 * self.local_edge_weight_score(related_id)
                candidates[related_id] = max(
                    candidates.get(related_id, 0.0),
                    similarity * weight_lift,
                )
        return candidates

    def retrieve_red(
        self,
        flow_case: Dict[str, Any],
        top_k: int = 6,
        candidate_top_n: int = 40,
        max_per_risk: int = 2,
    ) -> Dict[str, Any]:
        query = build_flow_query(flow_case)
        hits = self.vector_search(query, RED_SEARCH_NODE_TYPES, top_n=candidate_top_n)
        candidates = self.expand_red_candidates(hits)

        scored: List[Dict[str, Any]] = []
        for node_id, similarity in candidates.items():
            node = self.nodes[node_id]
            node_text = self.node_search_text(node)
            context_fit = token_overlap_score(query, node_text)
            graph_support = self.graph_support_score(node_id)
            evo_edge_weight_score = self.local_edge_weight_score(node_id)
            type_bonus = {"AttackPattern": 0.08, "Risk": 0.06, "Technique": 0.04}.get(
                node.get("node_type"), 0.0
            )
            score = (
                0.52 * similarity
                + 0.16 * context_fit
                + 0.18 * graph_support
                + 0.06 * evo_edge_weight_score
                + type_bonus
                + self.source_priority(node_id)
            )
            scored.append(
                {
                    "node_id": node_id,
                    "score": score,
                    "score_breakdown": {
                        "vector_similarity": round(similarity, 6),
                        "context_fit": round(context_fit, 6),
                        "graph_support": round(graph_support, 6),
                        "evo_edge_weight_score": round(evo_edge_weight_score, 6),
                        "type_bonus": round(type_bonus, 6),
                        "source_priority": round(self.source_priority(node_id), 6),
                    },
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["node_id"]))
        selected: List[Dict[str, Any]] = []
        per_risk_count: Dict[str, int] = defaultdict(int)
        for item in scored:
            risk_ids = self.related_risks(item["node_id"], limit=1)
            risk_key = risk_ids[0] if risk_ids else item["node_id"]
            if per_risk_count[risk_key] >= max_per_risk:
                continue
            per_risk_count[risk_key] += 1
            selected.append(item)
            if len(selected) >= top_k:
                break

        return {
            "meta": {
                "retrieval_role": "red_team",
                "retriever_version": VERSION,
                "generated_at_utc": now_utc(),
                "kg_dir": str(self.kg_dir),
                "use_case_id": flow_case.get("use_case_id"),
                "top_k": top_k,
                "candidate_top_n": candidate_top_n,
                "rerank_policy": (
                    "TF-IDF candidate retrieval plus graph expansion, then deterministic reranking "
                    "over vector similarity, context fit, graph support, critic-evolved edge weight, "
                    "node type, and source priority."
                ),
            },
            "query": truncate(query, 1600),
            "items": [self.build_red_item(item, rank + 1) for rank, item in enumerate(selected)],
        }

    def retrieve_blue(
        self,
        flow_case: Dict[str, Any],
        threat: Dict[str, Any],
        top_k: int = 6,
        candidate_top_n: int = 40,
        fallback_top_n: int = 12,
    ) -> Dict[str, Any]:
        query = build_threat_query(flow_case, threat)
        source_candidates = self.blue_source_candidates(flow_case, threat, candidate_top_n)

        mitigation_candidates: Dict[str, Dict[str, Any]] = {}
        for source_id, source_similarity in source_candidates.items():
            for path in self.mitigation_paths_from_source(source_id):
                mitigation_id = path["mitigation_id"]
                path_weight = self.path_weight_score(path["path_edges"])
                weighted_path_score = clamp(
                    path["path_score"] * (0.95 + 0.15 * path_weight),
                    0.0,
                    1.0,
                )
                record = mitigation_candidates.setdefault(
                    mitigation_id,
                    {
                        "mitigation_id": mitigation_id,
                        "source_similarity": 0.0,
                        "path_score": 0.0,
                        "max_edge_weight": 1.0,
                        "max_path_weight_score": STATIC_WEIGHT_NORM,
                        "matched_source_ids": [],
                        "paths": [],
                    },
                )
                record["source_similarity"] = max(record["source_similarity"], source_similarity)
                record["path_score"] = max(record["path_score"], weighted_path_score)
                record["max_edge_weight"] = max(
                    record["max_edge_weight"],
                    max(edge_weight(edge) for edge in path["path_edges"]),
                )
                record["max_path_weight_score"] = max(record["max_path_weight_score"], path_weight)
                if source_id not in record["matched_source_ids"]:
                    record["matched_source_ids"].append(source_id)
                if len(record["paths"]) < 5:
                    path_with_evo = {
                        **path,
                        "path_score": weighted_path_score,
                        "base_path_score": path["path_score"],
                        "evo_path_weight_score": path_weight,
                    }
                    record["paths"].append(path_with_evo)

        fallback_hits = self.vector_search(query, {"Mitigation"}, top_n=fallback_top_n)
        for hit in fallback_hits:
            mitigation_id = hit["node_id"]
            record = mitigation_candidates.setdefault(
                mitigation_id,
                {
                    "mitigation_id": mitigation_id,
                    "source_similarity": 0.0,
                    "path_score": 0.0,
                    "max_edge_weight": 1.0,
                    "max_path_weight_score": STATIC_WEIGHT_NORM,
                    "matched_source_ids": [],
                    "paths": [],
                },
            )
            record["fallback_similarity"] = max(record.get("fallback_similarity", 0.0), hit["similarity"])

        scored: List[Dict[str, Any]] = []
        for mitigation_id, record in mitigation_candidates.items():
            mitigation_text = self.node_search_text(self.nodes[mitigation_id])
            mitigation_similarity = max(
                safe_float(record.get("fallback_similarity")),
                token_overlap_score(query, mitigation_text),
            )
            context_fit = token_overlap_score(
                " ".join([get_anchor_step_text(flow_case, threat), threat.get("threat_mechanism", "")]),
                mitigation_text,
            )
            graph_path_score = safe_float(record.get("path_score"))
            edge_weight_score = clamp(safe_float(record.get("max_path_weight_score"), STATIC_WEIGHT_NORM), 0.0, 1.0)
            score = (
                0.29 * safe_float(record.get("source_similarity"))
                + 0.19 * mitigation_similarity
                + 0.23 * graph_path_score
                + 0.09 * context_fit
                + 0.13 * edge_weight_score
                + self.source_priority(mitigation_id)
            )
            scored.append(
                {
                    **record,
                    "score": score,
                    "score_breakdown": {
                        "source_similarity": round(safe_float(record.get("source_similarity")), 6),
                        "mitigation_similarity": round(mitigation_similarity, 6),
                        "graph_path_score": round(graph_path_score, 6),
                        "context_fit": round(context_fit, 6),
                        "evo_edge_weight_score": round(edge_weight_score, 6),
                        "source_priority": round(self.source_priority(mitigation_id), 6),
                    },
                }
            )

        scored.sort(key=lambda item: (-item["score"], item["mitigation_id"]))
        selected = scored[:top_k]

        return {
            "meta": {
                "retrieval_role": "blue_team",
                "retriever_version": VERSION,
                "generated_at_utc": now_utc(),
                "kg_dir": str(self.kg_dir),
                "use_case_id": flow_case.get("use_case_id"),
                "threat_id": threat.get("threat_id"),
                "top_k": top_k,
                "candidate_top_n": candidate_top_n,
                "fallback_top_n": fallback_top_n,
                "rerank_policy": (
                    "Threat/source node retrieval plus graph-path mitigation expansion, then deterministic "
                    "reranking over source similarity, mitigation fit, graph path strength, context fit, "
                    "critic-evolved edge weight, and source priority."
                ),
            },
            "query": truncate(query, 1800),
            "threat": {
                "threat_id": threat.get("threat_id"),
                "threat_name": threat.get("threat_name"),
                "anchor_steps": threat.get("anchor_steps") or [],
                "threat_mechanism": threat.get("threat_mechanism"),
                "security_impact": threat.get("security_impact"),
                "source_knowledge_id": threat.get("source_knowledge_id"),
            },
            "items": [self.build_blue_item(item, rank + 1) for rank, item in enumerate(selected)],
        }

    def build_blue_item(self, item: Dict[str, Any], rank: int) -> Dict[str, Any]:
        result = super().build_blue_item(item, rank)
        for path_result, path_source in zip(result.get("graph_paths") or [], item.get("paths") or []):
            if "base_path_score" in path_source:
                path_result["base_path_score"] = path_source["base_path_score"]
            if "evo_path_weight_score" in path_source:
                path_result["evo_path_weight_score"] = round(path_source["evo_path_weight_score"], 6)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rec-EvoGraph-RAG evolved retrieval v0.2.")
    parser.add_argument("--mode", choices=["red", "blue"], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--threat-id", default=None, help="For blue mode; defaults to the first threat.")
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_EVO_KG_DIR)
    parser.add_argument("--flow-path", type=Path, default=DEFAULT_FLOW_PATH)
    parser.add_argument("--threat-path", type=Path, default=DEFAULT_THREAT_PATH)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-top-n", type=int, default=40)
    parser.add_argument("--fallback-top-n", type=int, default=12)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flows = load_functional_cases(args.flow_path)
    if args.case_id not in flows:
        raise ValueError(f"Unknown case_id={args.case_id!r} in {args.flow_path}")
    flow_case = flows[args.case_id]
    retriever = RecEvoGraphRetriever(args.kg_dir)

    if args.mode == "red":
        result = retriever.retrieve_red(
            flow_case,
            top_k=args.top_k,
            candidate_top_n=args.candidate_top_n,
        )
    else:
        threat_cases = load_threat_cases(args.threat_path)
        if args.case_id not in threat_cases:
            raise ValueError(f"Unknown case_id={args.case_id!r} in {args.threat_path}")
        threat = select_threat(threat_cases[args.case_id], args.threat_id)
        result = retriever.retrieve_blue(
            flow_case,
            threat,
            top_k=args.top_k,
            candidate_top_n=args.candidate_top_n,
            fallback_top_n=args.fallback_top_n,
        )

    if args.output_path:
        write_json(args.output_path, result)
        print(f"[Done] Wrote retrieval output to {args.output_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
